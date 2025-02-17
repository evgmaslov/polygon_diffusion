from abc import ABC, abstractmethod

import numpy as np
import torch

from torch import nn
from transformers import GenerationConfig
from transformers.integrations import WandbCallback
import random
import wandb
from .data_utils import polygon_to_mask, corruption_collator, model_outputs_to_reconstructions
from PIL import Image
from tqdm.auto import tqdm

def train_defaults():
    defaults = dict(
        dataset_json_folder = 'rplan_json',
        loss_type = "MSE",
        lr=1e-4,
        weight_decay=0.0,
        batch_size=1024,
        n_epochs=40,
        test_size=0.1,
        random_seed=42,
        output_dir="",
        save_steps=100,
        eval_strategy="steps",
        eval_steps=100,
        logging_steps=20,
        push_to_hub=False,
        hub_strategy="checkpoint"
    )
    return defaults

class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long()
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float()
        return indices, weights

class UniformSampler(ScheduleSampler):
    def __init__(self, diffusion):
        self.diffusion = diffusion
        self._weights = np.ones([diffusion.num_timesteps])

    def weights(self):
        return self._weights

class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, local_ts, local_losses):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        batch_sizes = [torch.tensor([0], dtype=torch.int32, device=local_ts.device)]

        # Pad all_gather batches to be the maximum batch size.
        batch_sizes = [x.item() for x in batch_sizes]
        max_bs = max(batch_sizes)

        timestep_batches = [torch.zeros(max_bs).to(local_ts) for bs in batch_sizes]
        loss_batches = [torch.zeros(max_bs).to(local_losses) for bs in batch_sizes]
        timesteps = [
            x.item() for y, bs in zip(timestep_batches, batch_sizes) for x in y[:bs]
        ]
        losses = [x.item() for y, bs in zip(loss_batches, batch_sizes) for x in y[:bs]]
        self.update_with_all_losses(timesteps, losses)

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """

class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, diffusion, history_per_term=10, uniform_prob=0.001):
        self.diffusion = diffusion
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [diffusion.num_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([diffusion.num_timesteps], dtype=np.int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.diffusion.num_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history ** 2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()

class GaussianKLDivLoss(nn.Module):
    def __init__(self, reduce=False):
        super().__init__()
        self.reduce = reduce
    def forward(self, mean1, logvar1, mean2, logvar2):
        kl = 0.5 * (
            -1.0
            + logvar2
            - logvar1
            + torch.exp(logvar1 - logvar2)
            + ((mean1 - mean2) ** 2) * torch.exp(-logvar2)
        )
        if self.reduce:
            return kl.mean()
        else:
            return kl

class DiscreteGaussianNLLLoss(nn.Module):
    def __init__(self, reduce=False):
        super().__init__()
        self.reduce = reduce
    def approx_standard_normal_cdf(self, x):
        """
        A fast approximation of the cumulative distribution function of the
        standard normal.
        """
        return 0.5 * (1.0 + th.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * th.pow(x, 3))))
    def forward(self, x_true, mean, logvar):
        assert x_true.shape == mean.shape == logvar.shape
        centered_x = x_true - mean
        
        inv_stdv = th.exp(-logvar)
        plus_in = inv_stdv * (centered_x + 1.0 / 255.0)
        cdf_plus = self.approx_standard_normal_cdf(plus_in)

        min_in = inv_stdv * (centered_x - 1.0 / 255.0)
        cdf_min = self.approx_standard_normal_cdf(min_in)

        log_cdf_plus = th.log(cdf_plus.clamp(min=1e-12))
        log_one_minus_cdf_min = th.log((1.0 - cdf_min).clamp(min=1e-12))
        cdf_delta = cdf_plus - cdf_min

        log_probs = th.where(
            x_true < -0.999,
            log_cdf_plus,
            th.where(x_true > 0.999, log_one_minus_cdf_min, th.log(cdf_delta.clamp(min=1e-12))),
        )
        assert log_probs.shape == x_true.shape
        nll = -log_probs
        if self.reduce:
            return nll.mean()
        else:
            return nll

class ImageCallback(WandbCallback):
    def __init__(self, trainer, test_dataset, num_samples=2):
        super().__init__()
        self.sample_dataset = test_dataset.select(random.choices(range(len(test_dataset)), k=num_samples))
        self.model = trainer.model
        self.inference_model = None

    def reconstruct_polygons(self, examples):
        inputs = corruption_collator(examples)
        inputs = {k: inputs[k].to("cuda") for k in inputs.keys()}
        geometry_conditions = {k: inputs[k] for k in inputs.keys() if k != "shift"}
        shape = inputs["shift"].shape
        shift = self.inference_model.generate(shape, geometry_conditions)
        reconstructions = model_outputs_to_reconstructions(inputs, shift)
        return reconstructions

    def samples_table(self, examples):
        corrupt_images = []
        for example in examples:
            corrupt = example["corrupted_polygons"]
            corrupt = [[np.expand_dims(np.array(v), axis=1) for v in np.array(polygon).T.tolist()] for polygon in corrupt]
            flat_mask = np.zeros((256, 256))
            for polygon in corrupt:
                polygon_mask = polygon_to_mask(polygon)
                flat_mask[np.where(polygon_mask == 1)] += 1
            corrupt_image = Image.fromarray(50*flat_mask.astype(np.uint8))
            corrupt_images.append(corrupt_image)
        
        reconstructed_polygons = self.reconstruct_polygons(examples)
        reconstructed_images = []
        for polygon_set in reconstructed_polygons:
            polygons = [[np.expand_dims(np.array(v), axis=1) for v in np.array(polygon).T.tolist()] for polygon in polygon_set]
            flat_mask = np.zeros((256, 256))
            for polygon in polygons:
                polygon_mask = polygon_to_mask(polygon)
                flat_mask[np.where(polygon_mask == 1)] += 1
            reconstructed_image = Image.fromarray(50*flat_mask.astype(np.uint8))
            reconstructed_images.append(reconstructed_image)

        wandb.log({"corrupted_polygons": [wandb.Image(image) for image in corrupt_images], "reconstructed_polygons": [wandb.Image(image) for image in reconstructed_images]})

    def on_evaluate(self, args, state, control,  **kwargs):
        super().on_evaluate(args, state, control, **kwargs)
        self.inference_model = self.model
        self.samples_table(self.sample_dataset)