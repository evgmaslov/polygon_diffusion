"""
This code started out as a PyTorch port of Ho et al's diffusion models:
https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py

Docstrings have been added, as well as DDIM sampling and a new collection of beta schedules.
"""

import enum
import math

import numpy as np
import torch as th
from tqdm.auto import tqdm
from train_utils import GaussianKLDivLoss, DiscreteGaussianNLLLoss
from torch import nn

def diffusion_defaults():
    return dict(
        model_mean_type="EPSILON",
        model_var_type="FIXED_LARGE",
        steps=1000,
        noise_schedule="cosine",
        timestep_respacing=None,
        rescale_timesteps=False,
        use_ddim=False,
        clip_train_denoised=False,
        clip_inference_denoised=False,
    )

class TimestepSelector():
    def __init__(self, original_num_timesteps, resampling_section_counts, rescale_timesteps, use_ddim):
        self.original_num_timesteps = original_num_timesteps
        self.resampling_section_counts = resampling_section_counts
        self.rescale_timesteps = rescale_timesteps
        self.use_ddim = use_ddim
        
        if use_ddim:
            assert len(resampling_section_counts) == 1, "section_counts must contains only one section for ddim sampling"
        self.use_timesteps = set(self.space_timesteps())
        self.timestep_map = []
        for i in range(self.original_num_timesteps):
            if i in self.use_timesteps:
                self.timestep_map.append(i)

    def space_timesteps(self):
        #section_counts - list of ints, where each int is length of the span
        """
        Create a list of timesteps to use from an original diffusion process,
        given the number of timesteps we want to take from equally-sized portions
        of the original process.

        For example, if there's 300 timesteps and the section counts are [10,15,20]
        then the first 100 timesteps are strided to be 10 timesteps, the second 100
        are strided to be 15 timesteps, and the final 100 are strided to be 20.

        If the stride is a string starting with "ddim", then the fixed striding
        from the DDIM paper is used, and only one section is allowed.

        :param num_timesteps: the number of diffusion steps in the original
                            process to divide up.
        :param section_counts: either a list of numbers, or a string containing
                            comma-separated numbers, indicating the step count
                            per section. As a special case, use "ddimN" where N
                            is a number of steps to use the striding from the
                            DDIM paper.
        :return: a set of diffusion steps from the original process to use.
        """
        if self.use_ddim:
            desired_count = self.resampling_section_counts[0]
            for i in range(1, self.original_num_timesteps):
                if len(range(0, self.original_num_timesteps, i)) == desired_count:
                    return set(range(0, self.original_num_timesteps, i))
            raise ValueError(
                f"cannot create exactly {self.original_num_timesteps} steps with an integer stride"
            )
        
        size_per = self.original_num_timesteps // len(self.resampling_section_counts)
        extra = self.original_num_timesteps % len(self.resampling_section_counts)
        start_idx = 0
        all_steps = []
        for i, section_count in enumerate(self.resampling_section_counts):
            size = size_per + (1 if i < extra else 0)
            if size < section_count:
                raise ValueError(
                    f"cannot divide section of {size} steps into {section_count}"
                )
            if section_count <= 1:
                frac_stride = 1
            else:
                frac_stride = (size - 1) / (section_count - 1)
            cur_idx = 0.0
            taken_steps = []
            for _ in range(section_count):
                taken_steps.append(start_idx + round(cur_idx))
                cur_idx += frac_stride
            all_steps += taken_steps
            start_idx += size
        return set(all_steps)
    
    def select_betas(self, base_betas):
        assert len(base_betas) == self.original_num_timesteps
        new_betas = []

        base_alphas = 1.0 - base_betas
        base_alphas_cumprod = np.cumprod(base_alphas, axis=0)

        last_alpha_cumprod = 1.0
        for i, alpha_cumprod in enumerate(base_alphas_cumprod):
            if i in self.use_timesteps:
                new_betas.append(1 - alpha_cumprod / last_alpha_cumprod)
                last_alpha_cumprod = alpha_cumprod
        new_betas = np.array(new_betas)
        return new_betas

    def select_timesteps(self, ts):
        map_tensor = th.tensor(self.timestep_map, device=ts.device, dtype=ts.dtype)
        new_ts = map_tensor[ts]
        if self.rescale_timesteps:
            new_ts = new_ts.float() * (1000.0 / self.original_num_timesteps)
        return new_ts

def get_named_beta_schedule(schedule_name, num_diffusion_timesteps):
    """
    Get a pre-defined beta schedule for the given name.

    The beta schedule library consists of beta schedules which remain similar
    in the limit of num_diffusion_timesteps.
    Beta schedules may be added, but should not be removed or changed once
    they are committed to maintain backwards compatibility.
    """
    if schedule_name == "linear":
        # Linear schedule from Ho et al, extended to work for any number of
        # diffusion steps.
        scale = 1000 / num_diffusion_timesteps
        beta_start = scale * 0.0001
        beta_end = scale * 0.02
        return np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif schedule_name == "cosine":
        return betas_for_alpha_bar(
            num_diffusion_timesteps,
            # lambda t: math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2,
            lambda t: math.cos((t) / 1.000 * math.pi / 2) ** 2,
        )
    else:
        raise NotImplementedError(f"unknown beta schedule: {schedule_name}")

def betas_for_alpha_bar(num_diffusion_timesteps, alpha_bar, max_beta=0.999):
    """
    Create a beta schedule that discretizes the given alpha_t_bar function,
    which defines the cumulative product of (1-beta) over time from t = [0,1].

    :param num_diffusion_timesteps: the number of betas to produce.
    :param alpha_bar: a lambda that takes an argument t from 0 to 1 and
                      produces the cumulative product of (1-beta) up to that
                      part of the diffusion process.
    :param max_beta: the maximum beta to use; use values lower than 1 to
                     prevent singularities.
    """
    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return np.array(betas)

def extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    res = th.from_numpy(arr).to(device=timesteps.device)[timesteps].float()
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)

class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models.

    Ported directly from here, and then adapted over time to further experimentation.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/diffusion_utils_2.py#L42

    :param noise_schedule: name of the noise schedule from the predefined set: cosine, linear.
    :param model_mean_type: a string determining the type of the mean what the model outputs from the predefined set: PREVIOUS_X (the model predicts x_{t-1}), START_X (the model predicts x_0), EPSILON (the model predicts epsilon).
    :param model_var_type: a string determining the type of the var what the model outputs from the predefined set: LEARNED, FIXED_SMALL, FIXED_LARGE, LEARNED_RANGE.
    The LEARNED_RANGE option has been added to allow the model to predict values between FIXED_SMALL and FIXED_LARGE, making its job easier.
    :param loss_type: a string determining the loss function to use from the predefined set: MSE (use raw MSE loss (and KL when learning variances)), 
    RESCALED_MSE (use raw MSE loss (with RESCALED_KL when learning variances)), KL (use the variational lower-bound), RESCALED_KL (like KL, but rescale to estimate the full VLB).
    :param rescale_timesteps: if True, pass floating point timesteps into the model so that they are always scaled like in the original paper (0 to 1000).
    :param use_ddim: use diffusion from the ddim paper.
    :param clip_denoised: if True, clip x_start predictions to [-1, 1].
    :param timestep_respacing: either a list of numbers, used to respace timestep.
    """

    def __init__(
        self,
        *,
        noise_schedule,
        steps,
        model_mean_type,
        model_var_type,
        rescale_timesteps=False,
        use_ddim=False,
        clip_train_denoised=False,
        clip_inference_denoised=True,
        timestep_respacing=None,
    ):
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.use_ddim = use_ddim
        self.clip_train_denoised = clip_train_denoised
        self.clip_inference_denoised = clip_inference_denoised

        if timestep_respacing is not list:
            timestep_respacing = [steps]
        self.timestep_selector = TimestepSelector(steps, timestep_respacing, rescale_timesteps, use_ddim)

        # Use float64 for accuracy.
        betas = np.array(get_named_beta_schedule(noise_schedule, steps), dtype=np.float64)
        self.betas = self.timestep_selector.select_betas(betas)
        assert len(betas.shape) == 1, "betas must be 1-D"
        assert (betas > 0).all() and (betas <= 1).all()

        self.num_timesteps = int(betas.shape[0])

        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)
        self.alphas_cumprod_prev = np.append(1.0, self.alphas_cumprod[:-1])
        self.alphas_cumprod_next = np.append(self.alphas_cumprod[1:], 0.0)
        assert self.alphas_cumprod_prev.shape == (self.num_timesteps,)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = np.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = np.sqrt(1.0 / self.alphas_cumprod - 1)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        # log calculation clipped because the posterior variance is 0 at the
        # beginning of the diffusion chain.
        self.posterior_log_variance_clipped = np.log(
            np.append(self.posterior_variance[1], self.posterior_variance[1:])
        )
        self.posterior_mean_coef1 = (
            betas * np.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev)
            * np.sqrt(alphas)
            / (1.0 - self.alphas_cumprod)
        )
    
    def denoise(
        self,
        model,
        shape,
        noise=None,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Generate samples from the model.

        :param model: the model module.
        :param shape: the shape of the samples, (N, C, H, W).
        :param noise: if specified, the noise from the encoder to sample.
                      Should be of the same shape as `shape`.
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a non-differentiable batch of samples.
        """
        device = next(model.parameters()).device
        assert isinstance(shape, (tuple, list))
        img = th.randn(*shape, device=device)
        if noise is not None:
            img = noise
        timesteps = list(range(self.num_timesteps))[::-1]
        myfinal = []
        for ind, i in tqdm(enumerate(timesteps), total=len(timesteps)):
            t = th.tensor([i] * shape[0], device=device)
            with th.no_grad():
                out = self.p_sample(
                    model,
                    img,
                    t,
                    denoised_fn=denoised_fn,
                    cond_fn=cond_fn,
                    model_kwargs=model_kwargs,
                )
            img = out["sample"]
            if ind>970:
                myfinal.append(img)
        return img 

    def p_sample(
        self,
        model,
        x,
        t,
        denoised_fn=None,
        cond_fn=None,
        model_kwargs=None,
    ):
        """
        Sample x_{t-1} from the model at the given timestep.

        :param model: the model to sample from.
        :param x: the current tensor at x_{t-1}.
        :param t: the value of t, starting at 0 for the first diffusion step.
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample.
        :param cond_fn: if not None, this is a gradient function that acts
                        similarly to the model.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict containing the following keys:
                 - 'sample': a random sample from the model.
                 - 'pred_xstart': a prediction of x_0.
        """
        model_output = self.get_model_output(model, x, t, model_kwargs)
        mean, variance, log_variance, pred_xstart = self.p_mean_variance(
            model_output,
            x,
            t,
            clip_denoised=self.clip_inference_denoised,
            denoised_fn=denoised_fn,
        )

        noise = th.randn_like(x)

        if not self.use_ddim:
            if cond_fn is not None:
                mean = self.condition_mean(cond_fn, out, x, t, model_kwargs=model_kwargs)
            sample = mean + th.exp(0.5 * log_variance) * noise if t[0] != 0 else mean
        else:
            # Usually our model outputs epsilon, but we re-derive it
            # in case we used x_start or x_prev prediction.
            if cond_fn is not None:
                out = self.condition_score(cond_fn, out, x, t, model_kwargs=model_kwargs)
            eps = self.predict_eps_from_xstart(x, t, pred_xstart)

            eta=0
            alpha_bar = extract_into_tensor(self.alphas_cumprod, t, x.shape)
            alpha_bar_prev = extract_into_tensor(self.alphas_cumprod_prev, t, x.shape)
            sigma = eta * th.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar)) * th.sqrt(1 - alpha_bar / alpha_bar_prev)
            # Equation 12.
            noise = th.randn_like(x)
            mean_pred = pred_xstart * th.sqrt(alpha_bar_prev) + th.sqrt(1 - alpha_bar_prev - sigma ** 2) * eps

            sample = mean_pred + sigma * noise if t != 0 else mean_pred

        return {"sample": sample, "pred_xstart": pred_xstart}
    
    def q_mean_variance(self, x_start, t):
        """
        Get the distribution q(x_t | x_0).

        :param x_start: the [N x C x ...] tensor of noiseless inputs.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :return: A tuple (mean, variance, log_variance), all of x_start's shape.
        """
        mean = (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        )
        variance = extract_into_tensor(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract_into_tensor(
            self.log_one_minus_alphas_cumprod, t, x_start.shape
        )
        return mean, variance, log_variance

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data for a given number of diffusion steps.

        In other words, sample from q(x_t | x_0).

        :param x_start: the initial data batch.
        :param t: the number of diffusion steps (minus 1). Here, 0 means one step.
        :param noise: if specified, the split-out normal noise.
        :return: A noisy version of x_start.
        """
        if noise is None:
            noise = th.randn_like(x_start)
        assert noise.shape == x_start.shape
        return (
            extract_into_tensor(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            * noise
        )

    def q_posterior_mean_variance(self, x_start, x_t, t):
        """
        Compute the mean and variance of the diffusion posterior:

            q(x_{t-1} | x_t, x_0)

        """
        assert x_start.shape == x_t.shape
        posterior_mean = (
            extract_into_tensor(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract_into_tensor(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract_into_tensor(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract_into_tensor(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        assert (
            posterior_mean.shape[0]
            == posterior_variance.shape[0]
            == posterior_log_variance_clipped.shape[0]
            == x_start.shape[0]
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def get_model_output(self, model, x, t, model_kwargs=None):
        B, C = x.shape[:2]
        assert t.shape == (B,)

        model_output = model(x, self.timestep_selector.select_timesteps(t), **model_kwargs)
        
        return model_output

    def p_mean_variance(
        self, model_output, x, t, clip_denoised, denoised_fn=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.

        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        B, C = x.shape[:2]
        assert t.shape == (B,)

        if self.model_var_type == "LEARNED":
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            model_log_variance = model_var_values
            model_variance = th.exp(model_log_variance)
        elif self.model_var_type == "LEARNED_RANGE":
            assert model_output.shape == (B, C * 2, *x.shape[2:])
            model_output, model_var_values = th.split(model_output, C, dim=1)
            min_log = extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
            max_log = extract_into_tensor(np.log(self.betas), t, x.shape)
            # The model_var_values is [-1, 1] for [min_var, max_var].
            frac = (model_var_values + 1) / 2
            model_log_variance = frac * max_log + (1 - frac) * min_log
            model_variance = th.exp(model_log_variance)
        elif self.model_var_type == "FIXED_SMALL":
            model_variance = extract_into_tensor(self.posterior_variance, t, x.shape)
            model_log_variance = extract_into_tensor(self.posterior_log_variance_clipped, t, x.shape)
        elif self.model_var_type == "FIXED_LARGE":
            model_variance = np.append(self.posterior_variance[1], self.betas[1:])
            model_log_variance = np.log(np.append(self.posterior_variance[1], self.betas[1:]))
            model_variance = extract_into_tensor(model_variance, t, x.shape)
            model_log_variance = extract_into_tensor(model_log_variance, t, x.shape)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x
        
        if self.model_mean_type == "PREVIOUS_X":
            pred_xstart = process_xstart(self.predict_xstart_from_xprev(x_t=x, t=t, xprev=model_output))
            model_mean = model_output
        elif self.model_mean_type == "START_X":
            pred_xstart = process_xstart(model_output)
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        elif self.model_mean_type == 'EPSILON':
            pred_xstart = process_xstart(self.predict_xstart_from_eps(x_t=x, t=t, eps=model_output))
            model_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_xstart, x_t=x, t=t)
        else:
            raise NotImplementedError(self.model_mean_type)

        assert (model_mean.shape == model_log_variance.shape == pred_xstart.shape == x.shape)
        return model_mean, model_variance, model_log_variance, pred_xstart

    def predict_xstart_from_eps(self, x_t, t, eps):
        assert x_t.shape == eps.shape
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps
        )

    def predict_xstart_from_xprev(self, x_t, t, xprev):
        assert x_t.shape == xprev.shape
        return (  # (xprev - coef2*x_t) / coef1
            extract_into_tensor(1.0 / self.posterior_mean_coef1, t, x_t.shape) * xprev
            - extract_into_tensor(
                self.posterior_mean_coef2 / self.posterior_mean_coef1, t, x_t.shape
            )
            * x_t
        )

    def predict_eps_from_xstart(self, x_t, t, pred_xstart):
        return (
            extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - pred_xstart
        ) / extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def condition_mean(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute the mean for the previous step, given a function cond_fn that
        computes the gradient of a conditional log probability with respect to
        x. In particular, cond_fn computes grad(log(p(y|x))), and we want to
        condition on y.

        This uses the conditioning strategy from Sohl-Dickstein et al. (2015).
        """
        gradient = cond_fn(x, self.timestep_selector.select_timesteps(t), **model_kwargs)
        new_mean = p_mean_var["mean"].float() + p_mean_var["variance"] * gradient.float()
        return new_mean

    def condition_score(self, cond_fn, p_mean_var, x, t, model_kwargs=None):
        """
        Compute what the p_mean_variance output would have been, should the
        model's score function be conditioned by cond_fn.

        See condition_mean() for details on cond_fn.

        Unlike condition_mean(), this instead uses the conditioning strategy
        from Song et al (2020).
        """
        alpha_bar = extract_into_tensor(self.alphas_cumprod, t, x.shape)

        eps = self.predict_eps_from_xstart(x, t, p_mean_var["pred_xstart"])
        eps = eps - (1 - alpha_bar).sqrt() * cond_fn(x, self.timestep_selector.select_timesteps(t), **model_kwargs)

        out = p_mean_var.copy()
        out["pred_xstart"] = self.predict_xstart_from_eps(x, t, eps)
        out["mean"], _, _ = self.q_posterior_mean_variance(
            x_start=out["pred_xstart"], x_t=x, t=t
        )
        return out
