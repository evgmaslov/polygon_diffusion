import argparse
from script_utils import add_dict_to_argparser
from diffusion_utils import diffusion_defaults, GaussianDiffusion
from model_utils import PolygonDiffusionModel, PolygonTransformerModel, polygon_model_defaults
from train_utils import train_defaults, ImageCallback
from data_utils import get_corruption_dataset, corruption_collator
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

def main():
    args = create_argparser().parse_args()

    diffusion_model = PolygonDiffusionModel(polygon_model_defaults(), guided=True)

    dataset = load_dataset("evgmaslov/polygons")
    dataset = dataset.train_test_split(test_size=args.test_size, seed=args.random_seed)

    training_params = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.n_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        save_steps=args.save_steps, 
        learning_rate=args.lr, weight_decay=args.weight_decay,
        fp16=False, bf16=False,
        report_to="wandb",
        logging_steps=args.logging_steps,
        save_total_limit=2,
        eval_strategy=args.eval_strategy,
        eval_steps=args.eval_steps,
        push_to_hub=args.push_to_hub,
        hub_strategy=args.hub_strategy,
        remove_unused_columns=False)
    trainer = Trainer(
        model=diffusion_model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=corruption_collator,
        args=training_params)
    wandb_callback = ImageCallback(trainer, dataset["test"], num_samples=2)
    trainer.add_callback(wandb_callback)
    trainer.train()


def create_argparser():
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, train_defaults())
    return parser

if __name__ == "__main__":
    main()