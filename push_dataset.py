import argparse
from src.script_utils import add_dict_to_argparser
from src.data_utils import get_corruption_dataset
from huggingface_hub import login

def main():
    args = create_argparser().parse_args()

    login()
    dataset = get_corruption_dataset(args.dataset_json_folder)
    dataset.push_to_hub("evgmaslov/polygons")

def create_argparser():
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, dict(dataset_json_folder = 'rplan_json'))
    return parser

if __name__ == "__main__":
    main()