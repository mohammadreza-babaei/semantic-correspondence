import argparse
import datasets
import os
from src.spair_dataset import SPair71k

def main():
    parser = argparse.ArgumentParser(description="Semantic Correspondence CLI")
    parser.add_argument("--dataset-path", type=str, default=None, 
                        help="Local path or URL to the SPair-71k dataset (overrides SPAIR_URL env var)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Train the model")
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")



    args = parser.parse_args()

    # Set dataset URL override if provided
    if args.dataset_path:
        SPair71k.set_url_override(args.dataset_path)

    if args.command == "train":
        train()
    elif args.command == "eval":
        evaluate()


def train():
    print("Training logic goes here.")
    print("Loading dataset...")
    builder = SPair71k(config_name="pairs")
    builder.download_and_prepare()
    dataset = builder.as_dataset()
    print("Dataset loaded successfully!")
    print("Dataset info:", dataset)
    print("First train example:", dataset['train'][0])

def evaluate():
    print("Evaluation logic goes here.")


if __name__ == "__main__":
    main()
