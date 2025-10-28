import argparse

def main():
    parser = argparse.ArgumentParser(description="Semantic Correspondence CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train", help="Train the model")
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")

    args = parser.parse_args()

    if args.command == "train":
        train()
    elif args.command == "eval":
        evaluate()


def train():
    print("Training logic goes here.")

def evaluate():
    print("Evaluation logic goes here.")


if __name__ == "__main__":
    main()
