import argparse
import os
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.dinov3_features import DINOv3FineTuner
from src.dinov2_features import DINOv2FeatureExtractor, DINOv2FineTuner
from src.spair_dataset import SPair71kImages, SPair71kPairs
from src.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Semantic Correspondence CLI")
    parser.add_argument("--dataset-path", type=str, default=None, 
                        help="Local path or URL to the SPair-71k dataset (overrides SPAIR_URL env var)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    fine_tune_parser = subparsers.add_parser("fine_tune", help="Train the model")
    fine_tune_parser.add_argument("--model-name", type=str, default="dinov2_vits14",
                                  choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14', 'dinov3_vits16'],
                                  help="DINOv2 model variant to use")
    fine_tune_parser.add_argument("--num-unfrozen-blocks", type=int, default=2,
                                  help="Number of transformer blocks to unfreeze (from the end)")
    fine_tune_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="Learning rate for training")
    fine_tune_parser.add_argument("--epochs", type=int, default=5,
                                  help="Number of training epochs")
    fine_tune_parser.add_argument("--batch-size", type=int, default=1,
                                  help="Batch size (typically 1 for correspondence tasks)")
    fine_tune_parser.add_argument("--log-interval", type=int, default=50,
                                  help="How often to log training progress (in batches)")
    fine_tune_parser.add_argument("--max-iters-per-epoch", type=int, default=None,
                                  help="Maximum iterations per epoch (useful for testing). None means full epoch")
    fine_tune_parser.add_argument("--save-path", type=str, default="checkpoints/finetuned_dinov2",
                                  help="Path to save model checkpoints")
    fine_tune_parser.add_argument("--no-plot", action="store_true",
                                  help="Disable interactive plotting during training")
    
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    
    args = parser.parse_args()

    if args.command == "fine_tune":
        fine_tune(args)
    elif args.command == "eval":
        evaluate()


def fine_tune(args):
    """Fine-tune Model for semantic correspondence."""
    print("="*60)
    print("Model Fine-Tuning for Semantic Correspondence")
    print("="*60)
    
    # Get dataset path
    dataset_path = args.dataset_path or os.environ.get('SPAIR_URL', './data')
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Load datasets
    print(f"\nLoading datasets from: {dataset_path}")
    train_dataset = SPair71kPairs(root=dataset_path, split='train')
    val_dataset = SPair71kPairs(root=dataset_path, split='val')
    
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    # Initialize fine-tuner
    print(f"\nInitializing Model Fine-Tuner...")
    print(f"  Model: {args.model_name}")
    print(f"  Unfrozen blocks: {args.num_unfrozen_blocks}")
    print(f"  Learning rate: {args.lr}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # model = Trainer(
    #     model=DINOv2FineTuner(
    #         model_name=args.model_name,
    #         num_unfrozen_blocks=args.num_unfrozen_blocks,
    #         device=device,
    #         learning_rate=args.lr,
    #     ),
    #     device=device,
    #     num_unfrozen_blocks=args.num_unfrozen_blocks,
    #     learning_rate=args.lr,
    # )

    model = Trainer(
    model=DINOv3FineTuner(
            model_name=args.model_name,
            num_unfrozen_blocks=args.num_unfrozen_blocks,
            device=device,
            learning_rate=args.lr,
        ),
        device=device,
        num_unfrozen_blocks=args.num_unfrozen_blocks,
        learning_rate=args.lr,
    )
    
    # Run training
    print("\nStarting training...")
    history = model.train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        epochs=args.epochs,
        batch_size=args.batch_size,
        log_interval=args.log_interval,
        save_path=args.save_path,
        plot_every_epoch=not args.no_plot,
        max_iters_per_epoch=args.max_iters_per_epoch
    )
    
    # Plot final results
    
    print("\n" + "="*60)
    print("Fine-tuning Complete!")
    print(f"Checkpoints saved to: {args.save_path}")
    print("="*60)
    

def evaluate():
    print("Evaluation logic goes here.")

if __name__ == "__main__":
    main()
