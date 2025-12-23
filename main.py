import argparse
import os
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.dinov3_features import DINOv3FineTuner
from src.dinov2_features import DINOv2FeatureExtractor, DINOv2FineTuner
from src.sam_features import SAMFeatureExtractor, SAMFineTuner
from src.spair_dataset import SPair71kImages, SPair71kPairs
from src.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Semantic Correspondence CLI")
    parser.add_argument("--dataset-path", type=str, default=None, 
                        help="Local path or URL to the SPair-71k dataset (overrides SPAIR_URL env var)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    fine_tune_parser = subparsers.add_parser("fine_tune", help="Train the model")
    fine_tune_parser.add_argument("--model-name", type=str, default="dinov2_vits14",
                                  choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14', 
                                           'dinov3_vits16',
                                           'facebook/sam-vit-base', 'facebook/sam-vit-large', 'facebook/sam-vit-huge'],
                                  help="Model variant to use (DINOv2, DINOv3, or SAM)")
    fine_tune_parser.add_argument("--num-unfrozen-blocks", type=int, default=2,
                                  help="Number of transformer blocks to unfreeze (from the end)")
    fine_tune_parser.add_argument("--lr", type=float, default=1e-3,
                                  help="Learning rate for training")
    fine_tune_parser.add_argument("--fixed-lr", action="store_true",
                                  help="Use fixed learning rate")
    fine_tune_parser.add_argument("--epochs", type=int, default=5,
                                  help="Number of training epochs")
    fine_tune_parser.add_argument("--batch-size", type=int, default=1,
                                  help="Batch size (typically 1 for correspondence tasks)")
    fine_tune_parser.add_argument("--log-interval", type=int, default=50,
                                  help="How often to log training progress (in batches)")
    fine_tune_parser.add_argument("--max-iters-per-epoch", type=int, default=None,
                                  help="Maximum iterations per epoch (useful for testing). None means full epoch")
    fine_tune_parser.add_argument("--save-path", type=str, default=None,
                                  help="Path to save model checkpoints")
    fine_tune_parser.add_argument("--weights-path", type=str, default=None,
                                  help="Path to custom model weights (.safetensors or .pth file)")
    fine_tune_parser.add_argument("--no-plot", action="store_true",
                                  help="Disable interactive plotting during training")
    
    # WandB arguments
    fine_tune_parser.add_argument("--use-wandb", action="store_true",
                                  help="Enable Weights & Biases logging")
    fine_tune_parser.add_argument("--accumulation-steps", type=int, default=1,
                                  help="Number of steps to accumulate gradients before updating optimizer")
    fine_tune_parser.add_argument("--wandb-project", type=str, default="semantic_correspondence",
                                  help="WandB project name")
    fine_tune_parser.add_argument("--wandb-run-name", type=str, default=None,
                                  help="WandB run name (optional)")
    
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    
    args = parser.parse_args()

    # Infer model type from model name
    if "dinov2" in args.model_name:
        model_type = "dinov2"
    elif "dinov3" in args.model_name:
        model_type = "dinov3"
    elif "sam" in args.model_name:
        model_type = "sam"
    else:
        # Fallback or error, though choices constraint handles most valid cases
        if "dino" in args.model_name:
             model_type = "dinov2" # Default assumption for legacy
        else:
             raise ValueError(f"Could not infer model type from name: {args.model_name}")

    # Set default save path if not provided
    if args.save_path is None:
        args.save_path = f"checkpoints/finetuned_{model_type}"

    if args.command == "fine_tune":
        fine_tune(args, model_type)
    elif args.command == "eval":
        evaluate()


def fine_tune(args, model_type):
    """Fine-tune the selected model for semantic correspondence."""
    model_display_name = model_type.upper()
    print("="*60)
    print(f"{model_display_name} Fine-Tuning for Semantic Correspondence")
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
    print(f"\nInitializing {model_display_name} Fine-Tuner...")
    print(f"  Model: {args.model_name}")
    print(f"  Unfrozen blocks: {args.num_unfrozen_blocks}")
    print(f"  Learning rate: {args.lr}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_type == 'dinov2':
        fine_tuner = DINOv2FineTuner(
            model_name=args.model_name,
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks,
            learning_rate=args.lr,
        )
        if args.weights_path:
             print("Warning: Custom weights path provided but not supported for DINOv2 yet.")

    elif model_type == 'dinov3':
        fine_tuner = DINOv3FineTuner(
            model_name=args.model_name,
            weights_path=args.weights_path,
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks,
            learning_rate=args.lr,
        )
    elif model_type == 'sam':
        fine_tuner = SAMFineTuner(
            model_name=args.model_name,
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks,
            learning_rate=args.lr,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = Trainer(
        model=fine_tuner,
        device=device,
        num_unfrozen_blocks=args.num_unfrozen_blocks,
        learning_rate=args.lr,
        fixed_lr=args.fixed_lr,
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
        max_iters_per_epoch=args.max_iters_per_epoch,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
        accumulation_steps=args.accumulation_steps
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
