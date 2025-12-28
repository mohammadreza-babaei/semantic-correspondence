import argparse
import os
import csv
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.models import DINOv3Adapter, DINOv2Adapter, SAMAdapter
from src.spair_dataset import SPair71kImages, SPair71kPairs
from src.trainer import Trainer
from src.pck import compute_raw_distances
from src.evaluator import PCKEvaluator
from IPython.display import Image, display


def main():
    parser = argparse.ArgumentParser(description="Semantic Correspondence CLI")
    parser.add_argument("--dataset-path", type=str, default=None, 
                        help="Local path or URL to the SPair-71k dataset (overrides SPAIR_URL env var)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    fine_tune_parser = subparsers.add_parser("fine_tune", help="Train the model")
    fine_tune_parser.add_argument("--model-name", type=str, default="dinov2_vits14",
                                  choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14', 
                                           'dinov3_vits16',
                                           'sam_vit_b', 'sam_vit_l', 'sam_vit_h'],
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
    fine_tune_parser.add_argument("--no-shuffle", action="store_true",
                                  help="Disable dataset shuffling during training")
    
    # WandB arguments
    fine_tune_parser.add_argument("--use-wandb", action="store_true",
                                  help="Enable Weights & Biases logging")
    fine_tune_parser.add_argument("--accumulation-steps", type=int, default=1,
                                  help="Number of steps to accumulate gradients before updating optimizer")
    fine_tune_parser.add_argument("--wandb-project", type=str, default="semantic_correspondence",
                                  help="WandB project name")
    fine_tune_parser.add_argument("--wandb-run-name", type=str, default=None,
                                  help="WandB run name (optional)")
    fine_tune_parser.add_argument("--num-augmentations", type=int, default=3,
                                  help="Number of augmented versions to cache per image (0 to disable)")
    fine_tune_parser.add_argument("--resume", type=str, default=None,
                                  help="Path to checkpoint to resume training from (e.g. checkpoints/best_model.pt)")
    
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")

    # We duplicate these args because 'eval' needs to know the architecture to load
    eval_parser.add_argument("--model-name", type=str, required=True,
                             help="Model variant to evaluate (must match the checkpoint)")
    eval_parser.add_argument("--num-unfrozen-blocks", type=int, default=2,
                             help="Must match the training configuration")
    eval_parser.add_argument("--save-path", type=str, default="checkpoints/finetuned_dinov2",
                             help="Folder containing 'best_model.pt'")
    eval_parser.add_argument("--weights-path", type=str, default=None,
                             help="Explicit path to the .pt file (e.g., epoch_5.pt)")
    eval_parser.add_argument("--alpha", type=float, default=0.1,
                             help="PCK threshold factor")
    eval_parser.add_argument("--split", type=str, default="val", choices=["test", "val"], 
                             help="Dataset split to evaluate on")
    eval_parser.add_argument("--plot-pair", type=int, default=42,
                             help="Print image bbased on index")
    
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
        evaluate(args)


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
        fine_tuner = DINOv2Adapter(
            model_name=args.model_name,
            weights_path=args.weights_path,
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks
        )

    elif model_type == 'dinov3':
        DINOV3_DOWNLOAD_URL="https://github.com/facebookresearch/dinov3"
        assert args.weights_path is not None, "Weights path must be specified for DINOV3. Download from {DINOV3_DOWNLOAD_URL}"
        fine_tuner = DINOv3Adapter(
            model_name=args.model_name,
            weights_path=args.weights_path,
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks
        )
    elif model_type == 'sam':
        SAM_DOWNLOAD_URL="https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints"
        assert args.weights_path is not None, "Weights path must be specified for SAM. Download from {SAM_DOWNLOAD_URL}"
        fine_tuner = SAMAdapter(
            model_name=args.model_name.replace("sam_", ""),
            weights_path=args.weights_path,
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks
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
    
    # Resume training if requested
    if args.resume:
        print(f"\nResuming training from: {args.resume}")
        model.load_checkpoint(args.resume)
    
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
        accumulation_steps=args.accumulation_steps,
        shuffle=not args.no_shuffle,
        num_augmentations=args.num_augmentations
    )
    
    # Plot final results
    
    print("\n" + "="*60)
    print("Fine-tuning Complete!")
    print(f"Checkpoints saved to: {args.save_path}")
    print("="*60)
    


def evaluate(args):
    """
    Standalone evaluation function using PCKEvaluator with Adapter support.
    """
    if args is None: return

    print("="*60)
    print(f"STARTING EVALUATION: {args.model_name}")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    
    # 1. Load Data
    dataset_path = args.dataset_path or os.environ.get('SPAIR_URL', './data')
    eval_dataset = SPair71kPairs(root=dataset_path, split=args.split)
    
    # 2. Initialize Adapter
    # IMPORTANT: Initialize with weights_path=None. 
    # We do NOT want to load the base weights here; we will overwrite them 
    # with the fine-tuned checkpoint in step 3.
    
    print("Initializing architecture...")
    if "dinov2" in args.model_name:
        adapter = DINOv2Adapter(
            model_name=args.model_name,
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks,
            weights_path=None
        )
    elif "dinov3" in args.model_name:
        adapter = DINOv3Adapter(
            model_name=args.model_name,
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks,
            weights_path=None
        )
    elif "sam" in args.model_name:
        adapter = SAMAdapter(
            model_name=args.model_name.replace("sam_", ""),
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks,
            weights_path=None
        )
    else:
        raise ValueError(f"Unknown model: {args.model_name}")
    
    # 3. Load Trained Weights (The Fine-Tuned Checkpoint)
    # This is where we load epoch.pt
    if args.weights_path and os.path.exists(args.weights_path):
        print(f"Loading fine-tuned checkpoint from: {args.weights_path}")
        checkpoint = torch.load(args.weights_path, map_location=device, weights_only=False)
        
        # Determine how to unwrap the checkpoint
        if isinstance(checkpoint, dict):
            if 'model_state' in checkpoint:
                adapter.load_model_state(checkpoint['model_state'])
            elif 'backbone_state_dict' in checkpoint:
                adapter.load_model_state(checkpoint)
            elif 'sam_model_state_dict' in checkpoint:
                adapter.load_model_state(checkpoint)
            else:
                # Fallback: assume the dict itself is the state dict
                adapter.load_model_state(checkpoint)
        else:
             print("Error: Checkpoint format not recognized (expected dict).")
    else:
        print(f"Warning: Checkpoint not found at {args.weights_path}. Using base/random weights.")

    # 4. Initialize Trainer (Wrapper for Caching)
    trainer = Trainer(
        model=adapter,
        device=device,
        num_unfrozen_blocks=args.num_unfrozen_blocks
    )

    # 5. Extract Features
    print(f"\nPre-extracting features for {args.split.capitalize()} Set...")
    trainer.cache_intermediate_features(eval_dataset, num_augmentations=0)
    
    # 6. Create Dataloader
    eval_loader = DataLoader(
        eval_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=0, 
        collate_fn=trainer._collate_fn
    )

    
    # 7. Run Evaluation
    evaluator = PCKEvaluator(trainer=trainer, device=device)

    if args.plot_pair:
        test_single_sample(trainer, device, eval_dataset, args.plot_pair,)
        return
    
    results_file = f"evaluations/metrics/{args.split}_results_{args.model_name}_alpha{args.alpha}.csv"
    
    test_single_sample(trainer, device, eval_dataset, sample_id=42)
    
    evaluator.evaluate(eval_loader, results_file, alpha=args.alpha)
    
    # 8. Print Summary
    evaluator.summarize_results(results_file, args.alpha)



def test_single_sample(trainer, device, dataset, sample_id):
    """
    Evaluates a specific sample ID using the PCKEvaluator and displays the result inline.

    Args:
        trainer: Your active Trainer instance.
        device: 'cuda' or 'cpu'.
        sample_id (int): The index of the pair to test.
        output_dir (str): Folder to save temporary visualization.
    """
    # 1. Instantiate the Evaluator
    # Ensure this matches your import (e.g., from evaluator import PCKEvaluator)
    evaluator = PCKEvaluator(trainer, device, dataset=dataset)

    output_dir = "evaluations/pictures"
    evaluator.evaluate_pair_by_index(sample_id, output_dir=output_dir)

    expected_filename = f"pair_{sample_id}_comparison.png"
    img_path = os.path.join(output_dir, expected_filename)


    if os.path.exists(img_path):
        print("\n")
        print("="*40)
        print(f"VISUALIZATION (Sample {sample_id})")
        print("="*40)
        display(Image(filename=img_path, width=800))
    else:
        print(f"Error: Image not found at {img_path}. Check if evaluate_pair_by_index ran correctly.")


if __name__ == "__main__":
    main()
