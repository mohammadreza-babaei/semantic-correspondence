import argparse
import os
import csv
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from src.models import DINOv3Adapter, DINOv2Adapter, SAMAdapter, TinyViTAdapter, TinySAMAdapter, MixedModelAdapter
from src.spair_dataset import SPair71kImages, SPair71kPairs
from src.trainer import Trainer
from src.evaluator import PCKEvaluator



def create_eval_adapter(model_name, weights_path, args, device):
    """Helper to create an adapter for evaluation."""
    if "dinov2" in model_name:
        return DINOv2Adapter(
            model_name=model_name,
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks,
            weights_path=weights_path
        )
    elif "dinov3" in model_name:
        return DINOv3Adapter(
            model_name=model_name,
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks,
            weights_path=weights_path
        )
    elif "sam" in model_name:
        return SAMAdapter(
            model_name=model_name.replace("sam_", ""),
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks,
            weights_path=weights_path,
            unfreeze_neck=args.unfreeze_neck if hasattr(args, 'unfreeze_neck') else False
        )
    elif "tiny_vit" in model_name:
        return TinyViTAdapter(
            model_name='tiny_vit_21m_512.dist_in22k_ft_in1k', 
            weights_path=weights_path,
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks
        )
    elif "tiny_sam" in args.model_name:
        return TinySAMAdapter(
            model_name=args.model_name,
            weights_path=args.weights_path,
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks,
            dropout=args.dropout,
            unfreeze_neck=args.unfreeze_neck,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha
        )
    else:
        raise ValueError(f"Unknown model: {model_name}")


def main():
    parser = argparse.ArgumentParser(description="Semantic Correspondence CLI")
    parser.add_argument("--dataset-path", type=str, default=None, 
                        help="Local path or URL to the SPair-71k dataset (overrides SPAIR_URL env var)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    fine_tune_parser = subparsers.add_parser("fine_tune", help="Train the model")
    fine_tune_parser.add_argument("--model-name", type=str, default="dinov2_vits14",
                                  choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14', 
                                           'dinov3_vits16',
                                           'sam_vit_b', 'sam_vit_l', 'sam_vit_h',
                                           'tiny_vit', 'tinysam_vit_t', 'tinysam'],
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
    fine_tune_parser.add_argument("--num-augmentations", type=int, default=0,
                                  help="Number of augmented versions to cache per image (0 to disable)")
    
    # Regularization arguments
    fine_tune_parser.add_argument("--weight-decay", type=float, default=0.01,
                                  help="L2 weight decay for AdamW optimizer (default: 0.01)")
    fine_tune_parser.add_argument("--dropout", type=float, default=0.0,
                                  help="Dropout rate for unfrozen blocks (0.0 = no dropout)")
    
    # LoRA arguments
    fine_tune_parser.add_argument("--use-lora", action="store_true", help="Use LoRA instead of full fine-tuning")
    fine_tune_parser.add_argument("--lora-rank", type=int, default=8, help="LoRA rank dimension")
    fine_tune_parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA scaling factor")
    fine_tune_parser.add_argument("--feature-reg", type=float, default=0.0,
                                  help="L2 regularization weight on feature magnitudes (0.0 = disabled)")
    fine_tune_parser.add_argument("--unfreeze-neck", action="store_true",
                                  help="Unfreeze SAM neck layers for training (SAM models only)")
    fine_tune_parser.add_argument("--resume", type=str, default=None,
                                  help="Path to checkpoint to resume training from (e.g. checkpoints/best_model.pt)")
    fine_tune_parser.add_argument("--no-save-checkpoints", action="store_true",
                                  help="Disable saving checkpoints during training")
    fine_tune_parser.add_argument("--temperature", type=float, default=0.02,
                                  help="Temperature for softmax during inference (default: 0.02)")
    fine_tune_parser.add_argument("--resolution", type=int, default=None,
                                  help="Force specific input resolution (e.g. 1024 for TinySAM)")
    
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")

    # We duplicate these args because 'eval' needs to know the architecture to load
    eval_parser.add_argument("--model-name", type=str, required=True,
                             help="Model variant to evaluate (must match the checkpoint)")
    eval_parser.add_argument("--num-unfrozen-blocks", type=int, default=2,
                             help="Must match the training configuration")
    eval_parser.add_argument("--unfreeze-neck", action="store_true",
                             help="Unfreeze SAM neck layers (must match training configuration)")
    eval_parser.add_argument("--save-path", type=str, default="checkpoints/finetuned_dinov2",
                             help="Folder containing 'best_model.pt'")
    eval_parser.add_argument("--weights-path", type=str, default=None,
                             help="Explicit path to the .pt file (e.g., epoch_5.pt)")
    eval_parser.add_argument("--alpha", type=str, default="0.1",
                             help="PCK threshold factor(s) - comma-separated for multiple values (e.g., '0.1,0.05,0.01')")
    eval_parser.add_argument("--split", type=str, default="val", choices=["test", "val"], 
                             help="Dataset split to evaluate on")
    eval_parser.add_argument("--plot-pair", type=int, default=None,
                             help="Print image bbased on index")
    eval_parser.add_argument("--window-size", type=int, default=5,
                             help="Window size for local refinement in window-based prediction (default: 5)")
    eval_parser.add_argument("--temperature", type=float, default=0.02,
                             help="Temperature for softmax during inference (default: 0.02)")
    eval_parser.add_argument("--resolution", type=int, default=None,
                             help="Force specific input resolution (e.g. 1024 for TinySAM)")
    
    # Mixed Model arguments
    eval_parser.add_argument("--mix-model-name", type=str, default=None,
                             help="Second model to mix with (for evaluation only)")
    eval_parser.add_argument("--mix-weights-path", type=str, default=None,
                             help="Weights for the second model")
    eval_parser.add_argument("--mix-weights", type=str, default=None,
                             help="Weights for mixing the two models (e.g., '1.0,1.0')")
    
    args = parser.parse_args()

    # Infer model type from model name
    if "dinov2" in args.model_name:
        model_type = "dinov2"
    elif "dinov3" in args.model_name:
        model_type = "dinov3"
    elif "tinysam" in args.model_name:
        model_type = "tinysam"
    elif "sam" in args.model_name:
        model_type = "sam"
    elif "tiny_vit" in args.model_name:
        model_type = "tiny_vit"
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
    if args.use_lora:
        print(f"  LoRA Enabled: Rank={args.lora_rank}, Alpha={args.lora_alpha}")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if model_type == 'dinov2':
        fine_tuner = DINOv2Adapter(
            model_name=args.model_name,
            weights_path=args.weights_path,
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks,
            dropout=args.dropout,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha
        )

    elif model_type == 'dinov3':
        DINOV3_DOWNLOAD_URL="https://github.com/facebookresearch/dinov3"
        assert args.weights_path is not None, "Weights path must be specified for DINOV3. Download from {DINOV3_DOWNLOAD_URL}"
        fine_tuner = DINOv3Adapter(
            model_name=args.model_name,
            weights_path=args.weights_path,
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks,
            dropout=args.dropout,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha
        )
    elif model_type == 'sam':
        SAM_DOWNLOAD_URL="https://github.com/facebookresearch/segment-anything?tab=readme-ov-file#model-checkpoints"
        assert args.weights_path is not None, "Weights path must be specified for SAM. Download from {SAM_DOWNLOAD_URL}"
        fine_tuner = SAMAdapter(
            model_name=args.model_name.replace("sam_", ""),
            weights_path=args.weights_path,
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks,
            dropout=args.dropout,
            unfreeze_neck=args.unfreeze_neck,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha
        )
    elif model_type == 'tiny_vit':
        fine_tuner = TinyViTAdapter(
            model_name='tiny_vit_21m_512.dist_in22k_ft_in1k', 
            weights_path=args.weights_path,
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks
        )
    elif model_type == 'tinysam':
        fine_tuner = TinySAMAdapter(
            model_name='vit_t',
            weights_path=args.weights_path,
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks,
            dropout=args.dropout,
            use_lora=args.use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            resolution=args.resolution
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    model = Trainer(
        model=fine_tuner,
        device=device,
        num_unfrozen_blocks=args.num_unfrozen_blocks,
        learning_rate=args.lr,
        fixed_lr=args.fixed_lr,
        weight_decay=args.weight_decay,
        feature_reg=args.feature_reg,
        temperature=args.temperature,
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
        num_augmentations=args.num_augmentations,
        weight_decay=args.weight_decay,
        feature_reg=args.feature_reg,
        save_checkpoints=not args.no_save_checkpoints,
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
    adapter1 = create_eval_adapter(args.model_name, args.weights_path, args, device)

    print("Initializing architecture...")
    if "dinov2" in args.model_name:
        adapter = DINOv2Adapter(
            model_name=args.model_name,
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks,
            weights_path=args.weights_path
        )
    elif "dinov3" in args.model_name:
        adapter = DINOv3Adapter(
            model_name=args.model_name,
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks,
            weights_path=args.weights_path
        )
    elif "tinysam" in args.model_name:
        adapter = TinySAMAdapter(
            model_name='vit_t',
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks,
            weights_path=args.weights_path,
            resolution=args.resolution
        )
    elif "sam" in args.model_name:
        adapter = SAMAdapter(
            model_name=args.model_name.replace("sam_", ""),
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks,
            weights_path=args.weights_path,
            unfreeze_neck=args.unfreeze_neck if hasattr(args, 'unfreeze_neck') else False
        )
    elif "tiny_vit" in args.model_name:
        adapter = TinyViTAdapter(
            model_name='tiny_vit_21m_512.dist_in22k_ft_in1k', 
            weights_path=args.weights_path,
            device=device,
            num_unfrozen_blocks=args.num_unfrozen_blocks
        )
    else:
        raise ValueError(f"Unknown model: {args.model_name}")
    
    if args.mix_model_name:
        print(f"Initializing second model for mixing: {args.mix_model_name}")
        adapter2 = create_eval_adapter(args.mix_model_name, args.mix_weights_path, args, device)
        
        mix_weights = None
        if args.mix_weights:
            try:
                mix_weights = [float(w) for w in args.mix_weights.split(',')]
                if len(mix_weights) != 2:
                    raise ValueError("mix-weights must have exactly two values")
                print(f"Using mixing weights: {mix_weights}")
            except ValueError as e:
                print(f"Error parsing mix-weights: {e}")
                raise
                
        adapter = MixedModelAdapter(adapter1, adapter2, mix_weights=mix_weights)
        print(f"Created MixedModelAdapter: {adapter.model_name}")
    else:
        adapter = adapter1
    
    
    # 4. Initialize Trainer (Wrapper for Caching)
    trainer = Trainer(
        model=adapter,
        device=device,
        num_unfrozen_blocks=args.num_unfrozen_blocks,
        temperature=args.temperature
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
    evaluator = PCKEvaluator(trainer=trainer, device=device, window_size=args.window_size, temperature=args.temperature)

    if args.plot_pair is not None:
        test_single_sample(trainer, device, eval_dataset, args.plot_pair, window_size=args.window_size)
        return
    
    # Parse alpha values from comma-separated string to list of floats
    alphas = [float(a.strip()) for a in args.alpha.split(',')]
    alphas_str = '_'.join([str(a) for a in alphas])
    
    results_file = f"evaluations/metrics/{args.split}_results_{args.model_name}_alpha_{alphas_str}.csv"

    # 8. Print Results
    print("\n")
    print(f"Results for {args.split} with alpha values: {alphas}")
    evaluator.evaluate(eval_loader, results_file, alphas=alphas)
    PCKEvaluator.process_results(results_file, "evaluations/metrics")
    


def test_single_sample(trainer, device, dataset, sample_id, window_size=5):
    """
    Evaluates a specific sample ID using the PCKEvaluator and displays the result inline.

    Args:
        trainer: Your active Trainer instance.
        device: 'cuda' or 'cpu'.
        sample_id (int): The index of the pair to test.
        output_dir (str): Folder to save temporary visualization.
        window_size (int): Window size for local refinement.
    """
    # 1. Instantiate the Evaluator
    # Ensure this matches your import (e.g., from evaluator import PCKEvaluator)
    evaluator = PCKEvaluator(trainer, device, dataset=dataset, window_size=window_size)

    output_dir = "evaluations/pictures"
    evaluator.evaluate_pair_by_index(sample_id, output_dir=output_dir)



if __name__ == "__main__":
    main()
