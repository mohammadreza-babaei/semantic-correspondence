import argparse
import datasets
import os
import torch
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.dinov2_features import DINOv2FeatureExtractor
from src.spair_dataset import SPair71kImages

def main():
    parser = argparse.ArgumentParser(description="Semantic Correspondence CLI")
    parser.add_argument("--dataset-path", type=str, default=None, 
                        help="Local path or URL to the SPair-71k dataset (overrides SPAIR_URL env var)")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    fine_tune_parser = subparsers.add_parser("fine_tune", help="Train the model")
    eval_parser = subparsers.add_parser("eval", help="Evaluate the model")
    
    # Add extract_dinov2 subcommand
    extract_parser = subparsers.add_parser("extract_dinov2", help="Extract DINOv2 features from dataset")
    extract_parser.add_argument("--model-name", type=str, default="dinov2_vits14",
                                choices=['dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'],
                                help="DINOv2 model variant to use")
    extract_parser.add_argument("--output-dir", type=str, default="checkpoints/dinov2_features",
                                help="Directory to save extracted features")
    extract_parser.add_argument("--batch-size", type=int, default=8,
                                help="Batch size for feature extraction")
    extract_parser.add_argument("--target-size", type=int, nargs=2, default=None,
                                help="Target image size (width height). If not provided, uses original size")

    args = parser.parse_args()

    if args.command == "fine_tune":
        fine_tune()
    elif args.command == "eval":
        evaluate()
    elif args.command == "extract_dinov2":
        extract_dinov2_features(args)


def fine_tune():
    pass
    

def evaluate():
    print("Evaluation logic goes here.")


def extract_dinov2_features(args):
    """Extract DINOv2 features from all images in the dataset and save them."""
    print(f"Extracting DINOv2 features using model: {args.model_name}")
    
    # Get dataset path
    dataset_path = args.dataset_path or os.environ.get('SPAIR_URL', './data')
    if not Path(dataset_path).exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")
    
    # Initialize feature extractor
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    extractor = DINOv2FeatureExtractor(model_name=args.model_name, device=device)
    
    # Load dataset
    print(f"Loading dataset from: {dataset_path}")
    dataset = SPair71kImages(root=dataset_path)
    print(f"Found {len(dataset)} images")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process images and extract features
    features_dict = {}
    target_size = tuple(args.target_size) if args.target_size else None
    
    print(f"Extracting features (target_size={target_size})...")
    for idx in tqdm(range(len(dataset)), desc="Processing images"):
        item = dataset[idx]
        img_name = item['name']
        
        # Get image path
        img_path = Path(dataset_path) / 'SPair-71k' / 'JPEGImages' / f'{img_name}.jpg'
        
        # Preprocess and extract features
        img_tensor = extractor.preprocess_image(str(img_path), target_size=target_size)
        features = extractor.extract_features(img_tensor)
        
        # Store features (move to CPU to save memory)
        features_dict[img_name] = {
            'features': features.cpu(),
            'shape': features.shape,
            'image_size': (item['image_width'], item['image_height']),
            'category': item['category']
        }
    
    # Save checkpoint
    checkpoint_path = output_dir / f"{args.model_name}_features.pt"
    checkpoint = {
        'model_name': args.model_name,
        'patch_size': extractor.patch_size,
        'target_size': target_size,
        'num_images': len(features_dict),
        'features': features_dict,
        'device': device
    }
    
    print(f"Saving checkpoint to: {checkpoint_path}")
    torch.save(checkpoint, checkpoint_path)
    
    # Print summary
    print(f"\nFeature extraction complete!")
    print(f"  Model: {args.model_name}")
    print(f"  Images processed: {len(features_dict)}")
    print(f"  Checkpoint saved: {checkpoint_path}")
    print(f"  Checkpoint size: {checkpoint_path.stat().st_size / (1024**2):.2f} MB")


if __name__ == "__main__":
    main()
