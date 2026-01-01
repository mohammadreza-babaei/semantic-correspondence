"""Cached Features Dataset for efficient training with pre-extracted features.

This module provides a PyTorch Dataset wrapper that caches intermediate model features
to disk, enabling faster training by avoiding redundant feature extraction.
"""

import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import gc

from .spair_dataset import SPair71kImages


class GaussianNoise:
    """Add Gaussian noise to a tensor image."""
    def __init__(self, std=0.05):
        self.std = std
    
    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0, 1)


def get_position_preserving_augmentations():
    """Return a list of position-preserving augmentations.
    
    These augmentations only modify pixel values, not spatial positions,
    so keypoint coordinates remain valid.
    
    Note: All transforms must work with float32 tensors (range 0-1).
    """
    return T.Compose([
        T.RandomApply([
            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
        ], p=0.8),
        T.RandomApply([
            T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        ], p=0.3),
        T.RandomGrayscale(p=0.2),
        T.RandomApply([GaussianNoise(std=0.05)], p=0.2),
        T.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
    ])


class CachedFeaturesDataset(Dataset):
    """Dataset wrapper that caches intermediate model features to disk.
    
    This class wraps an SPair71kPairs dataset and extracts intermediate features
    (output of frozen transformer blocks) that can be reused across training runs.
    
    Features are cached per-image (not per-pair) for efficiency, since images
    may appear in multiple pairs.
    
    Args:
        base_dataset: SPair71kPairs dataset to wrap
        model: Model adapter with extract_intermediate_features() method
        cache_dir: Directory to store cached features (default: checkpoints/feature_cache)
        num_augmentations: Number of augmented versions to cache per image
    """
    
    def __init__(
        self,
        base_dataset,
        model,
        cache_dir=None,
        num_augmentations=3,
    ):
        self.base_dataset = base_dataset
        self.model = model
        self.num_augmentations = num_augmentations
        self.device = next(model.model.parameters()).device
        
        # Build cache directory path
        if cache_dir is None:
            model_cache_name = f"{model.model_name.replace('/', '_')}_intermediate_{model.num_frozen_blocks}frozen"
            self.cache_dir = Path('checkpoints') / 'feature_cache' / model_cache_name
        else:
            self.cache_dir = Path(cache_dir)
        
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.augment_transform = get_position_preserving_augmentations()
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        """Return base dataset item with image names for feature loading.
        
        The actual feature loading is done by the Trainer using load_features().
        This keeps the Dataset lightweight and avoids loading features that might
        not be needed (e.g., if we're doing augmentation selection).
        """
        return self.base_dataset[idx]
    
    def load_features(self, img_name, aug_idx=None):
        """Load cached features for a single image from disk.
        
        Args:
            img_name: Image name in format 'category/image_stem'
            aug_idx: Optional augmentation index (None for original)
            
        Returns:
            dict with 'intermediate' tensor and 'orig_size' tuple
        """
        if aug_idx is not None:
            cache_path = self.cache_dir / f"{img_name.replace('/', '_')}_aug{aug_idx}.pt"
        else:
            cache_path = self.cache_dir / f"{img_name.replace('/', '_')}.pt"
        
        if not cache_path.exists():
            raise RuntimeError(f"Cached features not found for {img_name}. "
                             f"Expected file: {cache_path}")
        return torch.load(cache_path, map_location='cpu')
    
    def cache_features(self):
        """Extract and cache all intermediate features for the dataset.
        
        Memory-efficient approach: saves each image's features immediately to
        individual files instead of accumulating all in RAM.
        """
        # Collect ALL unique images from the JPEGImages directory (all splits)
        print("Scanning all images in JPEGImages directory...")
        # Use the factory method from the base dataset to list all images
        temp_images = self.base_dataset.get_images_dataset()
        unique_images = temp_images.list_all_images()
        
        print(f"Found {len(unique_images)} unique images across all splits")



        
        # Pre-scan cache directory to avoid thousands of file system calls
        print(f"Scanning cache directory: {self.cache_dir}")
        existing_files = set()
        if self.cache_dir.exists():
            for f in self.cache_dir.glob('*.pt'):
                existing_files.add(f.name)
        print(f"Found {len(existing_files)} already cached files.")
        
        # Count missing
        total_original = len(unique_images)
        total_augmented = len(unique_images) * self.num_augmentations
        
        missing_original_count = 0
        missing_augmented_count = 0
        
        for img_name in unique_images:
            fname = f"{img_name.replace('/', '_')}.pt"
            if fname not in existing_files:
                missing_original_count += 1
            
            for aug_idx in range(self.num_augmentations):
                fname_aug = f"{img_name.replace('/', '_')}_aug{aug_idx}.pt"
                if fname_aug not in existing_files:
                    missing_augmented_count += 1
        
        print(f"Original features to extract: {missing_original_count}/{total_original}")
        print(f"Augmented features to extract: {missing_augmented_count}/{total_augmented}")
        
        if missing_original_count == 0 and missing_augmented_count == 0:
            print("All features already cached!")
            return
        
        print(f"Extracting intermediate features at {self.model.standard_size}x{self.model.standard_size}...")
        
        # Extract ORIGINAL features
        if missing_original_count > 0:
            pbar = tqdm(total=missing_original_count, desc="Extracting original features")
            processed_count = 0
            for img_name in unique_images:
                fname = f"{img_name.replace('/', '_')}.pt"
                if fname in existing_files:
                    continue
                
                self._extract_and_cache_single(img_name)
                pbar.update(1)
                
                processed_count += 1
                if processed_count % 100 == 0:
                    gc.collect()
            pbar.close()
        
        # Extract AUGMENTED features
        if missing_augmented_count > 0:
            pbar = tqdm(total=missing_augmented_count, desc="Extracting augmented features")
            processed_count = 0
            
            for img_name in unique_images:
                for aug_idx in range(self.num_augmentations):
                    fname_aug = f"{img_name.replace('/', '_')}_aug{aug_idx}.pt"
                    if fname_aug in existing_files:
                        continue
                        
                    self._extract_and_cache_single(img_name, aug_idx=aug_idx)
                    pbar.update(1)
                    
                    processed_count += 1
                    if processed_count % 100 == 0:
                        gc.collect()
            pbar.close()
        
        # Final cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        print(f"Cached features to {self.cache_dir}")
    
    def _extract_and_cache_single(self, img_name, aug_idx=None):
        """Extract and cache features for a single image.
        
        Args:
            img_name: Image name in format 'category/image_stem'
            aug_idx: Augmentation index for filename (None for original)
        """
        img_path = self.base_dataset.get_image_path(img_name)
        
        # Load image and get original size
        img = Image.open(img_path).convert('RGB')
        orig_w, orig_h = img.size
        
        # Preprocess at standard_size
        img_tensor = self.model.preprocess_image_pil(
            img, target_size=(self.model.standard_size, self.model.standard_size)
        )
        
        # Apply augmentation if specified
        if aug_idx is not None:
            img_tensor = self.augment_transform(img_tensor)
        
        # Extract INTERMEDIATE features (output of frozen blocks)
        with torch.no_grad():
            intermediate = self.model.extract_intermediate_features(img_tensor)
        
        # Prepare feature data
        feature_data = {
            'intermediate': intermediate.cpu(),
            'orig_size': (orig_w, orig_h),
        }
        
        # Build cache path
        if aug_idx is not None:
            cache_path = self.cache_dir / f"{img_name.replace('/', '_')}_aug{aug_idx}.pt"
        else:
            cache_path = self.cache_dir / f"{img_name.replace('/', '_')}.pt"
        
        torch.save(feature_data, cache_path)
        
        # Free memory
        del intermediate, img_tensor, img, feature_data
