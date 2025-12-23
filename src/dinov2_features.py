import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os


# Standard image size for all feature extraction (divisible by patch_size=14)
STANDARD_SIZE = 518 # Multiple of 14 (14*37=518), used in the DINOv2 paper

class DINOv2FeatureExtractor:
    def __init__(self, model_name='dinov2_vits14', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initializes the DINOv2 model.
        
        Args:
            model_name (str): The DINOv2 model to load. Options: 
                              'dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14'
            device (str): Computation device ('cuda' or 'cpu').
        """
        self.device = device
        self.patch_size = 14 # DINOv2 usually uses patch size 14
        
        print(f"Loading {model_name} on {self.device}...")
        self.model = torch.hub.load('facebookresearch/dinov2', model_name).to(self.device)
        self.model.eval() # Set to evaluation mode (frozen features)

    def preprocess_image(self, image_path, target_size=None):
        """
        Loads and preprocesses an image. Ensures dimensions are multiples of patch_size.
        """
        img = Image.open(image_path).convert('RGB')
        return self.preprocess_image_pil(img, target_size)
    
    def preprocess_image_pil(self, img, target_size=None):
        """
        Preprocesses a PIL image. Ensures dimensions are multiples of patch_size.
        
        Args:
            img: PIL Image
            target_size: Optional tuple (width, height) for target size
            
        Returns:
            torch.Tensor: Preprocessed image tensor (1, C, H, W)
        """
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
         
        #ensure dimensions are divisible by patch_size.
        w, h = img.size if target_size is None else target_size
        new_w = (w // self.patch_size) * self.patch_size
        new_h = (h // self.patch_size) * self.patch_size
        
        resize_transform = T.Compose([
            T.Resize((new_h, new_w)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        return resize_transform(img).unsqueeze(0).to(self.device)
    
    def get_embed_dim(self):
        """Get the embedding dimension of the model."""
        return self.model.embed_dim

    def extract_features(self, image_tensor):
        """
        Extracts dense features from the image
        
        Returns:
            torch.Tensor: Feature map of shape (1, C, H_patch, W_patch)
                          where H_patch = H_img // 14, W_patch = W_img // 14
        """
        with torch.no_grad():
            # DINOv2 forward_features returns a dict. 
            # 'x_norm_patchtokens' contains the patch features after the last block & norm.
            features_dict = self.model.forward_features(image_tensor)
            patch_tokens = features_dict['x_norm_patchtokens'] # Shape: (B, N_patches, C)
            
            # Reshape tokens back to spatial grid
            B, N, C = patch_tokens.shape
            H_img, W_img = image_tensor.shape[2], image_tensor.shape[3]
            H_patch = H_img // self.patch_size
            W_patch = W_img // self.patch_size
            
            # Sanity check: ensure patch count matches dimensions
            assert N == H_patch * W_patch, "Patch count mismatch!"
            
            # Reshape to (B, C, H, W) for easier cosine similarity calculation later
            feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H_patch, W_patch)
            
            # L2 Normalize features (critical for Cosine Similarity) 
            feature_map = torch.nn.functional.normalize(feature_map, dim=1)
            
            return feature_map

    def map_keypoints(self, keypoints, inverse=False):
        """
        Maps keypoints between image pixel coordinates and feature map coordinates.
        
        Args:
            keypoints (torch.Tensor or numpy.ndarray): Shape (N, 2) containing (x, y) coordinates.
            inverse (bool): If False (default), maps Image Pixels -> Feature Grid (divides by patch_size).
                            If True, maps Feature Grid -> Image Pixels (multiplies by patch_size).
        
        Returns:
            torch.Tensor: Mapped keypoints.
        """
        if not isinstance(keypoints, torch.Tensor):
            keypoints = torch.tensor(keypoints, device=self.device)
            
        if inverse:
            # Feature Grid -> Image Pixels
            # We map the center of the patch back to the pixel space
            return keypoints * self.patch_size + (self.patch_size / 2)
        else:
            # Image Pixels -> Feature Grid
            return keypoints / self.patch_size

    def fine_tune(self, checkpoint_path='checkpoints/dinov2_features/dinov2_vits14_features.pt'):
        """
        Loads pretrained features from a checkpoint file.
        
        Args:
            checkpoint_path (str): Path to the checkpoint file containing pretrained features.
        
        Returns:
            dict: Dictionary containing the loaded features with image identifiers as keys.
        """
        print(f"Loading features from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if isinstance(checkpoint, dict):
            print(f"Loaded {len(checkpoint) - 1 if 'device' in checkpoint else len(checkpoint)} feature sets from checkpoint.")
            # Remove 'device' key if present since it's metadata
            if 'device' in checkpoint:
                del checkpoint['device']
            return checkpoint
        else:
            raise ValueError("Checkpoint format not recognized. Expected a dictionary.")



class DINOv2FineTuner:
    """Fine-tuning trainer for DINOv2-based semantic correspondence by unfreezing last layers.
    
    Architecture for gradient flow:
    - Pre-extracts INTERMEDIATE features from frozen blocks (cached to disk)
    - During training, only the unfrozen blocks + norm are computed with gradients
    - All images are resized to STANDARD_SIZE (518x518) for consistent feature maps
    
    This allows efficient training while maintaining proper gradient flow.
    """
    
    def __init__(
        self,
        model_name='dinov2_vits14',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_unfrozen_blocks=2,
        learning_rate=1e-5,
        feature_extractor=None,
    ):
        self.standard_size = STANDARD_SIZE
        """
        Initialize the fine-tuner.
        
        Args:
            model_name: DINOv2 model variant to use
            device: Device for computation
            num_unfrozen_blocks: Number of transformer blocks to unfreeze from the end
            learning_rate: Learning rate for training
            feature_extractor: Optional pre-initialized DINOv2FeatureExtractor. If None,
                               a new one will be created.
        """
        self.device = device
        self.model_name = model_name
        
        # Use provided feature extractor or create a new one
        if feature_extractor is not None:
            self.feature_extractor = feature_extractor
            self.backbone = feature_extractor.model
            print(f"Using provided DINOv2FeatureExtractor")
        else:
            self.feature_extractor = DINOv2FeatureExtractor(model_name=model_name, device=device)
            self.backbone = self.feature_extractor.model
        
        self.patch_size = self.feature_extractor.patch_size
        self.embed_dim = self.feature_extractor.get_embed_dim()
        
        # Store number of unfrozen blocks for feature caching logic
        self.num_unfrozen_blocks = num_unfrozen_blocks
        self.num_frozen_blocks = len(self.backbone.blocks) - num_unfrozen_blocks
        
        # Freeze all parameters first
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Unfreeze the last N transformer blocks
        num_blocks = len(self.backbone.blocks)
        print(f"Total transformer blocks: {num_blocks}")
        print(f"Frozen blocks: {self.num_frozen_blocks}, Unfrozen blocks: {num_unfrozen_blocks}")
        
        blocks_to_unfreeze = list(self.backbone.blocks)[-num_unfrozen_blocks:]
        for block in blocks_to_unfreeze:
            for param in block.parameters():
                param.requires_grad = True
        
        # Also unfreeze the final norm layer
        if hasattr(self.backbone, 'norm'):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True
            print("Unfreezing final norm layer...")
        
        # Count trainable parameters
        trainable_params = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.backbone.parameters())
        print(f"Trainable parameters: {trainable_params:,} / {total_params:,} ({100*trainable_params/total_params:.2f}%)")
        
        
        # Filter to ensure we only get leaf tensors that require grad
        self.trainable_params = [p for p in self.backbone.parameters() if p.requires_grad and p.is_leaf]
        
        # Pre-extracted INTERMEDIATE features cache (output of frozen blocks)
        # These are the inputs to the unfrozen blocks
        self.features_cache = {}
    
    def extract_intermediate_features(self, image_tensor):
        """
        Extract INTERMEDIATE features - output of frozen blocks, before unfrozen blocks.
        
        Args:
            image_tensor: Preprocessed image tensor (B, C, H, W)
            
        Returns:
            torch.Tensor: Intermediate token features (B, N, C) - input to unfrozen blocks
        """
        with torch.no_grad():
            # Run patch embedding
            x = self.backbone.patch_embed(image_tensor)
            
            # Add CLS token
            cls_tokens = self.backbone.cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)
            
            # Add position embeddings
            x = x + self.backbone.interpolate_pos_encoding(x, image_tensor.shape[2], image_tensor.shape[3])
            
            # Run through FROZEN blocks only
            for i, block in enumerate(self.backbone.blocks):
                if i >= self.num_frozen_blocks:
                    break
                x = block(x)
            
            return x
    
    def forward_unfrozen_blocks(self, intermediate_features):
        """
        Run intermediate features through unfrozen blocks + norm.
        
        This method DOES track gradients for training.
        
        Args:
            intermediate_features: Output from frozen blocks (B, N, C)
            
        Returns:
            torch.Tensor: Feature map (B, C, H, W) - L2 normalized spatial features
        """
        x = intermediate_features
        
        # Run through UNFROZEN blocks
        for block in list(self.backbone.blocks)[-self.num_unfrozen_blocks:]:
            x = block(x)
        
        # Apply final norm
        x = self.backbone.norm(x)
        
        # Remove CLS token to get patch tokens only
        patch_tokens = x[:, 1:]  # (B, N, C)
        
        # Reshape to spatial grid
        B, N, C = patch_tokens.shape
        H = W = int(N ** 0.5)  # For 518x518 input with patch_size=14: 60x60
        assert H * W == N, f"Patch count {N} is not a perfect square"
        
        feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)
        
        # L2 normalize features
        feature_map = F.normalize(feature_map, dim=1)
        
        return feature_map
    
    def extract_all_features(self, dataset, show_progress=True):
        """
        Pre-extract INTERMEDIATE features for all images at STANDARD_SIZE (518x518).
        
        Intermediate features are the output of frozen blocks (before unfrozen blocks).
        During training, only unfrozen blocks are computed with gradients.
        
        Args:
            dataset: SPair71kPairs dataset (any split, will extract from all JPEGImages)
            show_progress: Whether to show progress bar
            
        Returns:
            dict: Dictionary mapping image names to intermediate feature info
        """
        from tqdm import tqdm
        
        # Create cache directory and file path
        # Include num_frozen_blocks in filename since intermediate features depend on it
        cache_dir = Path('checkpoints') / 'feature_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = cache_dir / f"{self.model_name}_intermediate_{self.num_frozen_blocks}frozen.pt"
        
        # Try to load all features from disk cache first
        if cache_file.exists():
            try:
                print(f"Loading intermediate features from {cache_file}...")
                self.features_cache = torch.load(cache_file, map_location='cpu')
                print(f"Loaded {len(self.features_cache)} cached intermediate features from disk")
                print(f"Features stored in RAM (CPU memory)")
                return self.features_cache
            except Exception as e:
                print(f"Warning: Failed to load cache file: {e}")
                print("Will re-extract features...")
                self.features_cache = {}
        
        # Collect ALL unique images from the JPEGImages directory (all splits)
        print("Scanning all images in JPEGImages directory...")
        jpeg_dir = dataset.root / 'JPEGImages'
        unique_images = set()
        
        # Walk through all category subdirectories
        for category_dir in jpeg_dir.iterdir():
            if category_dir.is_dir():
                category = category_dir.name
                for img_file in category_dir.glob('*.jpg'):
                    img_name = f"{category}/{img_file.stem}"
                    unique_images.add(img_name)
        
        print(f"Found {len(unique_images)} unique images across all splits")
        print(f"Extracting intermediate features at {STANDARD_SIZE}x{STANDARD_SIZE}...")
        
        # Extract features for each unique image
        iterator = tqdm(unique_images, desc="Extracting intermediate features") if show_progress else unique_images
        
        for img_name in iterator:
            if img_name in self.features_cache:
                continue
            
            # Get image path
            img_path = dataset.root / 'JPEGImages' / f'{img_name}.jpg'
            
            # Load image and get original size
            img = Image.open(img_path).convert('RGB')
            orig_w, orig_h = img.size
            
            # Preprocess at STANDARD_SIZE (518x518)
            img_tensor = self.feature_extractor.preprocess_image_pil(img, target_size=(STANDARD_SIZE, STANDARD_SIZE))
            
            # Extract INTERMEDIATE features (output of frozen blocks)
            intermediate = self.extract_intermediate_features(img_tensor)
            
            # Store intermediate features in CPU memory (RAM) to save VRAM
            self.features_cache[img_name] = {
                'intermediate': intermediate.cpu(),  # Move to CPU: (1, N+1, C) including CLS token
                'orig_size': (orig_w, orig_h),
            }
        
        # Save all features to disk in a single file
        try:
            print(f"Saving {len(self.features_cache)} intermediate features to {cache_file}...")
            torch.save(self.features_cache, cache_file)
            print("Features saved successfully")
        except Exception as e:
            print(f"Warning: Failed to save cache file: {e}")
        
        print(f"Cached intermediate features for {len(self.features_cache)} images")
        return self.features_cache
    
    def _collate_fn(self, batch):
        """Custom collate function for SPair dataset."""
        # For batch_size=1, just return the single item
        if len(batch) == 1:
            return batch[0]
        # For larger batches, keep as list
        return batch
    
    def save_checkpoint(self, path):
        """Save model checkpoint."""
        checkpoint = {
            'backbone_state_dict': self.backbone.state_dict(),
            'embed_dim': self.embed_dim,
            'patch_size': self.patch_size
        }
        torch.save(checkpoint, path)
        print(f"Checkpoint saved: {path}")
    
    def load_checkpoint(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        print(f"Checkpoint loaded: {path}")