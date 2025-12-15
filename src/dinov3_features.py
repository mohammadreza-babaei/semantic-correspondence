import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

class DINOv3FeatureExtractor:
    def __init__(self, model_name='dinov3_vits16', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initializes the DINOv3 model.
        
        Args:
            model_name (str): The DINOv3 model to load. 
                              Common options: 'dinov3_vits16', 'dinov3_vitb16', 'dinov3_vitl16'
            device (str): Computation device ('cuda' or 'cpu').
        """
        self.device = device
        self.patch_size = 16
            
        print(f"Loading {model_name} on {self.device} with patch_size={self.patch_size}...")
        
        # Load from PyTorch Hub (assuming facebookresearch/dinov3 repo structure)
        # Note: If the official Hub repo is slightly different, this string might need adjustment 
        # based on the specific release tag (e.g. 'facebookresearch/dinov3:main')
        try:
            self.model = torch.hub.load('facebookresearch/dinov3', model_name).to(self.device)
        except Exception as e:
            print(f"Standard hub load failed ({e}). Trying with trust_repo=True...")
            self.model = torch.hub.load('facebookresearch/dinov3', model_name, trust_repo=True).to(self.device)
            
        self.model.eval() # Set to evaluation mode (frozen features)

    def preprocess_image(self, image_path, target_size=None):
        """
        Loads and preprocesses an image. Ensures dimensions are multiples of patch_size.
        """
        img = Image.open(image_path).convert('RGB')
        
        # Resize logic: If target_size is provided, use it. 
        # Otherwise, ensure dimensions are divisible by patch_size.
        w, h = img.size if target_size is None else target_size
        new_w = (w // self.patch_size) * self.patch_size
        new_h = (h // self.patch_size) * self.patch_size
        
        # DINOv3 uses standard ImageNet normalization
        resize_transform = T.Compose([
            T.Resize((new_h, new_w)),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])
        
        return resize_transform(img).unsqueeze(0).to(self.device)

    def extract_features(self, image_tensor):
        """
        Extracts dense features from the image.
        
        Returns:
            torch.Tensor: Feature map of shape (1, C, H_patch, W_patch)
                          where H_patch = H_img // patch_size, W_patch = W_img // patch_size
        """
        with torch.no_grad():
            # Forward pass to get intermediate features
            # DINOv3 API is generally consistent with v2, returning a dict
            features_dict = self.model.forward_features(image_tensor)
            
            # 'x_norm_patchtokens' is the standard key for normalized patch features
            patch_tokens = features_dict['x_norm_patchtokens'] # Shape: (B, N_patches, C)
            
            # Reshape tokens back to spatial grid
            B, N, C = patch_tokens.shape
            H_img, W_img = image_tensor.shape[2], image_tensor.shape[3]
            H_patch = H_img // self.patch_size
            W_patch = W_img // self.patch_size
            
            # Sanity check: ensure patch count matches dimensions
            if N != H_patch * W_patch:
                raise ValueError(f"Patch count mismatch! Expected {H_patch*W_patch}, got {N}. "
                                 f"Check input image size ({H_img}x{W_img}) and patch_size ({self.patch_size}).")
            
            # Reshape to (B, C, H, W)
            feature_map = patch_tokens.permute(0, 2, 1).reshape(B, C, H_patch, W_patch)
            
            # L2 Normalize features
            feature_map = torch.nn.functional.normalize(feature_map, dim=1)
            
            return feature_map

    def map_keypoints(self, keypoints, inverse=False):
        """
        Maps keypoints between image pixel coordinates and feature map coordinates.
        
        Args:
            keypoints (torch.Tensor or numpy.ndarray): Shape (N, 2) containing (x, y) coordinates.
            inverse (bool): If False (default), maps Image Pixels -> Feature Grid.
                            If True, maps Feature Grid -> Image Pixels.
        
        Returns:
            torch.Tensor: Mapped keypoints.
        """
        if not isinstance(keypoints, torch.Tensor):
            keypoints = torch.tensor(keypoints, device=self.device)
            
        if inverse:
            # Feature Grid -> Image Pixels
            # Map center of patch back to pixel space
            return keypoints * self.patch_size + (self.patch_size / 2)
        else:
            # Image Pixels -> Feature Grid
            return keypoints / self.patch_size

# --- Usage Example ---
if __name__ == "__main__":
    # 1. Initialize
    # Note: DINOv3 usually defaults to patch size 16 (e.g. dinov3_vits16)
    extractor = DINOv3FeatureExtractor(model_name='dinov3_vits16')

    # 2. Create Dummy Input (Standard DINOv3 size often divisible by 16)
    dummy_input = torch.randn(1, 3, 848, 848).to(extractor.device) # 848 is divisible by 16

    # 3. Extract Features
    features = extractor.extract_features(dummy_input)
    
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output Feature Map Shape: {features.shape}")
    # Expected H, W = 848/16 = 53