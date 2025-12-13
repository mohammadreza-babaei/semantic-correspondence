import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

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
        
        # Resize logic: If target_size is provided, use it. 
        # Otherwise, ensure dimensions are divisible by patch_size for ViT.
        w, h = img.size if target_size is None else target_size
        new_w = (w // self.patch_size) * self.patch_size
        new_h = (h // self.patch_size) * self.patch_size
        
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

# --- Usage Example ---
if __name__ == "__main__":
    # 1. Initialize
    extractor = DINOv2FeatureExtractor(model_name='dinov2_vits14')

    # 2. Load Dummy Image (Replace with path to SPair-71k image)
    # You can download a sample image or use a local path
    # img_tensor = extractor.preprocess_image("path/to/dog.jpg")
    
    # For testing, let's create a random tensor that mimics a preprocessed image
    dummy_input = torch.randn(1, 3, 840, 840).to(extractor.device)

    # 3. Extract Features
    features = extractor.extract_features(dummy_input)
    
    print(f"Output Feature Map Shape: {features.shape}") 
    # Expected: torch.Size([1, 384, 16, 16]) for ViT-S/14 and 224x224 input