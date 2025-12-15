import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from transformers import SamModel, SamConfig

class SAMFeatureExtractor:
    def __init__(self, model_name='facebook/sam-vit-base', device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initializes the SAM model using HuggingFace Transformers.
        
        Args:
            model_name (str): HF Model ID. Options:
                              'facebook/sam-vit-base' (Recommended for speed/memory)
                              'facebook/sam-vit-large'
                              'facebook/sam-vit-huge'
            device (str): Computation device.
        """
        self.device = device
        self.patch_size = 16 # SAM uses a patch size of 16 for its encoder
        
        print(f"Loading {model_name} on {self.device}...")
        
        # We load the full SAM model but will only use the vision_encoder part
        self.model = SamModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def preprocess_image(self, image_path, target_size=None):
        """
        Loads and preprocesses an image.
        SAM works best with 1024x1024, but can handle other sizes if divisible by 16.
        """
        img = Image.open(image_path).convert('RGB')
        
        # If no size provided, resize to 1024 (Standard SAM)
        if target_size is None:
            w, h = 1024, 1024
        else:
            w, h = target_size
            
        # Ensure divisible by patch_size (16)
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
        Extracts dense features from the image using SAM's Vision Encoder.
        
        Returns:
            torch.Tensor: Feature map of shape (1, 256, H_feat, W_feat)
        """
        with torch.no_grad():
            # SAM's get_image_embeddings does the forward pass through vision_encoder
            # and the neck, returning the final 256-dim embedding.
            # Output shape is typically (B, 256, 64, 64) for 1024x1024 input.
            features = self.model.get_image_embeddings(pixel_values=image_tensor)
            
            # L2 Normalize features (critical for Cosine Similarity)
            features = torch.nn.functional.normalize(features, dim=1)
            
            return features

    def map_keypoints(self, keypoints, inverse=False):
        """
        Maps keypoints between image pixel coordinates and feature map coordinates.
        SAM's scale factor is typically 16 (1024 pixels -> 64 features).
        """
        if not isinstance(keypoints, torch.Tensor):
            keypoints = torch.tensor(keypoints, device=self.device)
            
        if inverse:
            # Feature Grid -> Image Pixels
            return keypoints * self.patch_size + (self.patch_size / 2)
        else:
            # Image Pixels -> Feature Grid
            return keypoints / self.patch_size

# --- Usage Example ---
if __name__ == "__main__":
    # 1. Initialize
    # Ensure you have `pip install transformers`
    extractor = SAMFeatureExtractor(model_name='facebook/sam-vit-base')

    # 2. Dummy Input (Standard SAM size 1024)
    dummy_input = torch.randn(1, 3, 1024, 1024).to(extractor.device)

    # 3. Extract
    features = extractor.extract_features(dummy_input)
    
    print(f"Output Feature Map Shape: {features.shape}")
    # Expected: (1, 256, 64, 64)