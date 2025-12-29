import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import timm
import os

# TinyViT-21M-512 Native Resolution
STANDARD_SIZE = 512

class TinyViTAdapter:
    def __init__(
        self,
        model_name='tiny_vit_21m_512.dist_in22k_ft_in1k', 
        weights_path=None, 
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_unfrozen_blocks=2, 
    ):
        """
        Initializes the TinyViT adapter using timm.
        """
        self.device = device
        self.standard_size = STANDARD_SIZE
        self.model_name = "tiny_vit"
        
        # TinyViT usually uses 32 stride at the final stage
        self.patch_size = 32 
        
        print(f"Initializing {model_name} on {self.device}...")
        
        # 1. Load Model via TIMM
        self.model = timm.create_model(
            model_name, 
            pretrained=(weights_path is None), 
            num_classes=0,       
            global_pool='',      
            features_only=True   
        ).to(self.device)
        
        # 2. Load Custom Weights (if provided)
        if weights_path is not None and os.path.exists(weights_path):
            print(f"Loading custom weights from {weights_path}")
            checkpoint = torch.load(weights_path, map_location='cpu')
            if 'model' in checkpoint: checkpoint = checkpoint['model']
            if 'state_dict' in checkpoint: checkpoint = checkpoint['state_dict']
            
            self.model.load_state_dict(checkpoint, strict=False)

        self.model.eval()

        # 3. Freeze / Unfreeze Logic
        for param in self.model.parameters():
            param.requires_grad = False
            
        print("Unfreezing model stages...")
        
        # Locate stages
        if hasattr(self.model, 'stages'):
             stages = list(self.model.stages)
        elif hasattr(self.model, 'layers'):
             stages = list(self.model.layers)
        else:
            stages = [c for c in self.model.children() if isinstance(c, (nn.Sequential, nn.ModuleList))]

        if not stages:
            print("Warning: Could not automatically find 'stages'. Unfreezing last few named parameters instead.")
            # Fallback: Just define it as 0 to avoid Attribute Error in Trainer
            self.num_frozen_blocks = 0 
            
            params = list(self.model.parameters())
            cutoff = int(len(params) * 0.2 * num_unfrozen_blocks) 
            for p in params[-cutoff:]:
                p.requires_grad = True
        else:
            total_stages = len(stages)
            print(f"Found {total_stages} stages.")
            
            # --- FIX: DEFINE NUM_FROZEN_BLOCKS ---
            self.num_frozen_blocks = total_stages - num_unfrozen_blocks
            self.num_unfrozen_blocks = num_unfrozen_blocks
            # -------------------------------------
            
            stages_to_unfreeze = stages[-num_unfrozen_blocks:]
            for stage in stages_to_unfreeze:
                for param in stage.parameters():
                    param.requires_grad = True

        # Trainable stats
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable Params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        
        self.trainable_params = [p for p in self.model.parameters() if p.requires_grad and p.is_leaf]

    def preprocess_image(self, image_path, target_size=None):
        img = Image.open(image_path).convert('RGB')
        return self.preprocess_image_pil(img, target_size)

    def preprocess_image_pil(self, img, target_size=None):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
            
        t_h = self.standard_size
        t_w = self.standard_size
        
        transform = T.Compose([
            T.Resize((t_h, t_w)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(img).unsqueeze(0).to(self.device)

    def extract_intermediate_features(self, image_tensor):
        """Pass-Through: Returns the 512x512 image tensor."""
        return image_tensor

    def forward_unfrozen_blocks(self, intermediate_features):
        """Runs the full TinyViT backbone."""
        x = intermediate_features.to(self.device)
        
        features_list = self.model(x)
        last_feat = features_list[-1]
        
        target_dim = self.standard_size // 16 
        if last_feat.shape[-1] != target_dim:
            last_feat = F.interpolate(
                last_feat,
                size=(target_dim, target_dim),
                mode='bilinear',
                align_corners=False
            )
            
        return last_feat

    def get_model_state(self):
        return {
            "model_state_dict": self.model.state_dict(),
            "model_name": self.model_name
        }

    def load_model_state(self, state_dict):
        if "model_state_dict" in state_dict:
            self.model.load_state_dict(state_dict["model_state_dict"])
        else:
            self.model.load_state_dict(state_dict, strict=False)
        print(f"Model state loaded for {self.model_name}")