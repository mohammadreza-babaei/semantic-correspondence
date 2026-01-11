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
        
        # Load Model via TIMM
        self.model = timm.create_model(
            model_name, 
            pretrained=(weights_path is None), 
            num_classes=0,       
            global_pool='',      
            features_only=True   
        ).to(self.device)
        
        # Load Custom Weights (if provided)
        if weights_path is not None and os.path.exists(weights_path):
            print(f"Loading custom weights from {weights_path}")
            checkpoint = torch.load(weights_path, map_location='cpu')
            if 'model' in checkpoint: checkpoint = checkpoint['model']
            if 'state_dict' in checkpoint: checkpoint = checkpoint['state_dict']
            
            self.model.load_state_dict(checkpoint, strict=False)

        self.model.eval()

        # Freeze / Unfreeze Logic
        for param in self.model.parameters():
            param.requires_grad = False
            
        print("Unfreezing model blocks...")
        
        # Collect all individual TinyVitBlock instances from all stages
        # TinyViT has stages_0, stages_1, stages_2, stages_3 as separate attributes
        all_blocks = []
        stage_idx = 0
        while hasattr(self.model, f'stages_{stage_idx}'):
            stage = getattr(self.model, f'stages_{stage_idx}')
            if hasattr(stage, 'blocks'):
                stage_blocks = list(stage.blocks)
                all_blocks.extend(stage_blocks)
                print(f"  Stage {stage_idx}: {len(stage_blocks)} blocks")
            stage_idx += 1
        
        total_blocks = len(all_blocks)
        
        if total_blocks > 0:
            print(f"Total transformer blocks: {total_blocks}")
            
            # Block-level unfreezing for consistency with DINOv2/DINOv3/SAM
            self.num_frozen_blocks = max(0, total_blocks - num_unfrozen_blocks)
            self.num_unfrozen_blocks = min(num_unfrozen_blocks, total_blocks)
            
            print(f"Unfreezing last {self.num_unfrozen_blocks} transformer blocks...")
            
            # Unfreeze last N blocks
            blocks_to_unfreeze = all_blocks[-self.num_unfrozen_blocks:]
            for block_idx, block in enumerate(blocks_to_unfreeze):
                global_idx = total_blocks - self.num_unfrozen_blocks + block_idx
                for param in block.parameters():
                    param.requires_grad = True
                print(f"  Unfrozen block {global_idx}")
        else:
            print("Warning: Could not find transformer blocks. Unfreezing last few parameters.")
            self.num_frozen_blocks = 0 
            self.num_unfrozen_blocks = num_unfrozen_blocks
            
            params = list(self.model.parameters())
            cutoff = int(len(params) * 0.2 * num_unfrozen_blocks) 
            for p in params[-cutoff:]:
                p.requires_grad = True

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
        if isinstance(features_list, (list, tuple)):
            last_feat = features_list[-1]
        else:
            last_feat = features_list
        
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
        """Load model state from checkpoint.
        
        Note: LoRA weights should already be merged at save time.
        """
        if "model_state_dict" in state_dict:
            self.model.load_state_dict(state_dict["model_state_dict"], strict=False)
        else:
            self.model.load_state_dict(state_dict, strict=False)
        print(f"Model state loaded for {self.model_name}")