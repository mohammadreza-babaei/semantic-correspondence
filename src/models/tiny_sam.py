import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import math
from .checkpoint_utils import load_weights, extract_state_dict, clean_state_dict_keys, get_merged_state_dict
from src.lora_utils import apply_lora

try:
    from tinysam import sam_model_registry
except ImportError:
    print("Error: 'tinysam' module not found.")
    sam_model_registry = None

# We can now stick to 512 because we will dynamically resize the model's embedding
STANDARD_SIZE = 512

class TinySAMAdapter:
    def __init__(
        self,
        model_name='vit_t',
        weights_path=None,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        num_unfrozen_blocks=2,
        dropout=0.0,
        unfreeze_neck=False,
        use_lora=False,
        lora_rank=8,
        lora_alpha=16,
    ):
        if sam_model_registry is None:
            raise ImportError("TinySAM not installed.")

        self.device = device
        self.standard_size = STANDARD_SIZE
        self.model_name = "tinysam_" + model_name
        self.patch_size = 16 
        
        print(f"Loading TinySAM {model_name} on {self.device}...")
        
        # 1. Load Model (It defaults to 1024x1024)

        if model_name == 'tiny_sam' or model_name == 'tinysam':
            print(f"Mapping model name '{model_name}' to registry key 'vit_t'")
            registry_key = 'vit_t'
        else:
            registry_key = model_name

        self.sam_model = sam_model_registry[registry_key](checkpoint=None)
        self.model = self.sam_model.image_encoder
        
        # 2. Load Weights
        if weights_path is not None and os.path.exists(weights_path):
            load_weights(self.sam_model, weights_path, map_location='cpu', strict=False, verbose=True)
        else:
            print(f"Warning: TinySAM weights '{weights_path}' not found. Using random init.")
            
        # --- CRITICAL FIX: RESIZE POSITIONAL ENCODINGS TO 512 ---
        # This prevents "AssertionError: input feature has wrong size"
        # without modifying the original source code.
        self._resize_pos_embed(self.standard_size)
        # --------------------------------------------------------
        
        self.sam_model.to(self.device)
        self.sam_model.eval()
        
        # 3. Analyze Backbone Structure
        self.stages = self.model.layers
        self.total_blocks = sum(len(stage.blocks) for stage in self.stages)
        
        self.num_unfrozen_blocks = num_unfrozen_blocks
        self.num_frozen_blocks = self.total_blocks - num_unfrozen_blocks
        
        # 4. Freeze/Unfreeze Logic
        for param in self.sam_model.parameters():
            param.requires_grad = False
            
        if use_lora:
            print(f"Applying LoRA to TinySAM (Rank={lora_rank})...")
            self.model = apply_lora(
                self.model, model_type='sam', rank=lora_rank, alpha=lora_alpha, 
                num_unfrozen_blocks=num_unfrozen_blocks
            )
        else:
            print(f"Standard Fine-tuning: Unfreezing last {num_unfrozen_blocks} blocks...")
            blocks_unfrozen_count = 0
            for stage in reversed(self.stages):
                for block in reversed(stage.blocks):
                    if blocks_unfrozen_count < num_unfrozen_blocks:
                        for param in block.parameters():
                            param.requires_grad = True
                        blocks_unfrozen_count += 1
                    else:
                        break
                if blocks_unfrozen_count >= num_unfrozen_blocks:
                    break
            
            if unfreeze_neck:
                print("Unfreezing neck layers...")
                for param in self.model.neck.parameters():
                    param.requires_grad = True

        trainable = sum(p.numel() for p in self.sam_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.sam_model.parameters())
        print(f"Trainable Params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        self.trainable_params = [p for p in self.sam_model.parameters() if p.requires_grad]

    def _resize_pos_embed(self, new_size):
        """
        Dynamically resizes the model's positional embeddings and internal attributes
        to support the new image resolution (e.g., 1024 -> 512).
        """
        print(f"Resizing TinySAM from {self.model.img_size} to {new_size}...")
        
        # 1. Update the main attribute
        old_size = self.model.img_size
        self.model.img_size = new_size
        
        # 2. Interpolate Absolute Positional Embeddings (if present)
        # TinyViT usually puts pos_embed at the start or in stages.
        if hasattr(self.model, 'pos_embed') and self.model.pos_embed is not None:
            pos_embed = self.model.pos_embed # (1, H*W, C) or (1, C, H, W)
            
            # TinyViT usually has shape (1, C, H, W) for pos_embed
            if pos_embed.dim() == 4:
                # Interpolate (1, C, H, W)
                new_pos_embed = F.interpolate(
                    pos_embed, 
                    size=(new_size // 4, new_size // 4), # TinyViT stem usually downsamples by 4
                    mode='bilinear', 
                    align_corners=False
                )
                self.model.pos_embed = nn.Parameter(new_pos_embed)
                print(f"  Interpolated pos_embed: {pos_embed.shape} -> {new_pos_embed.shape}")

        # 3. Recursively update any "input_resolution" or "window_size" in layers
        # TinyViT layers often store (H, W) tuple as input_resolution
        for stage_idx, stage in enumerate(self.model.layers):
            if hasattr(stage, 'input_resolution'):
                # Calculate new resolution for this stage
                # Stages usually downsample: 4x, 8x, 16x, 32x
                stride = 4 * (2**stage_idx) 
                new_res = (new_size // stride, new_size // stride)
                stage.input_resolution = new_res
                
                # Update blocks inside stage
                for block in stage.blocks:
                    if hasattr(block, 'input_resolution'):
                        block.input_resolution = new_res
                    # Some implementations use 'window_size' that must be compatible
                    if hasattr(block, 'window_size'):
                        # If window_size > new_res, we might need to clip it or it handles padding
                        pass 
        
        print("  TinySAM resize complete.")

    def preprocess_image_pil(self, img, target_size=None):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
            
        new_w = self.standard_size
        new_h = self.standard_size
        
        transform = T.Compose([
            T.Resize((new_h, new_w)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(img).unsqueeze(0).to(self.device)

    def extract_intermediate_features(self, image_tensor):
        """Runs the FROZEN part of the backbone."""
        with torch.no_grad():
            x = image_tensor
            if hasattr(self.model, 'patch_embed'):
                x = self.model.patch_embed(x)
            
            # 2. Add Pos Embed (Now correctly resized!)
            if hasattr(self.model, 'pos_embed') and self.model.pos_embed is not None:
                x = x + self.model.pos_embed

            # 3. Run Frozen Blocks
            blocks_processed = 0
            for stage in self.model.layers:
                stage_blocks = stage.blocks
                
                # Full Stage Frozen
                if blocks_processed + len(stage_blocks) <= self.num_frozen_blocks:
                    x = stage(x)
                    blocks_processed += len(stage_blocks)
                # Split Stage
                else:
                    if hasattr(stage, 'downsample') and stage.downsample is not None:
                        x = stage.downsample(x)
                    
                    for block in stage_blocks:
                        if blocks_processed < self.num_frozen_blocks:
                            x = block(x)
                            blocks_processed += 1
                        else:
                            break
                    break      
            return x

    def forward_unfrozen_blocks(self, intermediate_features):
        x = intermediate_features.to(self.device)
        
        if self.num_unfrozen_blocks == 0:
             x = x.permute(0, 3, 1, 2)
             x = self.model.neck(x)
             return x

        blocks_processed = self.num_frozen_blocks
        current_block_count = 0
        
        for stage in self.model.layers:
            stage_blocks = stage.blocks
            n_blocks = len(stage_blocks)
            
            if current_block_count + n_blocks <= self.num_frozen_blocks:
                current_block_count += n_blocks
                continue
            
            if current_block_count >= blocks_processed:
                 if hasattr(stage, 'downsample') and stage.downsample is not None:
                        x = stage.downsample(x)

            for i, block in enumerate(stage_blocks):
                global_idx = current_block_count + i
                if global_idx >= self.num_frozen_blocks:
                    x = block(x)
            
            current_block_count += n_blocks

        x = x.permute(0, 3, 1, 2) 
        x = self.model.neck(x)
        x = self.dropout(x)
        return x

    def get_model_state(self):
        return {
            "backbone_state_dict": get_merged_state_dict(self.model),
            "model_name": self.model_name
        }

    def load_model_state(self, checkpoint):
        state, format_info = extract_state_dict(checkpoint)
        print(f"Loading from {format_info}")
        state = clean_state_dict_keys(state)
        self.model.load_state_dict(state, strict=False)
        
        # Re-apply resize after loading state!
        self._resize_pos_embed(self.standard_size)
        print(f"Model state loaded for {self.model_name}")