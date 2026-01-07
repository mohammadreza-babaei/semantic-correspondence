import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
from .checkpoint_utils import load_weights, extract_state_dict, clean_state_dict_keys, get_merged_state_dict
from src.lora_utils import apply_lora

# Try to import TinySAM. If not installed, raise clear error.
try:
    from tinysam import sam_model_registry
except ImportError:
    print("Error: 'tinysam' module not found.")
    print("Please clone the repo: https://github.com/xinghaochen/TinySAM")
    print("And ensure the 'tinysam' folder is in your project root.")
    sam_model_registry = None

# TinySAM native resolution is 1024, but for correspondence 512 is often sufficient/faster.
# Matches your other adapters.
STANDARD_SIZE = 1024

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
        resolution=None
    ):
        """
        Initializes the TinySAM model (XinghaoChen implementation).
        
        Args:
            model_name (str): Usually 'vit_t' for TinySAM.
            weights_path (str): Path to the TinySAM checkpoint (.pth).
            device (str): 'cuda' or 'cpu'.
            num_unfrozen_blocks (int): Number of blocks to unfreeze from the end.
                                       TinyViT is hierarchical (4 stages). This unfreezes
                                       the last N blocks of the LAST stage.
        """
        if sam_model_registry is None:
            raise ImportError("TinySAM not installed.")

        self.device = device
        self.standard_size = resolution if resolution is not None else STANDARD_SIZE
        self.model_name = "tinysam_" + model_name
        self.patch_size = 16 # TinyViT generally acts like patch 16 at output
        
        print(f"Loading TinySAM {model_name} on {self.device}...")
        
        # 1. Load Model
        # We pass checkpoint=None to avoid loading inside the registry (we handle it below)
        if resolution is not None:
             self.sam_model = sam_model_registry[model_name](checkpoint=None, image_size=resolution)
        else:
             self.sam_model = sam_model_registry[model_name](checkpoint=None)
        self.model = self.sam_model.image_encoder
        
        # 2. Load Weights Manually (Robust Loading)
        if weights_path is not None:
            if os.path.exists(weights_path):
                load_weights(
                    self.sam_model, # Load into full SAM model
                    weights_path, 
                    map_location=self.device, 
                    strict=False,
                    verbose=True
                )
            else:
                print(f"Warning: TinySAM weights '{weights_path}' not found. Using random init.")
        
        self.sam_model.to(self.device)
        self.sam_model.eval()
        
        # 3. Analyze Backbone Structure (TinyViT Hierarchy)
        # TinyViT has 'layers' (stages), each having 'blocks'.
        # We need to flatten this conceptually to decide what to freeze.
        self.stages = self.model.layers
        self.total_blocks = sum(len(stage.blocks) for stage in self.stages)
        
        self.num_unfrozen_blocks = num_unfrozen_blocks
        self.num_frozen_blocks = self.total_blocks - num_unfrozen_blocks
        
        # 4. Freeze/Unfreeze Logic
        # Freeze everything first
        for param in self.sam_model.parameters():
            param.requires_grad = False
            
        if use_lora:
            print(f"Applying LoRA to TinySAM (Rank={lora_rank})...")
            # Apply LoRA to the image encoder
            self.model = apply_lora(
                self.model, 
                model_type='sam', # TinySAM structure mimics SAM
                rank=lora_rank, 
                alpha=lora_alpha, 
                num_unfrozen_blocks=num_unfrozen_blocks
            )
        else:
            print(f"Standard Fine-tuning: Unfreezing last {num_unfrozen_blocks} blocks...")
            
            # Iterate backwards through stages and blocks to unfreeze
            blocks_unfrozen_count = 0
            
            # Reverse stages
            for stage in reversed(self.stages):
                # Reverse blocks in stage
                for block in reversed(stage.blocks):
                    if blocks_unfrozen_count < num_unfrozen_blocks:
                        for param in block.parameters():
                            param.requires_grad = True
                        blocks_unfrozen_count += 1
                    else:
                        break # Done unfreezing
                if blocks_unfrozen_count >= num_unfrozen_blocks:
                    break
            
            # Unfreeze neck if requested
            if unfreeze_neck:
                print("Unfreezing neck layers...")
                for param in self.model.neck.parameters():
                    param.requires_grad = True

        # Stats
        trainable = sum(p.numel() for p in self.sam_model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.sam_model.parameters())
        print(f"Trainable Params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
        # Dropout
        self.dropout = nn.Dropout2d(p=dropout) if dropout > 0 else nn.Identity()

        self.trainable_params = [p for p in self.sam_model.parameters() if p.requires_grad]

    def preprocess_image(self, image_path, target_size=None):
        img = Image.open(image_path).convert('RGB')
        return self.preprocess_image_pil(img, target_size)

    def preprocess_image_pil(self, img, target_size=None):
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img)
            
        w, h = img.size if target_size is None else target_size
        
        # Resize to standard size (TinySAM expects square usually, or divisible by 16)
        # We enforce square for consistency with cache
        new_w = self.standard_size
        new_h = self.standard_size
        
        transform = T.Compose([
            T.Resize((new_h, new_w)),
            T.ToTensor(),
            # TinySAM uses standard ImageNet normalization
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        return transform(img).unsqueeze(0).to(self.device)

    def extract_intermediate_features(self, image_tensor):
        """
        Runs the FROZEN part of the backbone.
        Handles the hierarchical stages of TinyViT.
        """
        with torch.no_grad():
            x = image_tensor
            
            # 1. Patch Embedding (Stem) - usually in self.model.patch_embed
            # TinySAM implementation might vary, but standard is `patch_embed`
            if hasattr(self.model, 'patch_embed'):
                x = self.model.patch_embed(x)
            
            # 2. Absolute Positional Embedding (if present at start)
            if hasattr(self.model, 'pos_embed') and self.model.pos_embed is not None:
                x = x + self.model.pos_embed

            # 3. Run Frozen Blocks across Stages
            # We need to keep track of how many blocks we've run to stop at the cut-off
            blocks_processed = 0
            
            for stage in self.model.layers:
                # A stage contains a list of blocks
                stage_blocks = stage.blocks
                
                # Check if we should run this whole stage (all frozen)
                if blocks_processed + len(stage_blocks) <= self.num_frozen_blocks:
                    # Run full stage
                    x = stage(x)
                    blocks_processed += len(stage_blocks)
                else:
                    # We are in the split stage. Run only the frozen blocks within this stage.
                    
                    # Run remaining frozen blocks individually
                    for block in stage_blocks:
                        if blocks_processed < self.num_frozen_blocks:
                            x = block(x)
                            blocks_processed += 1
                        else:
                            break # Stop exactly at the cut-off
                    
                    # We stop here. 'x' is now the input to the first Unfrozen block.
                    # We DO NOT run downsample here, because in TinyViT downsample is at the END.
                    # It will be run in forward_unfrozen_blocks after the remaining blocks.
                    break
                    
            return x

    def forward_unfrozen_blocks(self, intermediate_features):
        """
        Runs the UNFROZEN part of the backbone + Neck.
        """
        x = intermediate_features.to(self.device)
        
        # If num_unfrozen_blocks is 0, just run neck
        if self.num_unfrozen_blocks == 0:
             x = x.permute(0, 3, 1, 2) # (B, H, W, C) -> (B, C, H, W)
             x = self.model.neck(x)
             return x

        blocks_processed = self.num_frozen_blocks
        
        # Resume iteration from where we left off
        # We need to find the stage where we stopped
        current_block_count = 0
        
        for stage in self.model.layers:
            stage_blocks = stage.blocks
            n_blocks = len(stage_blocks)
            
            # If we fully processed this stage in extract(), skip it
            if current_block_count + n_blocks <= self.num_frozen_blocks:
                current_block_count += n_blocks
                continue
            
            # We are in a stage that has unfrozen blocks
            
            # Run Unfrozen Blocks
            for i, block in enumerate(stage_blocks):
                global_idx = current_block_count + i
                if global_idx >= self.num_frozen_blocks:
                    x = block(x)
                    
            # Run Downsample at the end of the stage
            if hasattr(stage, 'downsample') and stage.downsample is not None:
                x = stage.downsample(x)
            
            current_block_count += n_blocks

        # Final processing
        # TinyViT outputs (B, L, C) at the end
        if x.dim() == 3:
            B, L, C = x.shape
            H = int(L**0.5)
            W = H
            x = x.view(B, H, W, C)
        
        # Now x is (B, H, W, C)
        # Neck expects (B, C, H, W)
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
        print(f"Model state loaded for {self.model_name}")