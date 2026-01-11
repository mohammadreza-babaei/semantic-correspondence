from src.models.dinov2 import DINOv2Adapter
from src.models.dinov3 import DINOv3Adapter
from src.models.sam import SAMAdapter
from src.models.tiny_vit import TinyViTAdapter
from src.models.tiny_sam import TinySAMAdapter
from src.models.mixed import MixedModelAdapter

def create_model_adapter(
    model_name,
    device,
    weights_path=None,
    num_unfrozen_blocks=2,
    dropout=0.0,
    use_lora=False,
    lora_rank=8,
    lora_alpha=16,
    unfreeze_neck=False,
    resolution=None,
    mix_model_name=None,
    mix_weights_path=None,
    mix_weights=None
):
    """
    Factory function to create model adapters for training or evaluation.
    """
    
    # Handle Mixed Model 
    if mix_model_name:
        print(f"Initializing primary model: {model_name}")
        adapter1 = create_model_adapter(
            model_name=model_name,
            device=device,
            weights_path=weights_path,
            num_unfrozen_blocks=num_unfrozen_blocks,
            dropout=dropout,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            unfreeze_neck=unfreeze_neck,
            resolution=resolution
        )
        
        print(f"Initializing secondary model for mixing: {mix_model_name}")
        adapter2 = create_model_adapter(
            model_name=mix_model_name,
            device=device,
            weights_path=mix_weights_path,
            num_unfrozen_blocks=num_unfrozen_blocks,
        )
        
        mixing_weights_list = None
        if mix_weights:
            try:
                mixing_weights_list = [float(w) for w in mix_weights.split(',')]
                if len(mixing_weights_list) != 2:
                    raise ValueError("mix-weights must have exactly two values")
                print(f"Using mixing weights: {mixing_weights_list}")
            except ValueError as e:
                print(f"Error parsing mix-weights: {e}")
                raise

        print(f"Creating MixedModelAdapter...")
        return MixedModelAdapter(adapter1, adapter2, mix_weights=mixing_weights_list)

    # Standard Models
    if "dinov2" in model_name:
        return DINOv2Adapter(
            model_name=model_name,
            device=device,
            num_unfrozen_blocks=num_unfrozen_blocks,
            weights_path=weights_path,
            dropout=dropout,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha
        )
        
    elif "dinov3" in model_name:
        return DINOv3Adapter(
            model_name=model_name,
            device=device,
            num_unfrozen_blocks=num_unfrozen_blocks,
            weights_path=weights_path,
            dropout=dropout,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha
        )
        
    elif "sam" in model_name and "tinysam" not in model_name:
        # e.g. sam_vit_b -> vit_b
        return SAMAdapter(
            model_name=model_name.replace("sam_", ""),
            device=device,
            num_unfrozen_blocks=num_unfrozen_blocks,
            weights_path=weights_path,
            unfreeze_neck=unfreeze_neck,
            dropout=dropout,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha
        )
        
    elif "tiny_vit" in model_name:
        return TinyViTAdapter(
            model_name='tiny_vit_21m_512.dist_in22k_ft_in1k', 
            weights_path=weights_path,
            device=device,
            num_unfrozen_blocks=num_unfrozen_blocks
        )
        
    elif "tinysam" in model_name:
        # usually tinysam_vit_t or just tinysam
        return TinySAMAdapter(
            model_name='vit_t',
            weights_path=weights_path,
            device=device,
            num_unfrozen_blocks=num_unfrozen_blocks,
            dropout=dropout,
            unfreeze_neck=unfreeze_neck,
            use_lora=use_lora,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            resolution=resolution
        )
        
    else:
        # Fallback or error
        if "dino" in model_name:
             return DINOv2Adapter(
                model_name=model_name,
                device=device,
                num_unfrozen_blocks=num_unfrozen_blocks,
                weights_path=weights_path,
                dropout=dropout,
                use_lora=use_lora,
                lora_rank=lora_rank,
                lora_alpha=lora_alpha
            )
        raise ValueError(f"Unknown model: {model_name}")