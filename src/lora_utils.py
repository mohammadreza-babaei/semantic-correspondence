
import torch
import peft
from peft import LoraConfig, get_peft_model

def apply_lora(model, model_type, rank=8, alpha=16, num_unfrozen_blocks=2):
    """
    Applies LoRA to the specific 'unfrozen' blocks of the model.

    Args:
        model (nn.Module): The backbone or encoder model to wrap.
        model_type (str): 'dinov3', 'dinov2', or 'sam'.
        rank (int): LoRA rank dimension.
        alpha (int): LoRA scaling factor.
        num_unfrozen_blocks (int): Number of blocks from the END to apply LoRA to.

    Returns:
        peft.PeftModel: The wrapped model with LoRA adapters.
    """
    
    # Identify the total number of blocks based on inspection
    # All supported models (DINOv2, DINOv3, SAM) use a list called 'blocks' 
    # within the main container or directly.
    # We expect 'model' to be the Vision Transformer part which has 'blocks'.
    
    if not hasattr(model, 'blocks'):
        raise ValueError(f"Model {type(model)} does not have 'blocks' attribute. Cannot target blocks for LoRA.")

    total_blocks = len(model.blocks)
    start_block = total_blocks - num_unfrozen_blocks
    
    if start_block < 0:
        start_block = 0
        
    print(f"LoRA Configuration:")
    print(f"  - Rank: {rank}")
    print(f"  - Alpha: {alpha}")
    print(f"  - Targets: Blocks {start_block} to {total_blocks-1} (inclusive)")
    
    # Define target modules based on model type
    target_suffixes = []
    
    if 'dinov2' in model_type.lower():
        target_suffixes = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
        
    elif 'dinov3' in model_type.lower():
        target_suffixes = ["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"]
        
    elif 'sam' in model_type.lower():
        target_suffixes = ["attn.qkv", "attn.proj", "mlp.lin1", "mlp.lin2", "neck.0", "neck.1", "neck.2", "neck.3"]
        
    else:
        raise ValueError(f"Unknown model_type for LoRA: {model_type}")

    # Build the full target module list (exact names)
    target_modules = []
    for i in range(start_block, total_blocks):
        for suffix in target_suffixes:
            target_modules.append(f"blocks.{i}.{suffix}")
    
    modules_to_save = []
    if hasattr(model, 'norm'):
        modules_to_save.append("norm")
    
    print(f"  - Target Suffixes: {target_suffixes}")
    print(f"  - Modules to Save: {modules_to_save}")
    
    config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        modules_to_save=modules_to_save,
        lora_dropout=0.05,
        bias="none",
        task_type=None # Generic feature extraction
    )
    
    # Wrap model
    peft_model = get_peft_model(model, config)
    
    # Print trainable params
    peft_model.print_trainable_parameters()
    
    return peft_model
