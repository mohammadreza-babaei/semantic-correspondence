"""
Centralized checkpoint and weights loading utilities.

This module provides shared functions for loading model checkpoints across
different model types (SAM, DINOv2, DINOv3, MobileSAM) to eliminate code
duplication and ensure consistent behavior.
"""

import os
import torch


def safe_torch_load(path, map_location='cpu'):
    """Load a checkpoint file with proper handling for different PyTorch versions.
    
    Args:
        path: Path to the checkpoint file
        map_location: Device to load tensors to (default: 'cpu')
        
    Returns:
        Loaded checkpoint dictionary
        
    Raises:
        FileNotFoundError: If the path doesn't exist
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    try:
        # weights_only=False required for checkpoints with numpy scalars (newer torch)
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        # Fallback for older torch versions without weights_only parameter
        return torch.load(path, map_location=map_location)


def extract_state_dict(checkpoint, key_priority=None):
    """Extract the model state dict from a checkpoint, handling various formats.
    
    Supports checkpoints from:
    - Trainer.save_checkpoint() format (nested under 'model_state')
    - Model.get_model_state() format (with 'backbone_state_dict' or 'sam_model_state_dict')
    - Plain state dict (direct weights)
    
    Args:
        checkpoint: Loaded checkpoint dictionary
        key_priority: List of keys to try in order. Defaults to common formats:
                     ['model_state', 'backbone_state_dict', 'sam_model_state_dict']
        
    Returns:
        tuple: (state_dict, format_info) where format_info describes what was found
    """
    if key_priority is None:
        key_priority = ['model_state', 'backbone_state_dict', 'sam_model_state_dict']
    
    state_dict = checkpoint
    format_info = "plain"
    
    # Handle Trainer checkpoint format (model_state wraps everything)
    if 'model_state' in checkpoint:
        state_dict = checkpoint['model_state']
        format_info = f"training checkpoint (epoch {checkpoint.get('epoch', 'unknown')})"
        
        # Handle nested backbone_state_dict within model_state
        if isinstance(state_dict, dict) and 'backbone_state_dict' in state_dict:
            state_dict = state_dict['backbone_state_dict']
            format_info += " -> backbone_state_dict"
        elif isinstance(state_dict, dict) and 'sam_model_state_dict' in state_dict:
            state_dict = state_dict['sam_model_state_dict']
            format_info += " -> sam_model_state_dict"
    # Handle direct model state formats
    elif 'backbone_state_dict' in checkpoint:
        state_dict = checkpoint['backbone_state_dict']
        format_info = f"model state (model: {checkpoint.get('model_name', 'unknown')})"
    elif 'sam_model_state_dict' in checkpoint:
        state_dict = checkpoint['sam_model_state_dict']
        format_info = f"SAM model state (model: {checkpoint.get('model_name', 'unknown')})"
    
    return state_dict, format_info


# Common prefixes found in various checkpoint sources
DEFAULT_PREFIXES_TO_REMOVE = [
    "model.",           # Generic wrapper
    "base_model.model.", # Hugging Face format
    "teacher.",         # DINO teacher model
    "backbone.",        # Common backbone prefix
]


def clean_state_dict_keys(state_dict, prefixes_to_remove=None):
    """Remove common prefixes from state dict keys.
    
    Many checkpoints have keys like 'model.layer1.weight' that need to be
    cleaned to 'layer1.weight' for loading into the bare model.
    
    Args:
        state_dict: Dictionary of model weights
        prefixes_to_remove: List of prefixes to strip. Defaults to common ones.
        
    Returns:
        New state dict with cleaned keys
    """
    if prefixes_to_remove is None:
        prefixes_to_remove = DEFAULT_PREFIXES_TO_REMOVE
    
    new_state_dict = {}
    for k, v in state_dict.items():
        new_key = k
        for prefix in prefixes_to_remove:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]
        new_state_dict[new_key] = v
    
    return new_state_dict


def merge_lora_weights(state_dict):
    """Merge LoRA adapter weights back into the base layer weights.
    
    Converts a state dict with LoRA structure:
        layer.base_layer.weight
        layer.lora_A.default.weight
        layer.lora_B.default.weight
    
    Into a standard state dict:
        layer.weight
    
    The merged weight is: W = W_base + (lora_B @ lora_A) * scaling
    
    Args:
        state_dict: State dict potentially containing LoRA weights
        
    Returns:
        New state dict with LoRA weights merged into base layers
    """
    # Check if this is a LoRA checkpoint
    has_lora = any('.lora_A.' in k or '.lora_B.' in k for k in state_dict.keys())
    if not has_lora:
        return state_dict
    
    print("Detected LoRA checkpoint - merging LoRA weights into base layers...")
    
    new_state_dict = {}
    processed_layers = set()
    
    for key in state_dict.keys():
        # Handle LoRA layers
        if '.lora_A.' in key or '.lora_B.' in key:
            # Extract the base layer name
            # Format: "prefix.layer_name.lora_A.default.weight" -> "prefix.layer_name"
            if '.lora_A.' in key:
                base_name = key.split('.lora_A.')[0]
            else:
                base_name = key.split('.lora_B.')[0]
            
            if base_name in processed_layers:
                continue
            
            # Get all related keys
            base_weight_key = f"{base_name}.base_layer.weight"
            base_bias_key = f"{base_name}.base_layer.bias"
            lora_a_key = f"{base_name}.lora_A.default.weight"
            lora_b_key = f"{base_name}.lora_B.default.weight"
            
            # Check if we have the necessary components
            if base_weight_key in state_dict and lora_a_key in state_dict and lora_b_key in state_dict:
                # Merge: W = W_base + lora_B @ lora_A
                base_weight = state_dict[base_weight_key]
                lora_a = state_dict[lora_a_key]
                lora_b = state_dict[lora_b_key]
                
                # Compute LoRA delta: lora_B @ lora_A
                # lora_A: (rank, in_features), lora_B: (out_features, rank)
                lora_delta = torch.mm(lora_b, lora_a)
                
                # Merge into base weight
                merged_weight = base_weight + lora_delta
                
                # Save with standard naming (remove .base_layer)
                output_key = f"{base_name}.weight"
                new_state_dict[output_key] = merged_weight
                
                # Handle bias if present
                if base_bias_key in state_dict:
                    new_state_dict[f"{base_name}.bias"] = state_dict[base_bias_key]
                
                processed_layers.add(base_name)
            else:
                # Incomplete LoRA layer - skip for now
                continue
        
        # Handle base_layer keys that weren't merged (no LoRA adapters)
        elif '.base_layer.' in key:
            # This is a base layer without LoRA adapters
            base_name = key.split('.base_layer.')[0]
            param_name = key.split('.base_layer.')[1]  # weight or bias
            
            if base_name not in processed_layers:
                output_key = f"{base_name}.{param_name}"
                new_state_dict[output_key] = state_dict[key]
        
        # Regular keys (not LoRA-related)
        else:
            new_state_dict[key] = state_dict[key]
    
    print(f"Merged {len(processed_layers)} LoRA layers")
    return new_state_dict


def load_weights(model, path, map_location='cpu', strict=True, clean_keys=True, 
                 key_priority=None, verbose=True):
    """High-level function to load weights into a model.
    
    Combines safe loading, state dict extraction, key cleaning, and application.
    Automatically handles LoRA checkpoints by merging adapters into base weights.
    
    Args:
        model: PyTorch model (nn.Module) to load weights into
        path: Path to the checkpoint file
        map_location: Device to load tensors to
        strict: Whether to require exact key matching (passed to load_state_dict)
        clean_keys: Whether to clean common prefixes from keys
        key_priority: Custom key priority for extraction
        verbose: Whether to print loading information
        
    Returns:
        Message from load_state_dict (missing/unexpected keys)
        
    Raises:
        FileNotFoundError: If the path doesn't exist
    """
    # Load checkpoint
    checkpoint = safe_torch_load(path, map_location=map_location)
    
    # Extract state dict
    state_dict, format_info = extract_state_dict(checkpoint, key_priority)
    if verbose:
        print(f"Loading from {format_info}")
    
    # Merge LoRA weights if present
    state_dict = merge_lora_weights(state_dict)
    
    # Clean keys if requested
    if clean_keys:
        state_dict = clean_state_dict_keys(state_dict)
    
    # Load into model
    msg = model.load_state_dict(state_dict, strict=strict)
    
    if verbose:
        print(f"Weights loaded successfully!")
        if msg.missing_keys:
            print(f"  Missing keys: {len(msg.missing_keys)}")
        if msg.unexpected_keys:
            print(f"  Unexpected keys: {len(msg.unexpected_keys)}")
    
    return msg
