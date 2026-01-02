"""
Centralized checkpoint and weights loading utilities.

This module provides shared functions for loading model checkpoints across
different model types (SAM, DINOv2, DINOv3, TinyViT) to eliminate code
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


def get_merged_state_dict(model):
    """Get a clean state dict from a model, merging LoRA weights if present.
    
    If the model is a PEFT model with LoRA adapters, this will merge the adapters
    into the base weights and return a standard state dict with no LoRA keys.
    
    This should be called at checkpoint save time to produce portable weights
    that can be loaded without PEFT.
    
    IMPORTANT: This function works on a COPY of the model to preserve the original
    LoRA adapters for continued training.
    
    Args:
        model: The PyTorch model (may be wrapped with PEFT)
        
    Returns:
        A clean state dict with merged weights
    """
    try:
        import peft
        import copy
        if isinstance(model, peft.PeftModel):
            print("Merging LoRA adapters into base model for checkpoint...")
            # CRITICAL: Copy the model first, since merge_and_unload() modifies in-place
            # and would destroy the LoRA adapters on the original training model
            model_copy = copy.deepcopy(model)
            merged_model = model_copy.merge_and_unload(progressbar=False)
            state_dict = merged_model.state_dict()
            print(f"LoRA merge complete - {len(state_dict)} parameters")
            # Cleanup the copy to free memory
            del model_copy, merged_model
            return state_dict
    except ImportError:
        pass
    
    # Not a PEFT model, return regular state dict
    return model.state_dict()


def load_weights(model, path, map_location='cpu', strict=True, clean_keys=True, 
                 key_priority=None, verbose=True):
    """High-level function to load weights into a model.
    
    Combines safe loading, state dict extraction, key cleaning, and application.
    
    Note: LoRA weights should be merged at save time using get_merged_state_dict(),
    so checkpoints should contain standard weights that load directly.
    
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
