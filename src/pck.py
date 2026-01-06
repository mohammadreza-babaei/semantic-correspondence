import torch
import numpy as np

def get_bbox_size(bbox):
    """
    Calculates the size (max dimension) of the bounding box.
    
    Args:
        bbox (torch.Tensor or np.ndarray): Bounding box in format [xmin, ymin, xmax, ymax].
                                           Shape: (B, 4) or (4,)
    
    Returns:
        torch.Tensor or np.ndarray: Size of the bounding box (max(w, h)).
    """
    # Calculate width and height assuming [xmin, ymin, xmax, ymax]
    # bbox[..., 2] is xmax, bbox[..., 0] is xmin
    # bbox[..., 3] is ymax, bbox[..., 1] is ymin
    width = bbox[..., 2] - bbox[..., 0]
    height = bbox[..., 3] - bbox[..., 1]
    
    if isinstance(bbox, torch.Tensor):
        return torch.max(width, height)
    else:
        return np.maximum(width, height)

def compute_pck(pred_kps, gt_kps, bbox, alpha=0.1, mask=None, size_type='bbox'):
    """
    Calculates Percentage of Correct Keypoints (PCK).
    
    Based on the "Emergent Correspondence from Image Diffusion" paper,
    the standard metric is PCK@0.1 (bbox).
    
    Args:
        pred_kps (torch.Tensor): Predicted keypoints (B, N, 2) or (N, 2).
        gt_kps (torch.Tensor): Ground truth keypoints (B, N, 2) or (N, 2).
        bbox (torch.Tensor): Bounding box for the target object.
                             Shape (B, 4) or (4,).
                             Format assumed: [xmin, ymin, xmax, ymax].
        alpha (float): Threshold factor. Default is 0.1.
        mask (torch.Tensor, optional): Boolean mask indicating valid/visible keypoints.
                                       Shape (B, N) or (N,).
        size_type (str): 'bbox' (default) or 'img'. 
                         If 'img', bbox is treated as image size [w, h] or similar 
                         (though usually passed explicitly). 
                         Here we assume 'bbox' mode uses the object bounding box.

    Returns:
        pck (float): The mean PCK score (0.0 to 1.0).
        correct_mask (torch.Tensor): Boolean mask of correct predictions.
    """
    # Ensure inputs are Tensors
    if not isinstance(pred_kps, torch.Tensor):
        pred_kps = torch.tensor(pred_kps)
    if not isinstance(gt_kps, torch.Tensor):
        gt_kps = torch.tensor(gt_kps)
    if not isinstance(bbox, torch.Tensor):
        bbox = torch.tensor(bbox)
        
    # Calculate Euclidean distance between prediction and ground truth
    # Shape: (B, N) or (N,)
    dist = torch.norm(pred_kps - gt_kps, dim=-1)
    
    # Calculate threshold based on bounding box size
    # size shape: (B,) or scalar
    bbox_size = get_bbox_size(bbox)
    
    # Handle broadcasting if necessary
    if bbox_size.ndim > 0 and dist.ndim > 1:
        threshold = alpha * bbox_size.unsqueeze(1) # (B, 1)
    else:
        threshold = alpha * bbox_size
        
    # Determine correct keypoints
    correct = dist <= threshold
    
    # Apply mask if provided
    if mask is not None:
        if not isinstance(mask, torch.Tensor):
            mask = torch.tensor(mask).bool()
        
        # Only consider masked (valid) points
        valid_correct = correct[mask]
        
        if valid_correct.numel() == 0:
            return 0.0, correct
            
        pck_score = valid_correct.float().mean().item()
    else:
        pck_score = correct.float().mean().item()
        
    return pck_score, correct

def compute_pck_from_batch(batch, pred_kps, alpha=0.1):
    """
    Helper to compute PCK for a batch from SPair71k dataset.
    
    Args:
        batch (dict): Batch dictionary from SPair71kPairs.
        pred_kps (torch.Tensor): Predicted keypoints for target image.
        alpha (float): Threshold.
        
    Returns:
        float: PCK score for the batch.
    """
    gt_kps = batch['trg_kps']
    bbox = batch['trg_bndbox']
    
    # Determine valid keypoints (not (0,0) or marked visible)
    # SPair annotations often use specific flags, but raw coordinates are usually sufficient.
    # Often only keypoints present in both source and target are evaluated.
    # Here we assume the predictor outputs predictions for all N keypoints.
    
    # Intersection check if needed, but usually kps in SPair pairs are already the intersection
    # or padded. If padded with -1 or 0, we should mask them.
    
    # Simple visibility check: gt_kps not (0,0) or (-1, -1)
    # Adjust based on specific dataset preprocessing.
    mask = (gt_kps[..., 0] > 0) & (gt_kps[..., 1] > 0)
    
    return compute_pck(pred_kps, gt_kps, bbox, alpha, mask)


def compute_raw_distances(pred_kps, gt_kps, bbox):
    """
    Returns raw Euclidean distances and the threshold size for each sample.
    
    Args:
        pred_kps: (B, N, 2)
        gt_kps: (B, N, 2)
        bbox: (B, 4)
        
    Returns:
        dist: (B, N) - Distance in pixels
        bbox_size: (B,) - Max bbox dimension in pixels
    """
    if not isinstance(pred_kps, torch.Tensor): pred_kps = torch.tensor(pred_kps)
    if not isinstance(gt_kps, torch.Tensor): gt_kps = torch.tensor(gt_kps)
    if not isinstance(bbox, torch.Tensor): bbox = torch.tensor(bbox)
        
    dist = torch.norm(pred_kps - gt_kps, dim=-1) # (B, N)
    bbox_size = get_bbox_size(bbox)              # (B,)
    
    return dist, bbox_size