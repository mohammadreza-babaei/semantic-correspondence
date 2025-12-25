import torch
import torch.nn.functional as F

def predict_keypoints(src_feats, trg_feats, src_kps, src_img_size, trg_img_size=None, 
                      temperature=0.1):
    """
    Helper to predict keypoints (Soft-Argmax) without calculating loss.
    
    Args:
        src_feats: Source features (B, C, H, W)
        trg_feats: Target features (B, C, H, W)
        src_kps: Source keypoints (B, N, 2) [x, y]
        src_img_size: (H, W) of source image
        trg_img_size: (H, W) of target image
        temperature: Softmax temperature (default: 0.1). Lower = sharper predictions.
                            
    Returns: Predicted keypoints in NORMALIZED coordinates [-1, 1].
    """
    if trg_img_size is None:
        trg_img_size = src_img_size
    
    # 1. Get Dimensions
    B, C, H_src, W_src = src_feats.shape
    _, _, H_trg, W_trg = trg_feats.shape
    H_src_img, W_src_img = src_img_size

    # 2. Normalize Source Keypoints (using SOURCE Image Dimensions)
    src_grid = src_kps.clone()
    src_grid[:, :, 0] = 2.0 * (src_kps[:, :, 0] / (W_src_img - 1)) - 1.0
    src_grid[:, :, 1] = 2.0 * (src_kps[:, :, 1] / (H_src_img - 1)) - 1.0
    src_grid = src_grid.unsqueeze(1) # (B, 1, N, 2)

    # 3. Extract Source Descriptors
    src_desc = F.grid_sample(src_feats, src_grid, mode='bilinear', align_corners=True).squeeze(2)

    # 4. Compute Similarity
    src_vecs = src_desc.permute(0, 2, 1)      # (B, N, C)
    trg_vecs = trg_feats.view(B, C, -1)       # (B, C, H*W)
    
    # Always use Cosine Similarity (L2 normalize)
    # Note: We manually normalize and use bmm (dot product) because it is the most 
    # memory-efficient way to compute pairwise cosine similarity for all points.
    # F.cosine_similarity would require broadcasting to (B, N, HW, C) which is O(N*HW*C) memory.
    src_vecs = F.normalize(src_vecs, p=2, dim=-1)
    trg_vecs = F.normalize(trg_vecs, p=2, dim=1)
    
    raw_heatmaps = torch.bmm(src_vecs, trg_vecs)

    # 5. Soft-Argmax
    heatmaps = raw_heatmaps.view(B, -1, H_trg * W_trg)
    
    # Numerical stability: subtract max before softmax
    heatmaps = heatmaps - heatmaps.max(dim=-1, keepdim=True)[0]
    
    prob_map = F.softmax(heatmaps / temperature, dim=-1)
    prob_map = prob_map.view(B, -1, H_trg, W_trg)

    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, H_trg, device=trg_feats.device),
        torch.linspace(-1, 1, W_trg, device=trg_feats.device),
        indexing='ij'
    )
    
    pred_x = torch.sum(x_coords.unsqueeze(0).unsqueeze(0) * prob_map, dim=(-2, -1))
    pred_y = torch.sum(y_coords.unsqueeze(0).unsqueeze(0) * prob_map, dim=(-2, -1))
    
    return torch.stack([pred_x, pred_y], dim=-1)


import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

def predict_keypoints_window(src_feats, trg_feats, src_kps, src_img_size, trg_img_size=None, 
                             window_size=5, temperature=0.1):
    """
    Refined prediction using Window Soft-Argmax.
    """
    if trg_img_size is None:
        trg_img_size = src_img_size
    
    B, C, H_src, W_src = src_feats.shape
    _, _, H_trg, W_trg = trg_feats.shape
    H_src_img, W_src_img = src_img_size

    # --- 1. Extract Descriptors (Same as Global) ---
    src_grid = src_kps.clone()
    src_grid[:, :, 0] = 2.0 * (src_kps[:, :, 0] / (W_src_img - 1)) - 1.0
    src_grid[:, :, 1] = 2.0 * (src_kps[:, :, 1] / (H_src_img - 1)) - 1.0
    src_grid = src_grid.unsqueeze(1)

    src_desc = F.grid_sample(src_feats, src_grid, mode='bilinear', align_corners=True).squeeze(2)

    # --- 2. Compute Heatmaps (Same as Global) ---
    src_vecs = F.normalize(src_desc.permute(0, 2, 1), p=2, dim=-1)
    trg_vecs = F.normalize(trg_feats.view(B, C, -1), p=2, dim=1)
    raw_heatmaps = torch.bmm(src_vecs, trg_vecs) 
    heatmaps = raw_heatmaps.view(B, -1, H_trg, W_trg) # (B, N, H, W)

    # --- 3. Window Refinement Logic ---
    
    # Find coarse peak (Hard Argmax)
    flat_max_indices = torch.argmax(raw_heatmaps, dim=-1) # (B, N)
    center_y = torch.div(flat_max_indices, W_trg, rounding_mode='floor') # Robust integer division
    center_x = flat_max_indices % W_trg

    radius = window_size // 2
    pred_kps_pix = []

    # Iterate batch and keypoints
    for b in range(B):
        kps_list_b = []
        for n in range(src_kps.shape[1]):
            cx, cy = center_x[b, n].item(), center_y[b, n].item()
            
            # Define window boundaries (clamped)
            x_min = max(0, cx - radius)
            x_max = min(W_trg, cx + radius + 1)
            y_min = max(0, cy - radius)
            y_max = min(H_trg, cy + radius + 1)
            
            # Extract patch
            patch = heatmaps[b, n, y_min:y_max, x_min:x_max]
            
            # Numerical Stability: subtract max to prevent NaN in softmax
            patch = patch - patch.max()
            prob_patch = F.softmax(patch / temperature, dim=None)
            
            # Create local grid
            # Critical: Ensure grid is on same device and dtype as prob_patch
            grid_y, grid_x = torch.meshgrid(
                torch.arange(y_min, y_max, device=patch.device, dtype=patch.dtype),
                torch.arange(x_min, x_max, device=patch.device, dtype=patch.dtype),
                indexing='ij'
            )
            
            # Expectation (Center of Mass)
            refined_x = torch.sum(grid_x * prob_patch)
            refined_y = torch.sum(grid_y * prob_patch)
            
            kps_list_b.append(torch.stack([refined_x, refined_y]))
        
        pred_kps_pix.append(torch.stack(kps_list_b))

    # Stack to (B, N, 2)
    pred_kps_pix = torch.stack(pred_kps_pix)

    # --- 4. Normalize to [-1, 1] ---
    # This matches the output format of the global predict_keypoints function
    pred_norm = pred_kps_pix.clone()
    pred_norm[:, :, 0] = 2.0 * (pred_kps_pix[:, :, 0] / (W_trg - 1)) - 1.0
    pred_norm[:, :, 1] = 2.0 * (pred_kps_pix[:, :, 1] / (H_trg - 1)) - 1.0
    
    return pred_norm

def loss(src_feats, trg_feats, src_kps, trg_kps, src_img_size, trg_img_size=None, 
         temperature=0.1):
    """
    Computes the keypoint correspondence loss.
    
    Args:
        ...
        src_kps: Source keypoints (B, N, 2).
        trg_kps: Target keypoints (B, N, 2).
        
    IMPORTANT: This function expects that invisible/invalid keypoints have already
    been filtered out. Do NOT pass keypoints with visibility=0.
    """
    if trg_img_size is None:
        trg_img_size = src_img_size
        
    H_trg_img, W_trg_img = trg_img_size
    
    # 1. Get Prediction (Reusing the helper to avoid code duplication)
    pred_kps = predict_keypoints(src_feats, trg_feats, src_kps, src_img_size, trg_img_size, 
                                 temperature=temperature)

    # 2. Normalize Ground Truth Target Keypoints
    trg_norm = trg_kps.clone()
    trg_norm[:, :, 0] = 2.0 * (trg_kps[:, :, 0] / (W_trg_img - 1)) - 1.0
    trg_norm[:, :, 1] = 2.0 * (trg_kps[:, :, 1] / (H_trg_img - 1)) - 1.0

    # 3. Calculate Loss
    return F.mse_loss(pred_kps, trg_norm)