import torch
import torch.nn.functional as F


# --- NEW HELPER FUNCTION 
def predict_keypoints(src_feats, trg_feats, src_kps, src_img_size, trg_img_size=None, temperature=10.0):
    """
    Helper to predict keypoints (Soft-Argmax) without calculating loss.
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
    
    raw_heatmaps = torch.bmm(src_vecs, trg_vecs)

    # 5. Soft-Argmax
    heatmaps = raw_heatmaps.view(B, -1, H_trg, W_trg)
    prob_map = F.softmax(heatmaps.view(B, -1, H_trg * W_trg) * temperature, dim=-1)
    prob_map = prob_map.view(B, -1, H_trg, W_trg)

    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, H_trg, device=trg_feats.device),
        torch.linspace(-1, 1, W_trg, device=trg_feats.device),
        indexing='ij'
    )
    
    pred_x = torch.sum(x_coords.unsqueeze(0).unsqueeze(0) * prob_map, dim=(-2, -1))
    pred_y = torch.sum(y_coords.unsqueeze(0).unsqueeze(0) * prob_map, dim=(-2, -1))
    
    return torch.stack([pred_x, pred_y], dim=-1)

# --- YOUR EXISTING LOSS FUNCTION (Unchanged logic, just calls the helper) ---
def loss(src_feats, trg_feats, src_kps, trg_kps, src_img_size, trg_img_size=None, temperature=10.0):
    if trg_img_size is None:
        trg_img_size = src_img_size
        
    H_trg_img, W_trg_img = trg_img_size
    
    # 1. Get Prediction (Reusing the helper to avoid code duplication)
    pred_kps = predict_keypoints(src_feats, trg_feats, src_kps, src_img_size, trg_img_size, temperature)

    # 2. Normalize Ground Truth Target Keypoints
    trg_norm = trg_kps.clone()
    trg_norm[:, :, 0] = 2.0 * (trg_kps[:, :, 0] / (W_trg_img - 1)) - 1.0
    trg_norm[:, :, 1] = 2.0 * (trg_kps[:, :, 1] / (H_trg_img - 1)) - 1.0

    # 3. Calculate Loss
    return F.mse_loss(pred_kps, trg_norm)







# def loss(src_feats, trg_feats, src_kps, trg_kps, src_img_size, trg_img_size=None, temperature=10.0):
#     """
#     Computes L_dense (Soft-Argmax + L2) robustly by decoupling Source and Target dimensions.
    
#     Args:
#         src_feats: Source features (B, C, H_src, W_src)
#         trg_feats: Target features (B, C, H_trg, W_trg)
#         src_kps: Source keypoints (B, N, 2) in pixel coordinates
#         trg_kps: Target keypoints (B, N, 2) in pixel coordinates
#         src_img_size: (H, W) tuple for source image size
#         trg_img_size: (H, W) tuple for target image size. If None, uses src_img_size.
#         temperature: Temperature for softmax sharpness (higher = sharper)
    
#     Returns:
#         loss: MSE loss between predicted and ground truth keypoints in [-1, 1] space
#     """
#     # Handle backwards compatibility: if trg_img_size not provided, use src_img_size
#     if trg_img_size is None:
#         trg_img_size = src_img_size
    
#     # 1. Get Dimensions Independently
#     # Source: Used ONLY to extract the descriptor at the keypoint
#     B, C, H_src, W_src = src_feats.shape
    
#     # Target: Used for the search space (heatmap size)
#     _, _, H_trg, W_trg = trg_feats.shape
    
#     H_src_img, W_src_img = src_img_size
#     H_trg_img, W_trg_img = trg_img_size

#     # 2. Normalize Source Keypoints (using SOURCE Image Dimensions)
#     # We use src_kps to sample from src_feats
#     src_grid = src_kps.clone()
#     src_grid[:, :, 0] = 2.0 * (src_kps[:, :, 0] / (W_src_img - 1)) - 1.0
#     src_grid[:, :, 1] = 2.0 * (src_kps[:, :, 1] / (H_src_img - 1)) - 1.0
#     src_grid = src_grid.unsqueeze(1) # (B, 1, N, 2)

#     # 3. Extract Source Descriptors
#     # Sample from Source Features (H_src, W_src)
#     src_desc = F.grid_sample(src_feats, src_grid, mode='bilinear', align_corners=True).squeeze(2)
#     # src_desc: (B, C, N)

#     # 4. Compute Similarity (The "Search")
#     # Compare Source Descriptors (B, C, N) vs Target Features (B, C, H_trg, W_trg)
#     src_vecs = src_desc.permute(0, 2, 1)      # (B, N, C)
#     trg_vecs = trg_feats.view(B, C, -1)       # (B, C, H_trg*W_trg) - Flatten Target
    
#     # Result: (B, N, H_trg*W_trg)
#     raw_heatmaps = torch.bmm(src_vecs, trg_vecs)

#     # 5. Soft-Argmax (Prediction)
#     # A. Reshape Heatmap to TARGET dimensions
#     heatmaps = raw_heatmaps.view(B, -1, H_trg, W_trg)
    
#     # B. Spatial Softmax
#     prob_map = F.softmax(heatmaps.view(B, -1, H_trg * W_trg) * temperature, dim=-1)
#     prob_map = prob_map.view(B, -1, H_trg, W_trg)

#     # C. Create Grid matching TARGET features
#     y_coords, x_coords = torch.meshgrid(
#         torch.linspace(-1, 1, H_trg, device=trg_feats.device),
#         torch.linspace(-1, 1, W_trg, device=trg_feats.device),
#         indexing='ij'
#     )
    
#     # D. Weighted Sum (Expectation)
#     # (1, 1, H, W) * (B, N, H, W) -> Sum over H, W
#     pred_x = torch.sum(x_coords.unsqueeze(0).unsqueeze(0) * prob_map, dim=(-2, -1))
#     pred_y = torch.sum(y_coords.unsqueeze(0).unsqueeze(0) * prob_map, dim=(-2, -1))
    
#     pred_kps = torch.stack([pred_x, pred_y], dim=-1) # (B, N, 2) in range [-1, 1]

#     # 6. Normalize Ground Truth Target Keypoints (using TARGET Image Dimensions)
#     trg_norm = trg_kps.clone()
#     trg_norm[:, :, 0] = 2.0 * (trg_kps[:, :, 0] / (W_trg_img - 1)) - 1.0
#     trg_norm[:, :, 1] = 2.0 * (trg_kps[:, :, 1] / (H_trg_img - 1)) - 1.0

#     # 7. Calculate Loss
#     return F.mse_loss(pred_kps, trg_norm)