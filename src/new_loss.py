import torch
import torch.nn.functional as F

def loss(src_feats, trg_feats, src_kps, trg_kps, img_size, temperature=10.0):
    """
    Computes L_dense (Soft-Argmax + L2) robustly by decoupling Source and Target dimensions.
    """
    # 1. Get Dimensions Independently
    # Source: Used ONLY to extract the descriptor at the keypoint
    B, C, H_src, W_src = src_feats.shape
    
    # Target: Used for the search space (heatmap size)
    _, _, H_trg, W_trg = trg_feats.shape
    
    H_img, W_img = img_size

    # 2. Normalize Source Keypoints (using Image Dimensions)
    # We use src_kps to sample from src_feats
    src_grid = src_kps.clone()
    src_grid[:, :, 0] = 2.0 * (src_kps[:, :, 0] / (W_img - 1)) - 1.0
    src_grid[:, :, 1] = 2.0 * (src_kps[:, :, 1] / (H_img - 1)) - 1.0
    src_grid = src_grid.unsqueeze(1) # (B, 1, N, 2)

    # 3. Extract Source Descriptors
    # Sample from Source Features (H_src, W_src)
    src_desc = F.grid_sample(src_feats, src_grid, mode='bilinear', align_corners=True).squeeze(2)
    # src_desc: (B, C, N)

    # 4. Compute Similarity (The "Search")
    # Compare Source Descriptors (B, C, N) vs Target Features (B, C, H_trg, W_trg)
    src_vecs = src_desc.permute(0, 2, 1)      # (B, N, C)
    trg_vecs = trg_feats.view(B, C, -1)       # (B, C, H_trg*W_trg) - Flatten Target
    
    # Result: (B, N, H_trg*W_trg)
    raw_heatmaps = torch.bmm(src_vecs, trg_vecs)

    # 5. Soft-Argmax (Prediction)
    # A. Reshape Heatmap to TARGET dimensions
    # FIX: Use H_trg and W_trg here, NOT H_src/W_src
    heatmaps = raw_heatmaps.view(B, -1, H_trg, W_trg)
    
    # B. Spatial Softmax
    prob_map = F.softmax(heatmaps.view(B, -1, H_trg * W_trg) * temperature, dim=-1)
    prob_map = prob_map.view(B, -1, H_trg, W_trg)

    # C. Create Grid matching TARGET features
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, H_trg, device=trg_feats.device),
        torch.linspace(-1, 1, W_trg, device=trg_feats.device),
        indexing='ij'
    )
    
    # D. Weighted Sum (Expectation)
    # (1, 1, H, W) * (B, N, H, W) -> Sum over H, W
    pred_x = torch.sum(x_coords.unsqueeze(0).unsqueeze(0) * prob_map, dim=(-2, -1))
    pred_y = torch.sum(y_coords.unsqueeze(0).unsqueeze(0) * prob_map, dim=(-2, -1))
    
    pred_kps = torch.stack([pred_x, pred_y], dim=-1) # (B, N, 2) in range [-1, 1]

    # 6. Normalize Ground Truth Target Keypoints
    trg_norm = trg_kps.clone()
    trg_norm[:, :, 0] = 2.0 * (trg_kps[:, :, 0] / (W_img - 1)) - 1.0
    trg_norm[:, :, 1] = 2.0 * (trg_kps[:, :, 1] / (H_img - 1)) - 1.0

    # 7. Calculate Loss
    return F.mse_loss(pred_kps, trg_norm)