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