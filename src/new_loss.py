import torch
import torch.nn.functional as F

def loss(src_feats, trg_feats, src_kps, trg_kps, img_size, temperature=10.0):
    """
    Computes L_dense using Soft-Argmax and Euclidean Distance as described in the PDF.
    
    Args:
        src_feats: Raw output from the backbone model (B, C, H_f, W_f)
        trg_feats: Raw output for the target image (B, C, H_f, W_f)
        src_kps: coordinates of the points you want to track in the source image (B, num_keypoints, 2)
        trg_kps: coordinates of where the keypoint is on the target image (B, num_keypoints, 2)
        img_size: original dimension of the input image (840, 840)
        temperature: Controls sharpness of the heatmap.

    Note: B, num_keypoints, 2), that '2' is how many coordinates for each keypoint.
    """

    # (batch size, channels, height_feat, weidth_feat)
    B, C, H_f, W_f = src_feats.shape
    H_img, W_img = img_size
    
    # --- 1. PREPARATION ---
    # The goal is to convert pixels coordinates (e.g. x=420, y=100) into
    # a standard range [-1, 1].
    # (B, N, 2) -> (B, 1, N, 2)
    src_grid = src_kps.clone()
    src_grid[:, :, 0] = 2.0 * (src_kps[:, :, 0] / (W_img - 1)) - 1.0
    src_grid[:, :, 1] = 2.0 * (src_kps[:, :, 1] / (H_img - 1)) - 1.0
    src_grid = src_grid.unsqueeze(1) 

    # Normalize Target GT to [-1, 1] for loss calculation
    trg_norm = trg_kps.clone()
    trg_norm[:, :, 0] = 2.0 * (trg_kps[:, :, 0] / (W_img - 1)) - 1.0
    trg_norm[:, :, 1] = 2.0 * (trg_kps[:, :, 1] / (H_img - 1)) - 1.0

    # --- 2. EXTRACT SOURCE DESCRIPTORS ---
    # Sample the feature vector at the specific Source Keypoint locations
    # Result: (B, C, 1, N) -> squeeze to (B, C, N)
    src_desc = F.grid_sample(src_feats, src_grid, mode='bilinear', align_corners=True).squeeze(2)

    # --- 3. COMPUTE SIMILARITY MAP ---
    # Compare Source Descriptors (B, C, N) vs All Target Features (B, C, H_f, W_f)
    # We flatten target features to (B, C, H*W) for matrix multiplication
    B, N, _ = src_desc.permute(0, 2, 1).shape
    
    src_vecs = src_desc.permute(0, 2, 1)      # (B, N, C)
    trg_vecs = trg_feats.view(B, C, -1)       # (B, C, H*W)
    
    # Result: (B, N, H*W) - The heatmap for every keypoint
    heatmaps = torch.bmm(src_vecs, trg_vecs)

    # --- 4. SOFT-ARGMAX ---
    # A. Spatial Softmax: Convert scores to probabilities summing to 1
    # Reshape to spatial (B, N, H, W) for softmax
    heatmaps = heatmaps.view(B, N, H_f, W_f)
    prob_map = F.softmax(heatmaps.view(B, N, -1) * temperature, dim=-1).view(B, N, H_f, W_f)

    # B. Compute Expected Coordinates (Center of Mass)
    # Generate coordinate grid [-1, 1]
    y_coords, x_coords = torch.meshgrid(
        torch.linspace(-1, 1, H_f, device=src_feats.device),
        torch.linspace(-1, 1, W_f, device=src_feats.device),
        indexing='ij'
    )
    
    # Weighted Sum: sum(coordinate * probability)
    # Output shape: (B, N)
    pred_x = torch.sum(x_coords.unsqueeze(0).unsqueeze(0) * prob_map, dim=(-2, -1))
    pred_y = torch.sum(y_coords.unsqueeze(0).unsqueeze(0) * prob_map, dim=(-2, -1))
    
    # Stack to get (B, N, 2)
    pred_kps = torch.stack([pred_x, pred_y], dim=-1)

    # --- 5. EUCLIDEAN LOSS ---
    # Compare Predicted Coordinates vs Ground Truth
    # L2 Distance squared (MSE)
    return F.mse_loss(pred_kps, trg_norm)