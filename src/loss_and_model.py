import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Loss functions adapted from GeoAware-SC
# https://github.com/Junyi42/GeoAware-SC/blob/b20fab9c2d4f686536be9db34d1fb2079240fdd5/utils/utils_losses.py

def get_corr_map_loss(img1_desc, img2_desc, corr_map_net, img1_patch_idx,
                      gt_flow, num_patches=60, img2_patch_idx=None):
    """
    Calculate correlation map loss using predicted flow.
    
    Args:
        img1_desc: Descriptors for image 1, shape [1, num_patches^2, feature_dim]
        img2_desc: Descriptors for image 2, shape [1, num_patches^2, feature_dim]
        corr_map_net: Network that predicts flow from correlation maps
        img1_patch_idx: Patch indices for image 1
        gt_flow: Ground truth flow vectors
        num_patches: Number of patches per dimension (default 60)
        img2_patch_idx: Optional patch indices for image 2
    
    Returns:
        EPE_loss: End-point-error loss (mean L2 distance between predicted and GT flow)
    """
    # Compute correlation map [1, num_patches^2, num_patches^2]
    corr_map = torch.matmul(img1_desc, img2_desc.transpose(1, 2))
    
    # Reshape to spatial dimensions [1, num_patches, num_patches, num_patches, num_patches]
    corr_map = corr_map.reshape(1, num_patches, num_patches, num_patches, num_patches)
    
    # Feed through network to predict flow [1, num_patches, num_patches, 2]
    corr_map = corr_map_net(corr_map)
    corr_map = corr_map.reshape(1, num_patches * num_patches, 2)
    
    # Get predicted flow for the specified patches
    predict_flow = corr_map[0, img1_patch_idx, :]
    
    # Compute end-point error (L2 distance)
    EPE_loss = torch.norm(predict_flow - gt_flow, dim=-1).mean()
    
    return EPE_loss


def self_contrastive_loss(feat_map, instance_mask=None):
    """
    Self-contrastive loss that encourages local smoothness and global diversity.
    
    Args:
        feat_map: Feature map (B, C, H, W)
        instance_mask: Optional instance mask (B, H', W')
    
    Returns:
        loss: Combined local and global contrastive loss
    """
    B, C, H, W = feat_map.size()
    
    if instance_mask is not None:
        # Interpolate mask to feature map size
        instance_mask = F.interpolate(
            instance_mask.cuda().unsqueeze(1).float(),
            size=(H, W),
            mode='bilinear'
        ) > 0.5
        # Mask out the feature map
        feat_map = feat_map * instance_mask
        # Set masked regions to 1 to avoid affecting loss
        feat_map = feat_map + (~instance_mask)
    
    # Local loss: encourage similarity with 8-connected neighbors
    offsets = [(0, 1), (1, 0), (1, 1), (1, -1), (0, -1), (-1, 0), (-1, -1), (-1, 1)]
    local_loss = 0.0
    
    for i, j in offsets:
        # Shift feature map
        shifted_map = torch.roll(feat_map, shifts=(i, j), dims=(2, 3))
        # Compute dot product
        dot_product = (feat_map * shifted_map).sum(dim=1)
        
        # Mask out invalid regions (to avoid wrapping)
        if i > 0:
            dot_product[:, :i, :] = 0
        if j > 0:
            dot_product[:, :, :j] = 0
        if i < 0:
            dot_product[:, i:, :] = 0
        if j < 0:
            dot_product[:, :, j:] = 0
        
        # Negative because we want to maximize similarity with neighbors
        local_loss -= dot_product.mean()
    
    # Global loss: encourage dissimilarity with distant pixels
    num_samples = H * W
    idx_i = torch.randint(0, H, (num_samples,)).cuda()
    idx_j = torch.randint(0, W, (num_samples,)).cuda()
    idx_k = torch.randint(0, H, (num_samples,)).cuda()
    idx_l = torch.randint(0, W, (num_samples,)).cuda()
    
    # Ensure they are not neighbors
    mask = ((idx_k - idx_i).abs() > 1) | ((idx_l - idx_j).abs() > 1)
    
    if instance_mask is not None:
        mask = mask & instance_mask[0, 0, idx_i, idx_j] & instance_mask[0, 0, idx_k, idx_l]
    
    idx_i, idx_j, idx_k, idx_l = idx_i[mask], idx_j[mask], idx_k[mask], idx_l[mask]
    
    global_loss = 0.0
    for i, j, k, l in zip(idx_i, idx_j, idx_k, idx_l):
        dot_product = (feat_map[:, :, i, j] * feat_map[:, :, k, l]).sum(dim=1)
        # Positive because we want to minimize similarity with distant pixels
        global_loss += dot_product.mean()
    
    # Combine losses
    lambda_factor = 0.1
    loss = local_loss + lambda_factor * global_loss
    
    return loss


def get_logits(image_features, text_features, logit_scale):
    """Compute CLIP-style logits."""
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logit_scale * text_features @ image_features.T
    return logits_per_image, logits_per_text


def cal_clip_loss(image_features, text_features, logit_scale, self_logit_scale=None):
    """
    Calculate CLIP-style contrastive loss.
    
    Args:
        image_features: Image feature embeddings
        text_features: Text/target feature embeddings  
        logit_scale: Temperature parameter for scaling logits
        self_logit_scale: Optional separate scale for self-supervised component
    
    Returns:
        total_loss: Symmetric cross-entropy loss
    """
    device = image_features.device
    logits_per_image, logits_per_text = get_logits(image_features, text_features, logit_scale)
    
    labels = torch.arange(logits_per_image.shape[0], device=device, dtype=torch.long)
    
    total_loss = (
        F.cross_entropy(logits_per_image, labels) +
        F.cross_entropy(logits_per_text, labels)
    ) / 2
    
    return total_loss


def calculate_patch_indices_and_loss(args, kps_1, kps_2, desc_1, desc_2,
                                     scale_factor, num_patches, aggre_net, threshold,
                                     corr_map_net=None, device='cuda'):
    """
    Calculate patch indices and corresponding loss for keypoint matching.
    
    Args:
        args: Configuration arguments
        kps_1, kps_2: Keypoints for the two images
        desc_1, desc_2: Descriptors for the two images
        scale_factor, num_patches: Parameters for calculating patch indices
        aggre_net: Aggregation network for calculating loss
        threshold: Threshold for Gaussian augmentation
        corr_map_net: Correlation map network (required if DENSE_OBJ is True)
        device: Device for tensor operations
    
    Returns:
        loss: Calculated loss
    """
    def get_patch_idx(scale_factor, num_patches, y, x):
        """Convert keypoint coordinates to patch indices."""
        scaled_y = scale_factor * y
        scaled_x = scale_factor * x
        y_patch = scaled_y.astype(np.int32)
        x_patch = scaled_x.astype(np.int32)
        patch_idx = num_patches * y_patch + x_patch
        
        if args.DENSE_OBJ:
            return scaled_y, scaled_x, patch_idx
        else:
            return y_patch, x_patch, patch_idx
    
    # Extract coordinates
    y1, x1 = kps_1[:, 1].numpy(), kps_1[:, 0].numpy()
    y2, x2 = kps_2[:, 1].numpy(), kps_2[:, 0].numpy()
    
    # Get patch indices
    y_patch_1, x_patch_1, patch_idx_1 = get_patch_idx(scale_factor, num_patches, y1, x1)
    y_patch_2, x_patch_2, patch_idx_2 = get_patch_idx(scale_factor, num_patches, y2, x2)
    
    # Calculate loss based on training objective
    if not args.DENSE_OBJ:
        # Sparse objective: CLIP-style contrastive loss on patch descriptors
        desc_patch_1 = desc_1[0, patch_idx_1, :]
        desc_patch_2 = desc_2[0, patch_idx_2, :]
        loss = cal_clip_loss(
            desc_patch_1, desc_patch_2,
            aggre_net.logit_scale.exp(),
            self_logit_scale=aggre_net.self_logit_scale.exp()
        )
    else:
        # Dense objective: end-point error on predicted flow
        gt_flow = torch.stack([
            torch.tensor(x_patch_2) - torch.tensor(x_patch_1),
            torch.tensor(y_patch_2) - torch.tensor(y_patch_1)
        ], dim=-1).to(device)
        
        # Add Gaussian noise augmentation if specified
        if args.GAUSSIAN_AUGMENT > 0:
            std = args.GAUSSIAN_AUGMENT * threshold / 2
            noise = torch.randn_like(gt_flow, dtype=torch.float32) * std
            gt_flow += noise
        
        loss = get_corr_map_loss(
            desc_1, desc_2, corr_map_net, patch_idx_1,
            gt_flow, num_patches, img2_patch_idx=patch_idx_2
        )
    
    return loss


def calculate_loss(args, aggre_net, img1_kps, img2_kps, img1_desc, img2_desc,
                   img1_threshold, img2_threshold, mask1, mask2, num_patches, device,
                   raw_permute_list=None, img1_desc_flip=None, img2_desc_flip=None,
                   corr_map_net=None):
    """
    Main loss calculation function with support for dense/sparse objectives and contrastive loss.
    
    This version has flip augmentation options (ADAPT_FLIP, AUGMENT_DOUBLE_FLIP, AUGMENT_SELF_FLIP) removed.
    
    Args:
        args: Configuration with training parameters
        aggre_net: Aggregation network with learnable temperature parameters
        img1_kps, img2_kps: Keypoint annotations for both images
        img1_desc, img2_desc: Feature descriptors for both images
        img1_threshold, img2_threshold: Thresholds for each image
        mask1, mask2: Instance segmentation masks
        num_patches: Number of patches per dimension
        device: Computation device
        raw_permute_list: Permutation list for keypoint ordering (unused without flip augmentation)
        img1_desc_flip, img2_desc_flip: Flipped descriptors (unused without flip augmentation)
        corr_map_net: Network for predicting correspondence maps
    
    Returns:
        loss: Total computed loss
    """
    def get_patch_idx(args, scale_factor, num_patches, img1_y, img1_x):
        """Convert keypoint coordinates to patch indices."""
        scaled_img1_y = scale_factor * img1_y
        scaled_img1_x = scale_factor * img1_x
        img1_y_patch = scaled_img1_y.astype(np.int32)
        img1_x_patch = scaled_img1_x.astype(np.int32)
        img1_patch_idx = num_patches * img1_y_patch + img1_x_patch
        
        if args.DENSE_OBJ:
            return scaled_img1_y, scaled_img1_x, img1_patch_idx
        else:
            return img1_y_patch, img1_x_patch, img1_patch_idx
    
    # Filter by mutual visibility
    vis = (img1_kps[:, 2] * img2_kps[:, 2]).bool()
    scale_factor = num_patches / args.ANNO_SIZE
    
    # Get patch indices for both images
    img1_y, img1_x = img1_kps[vis, 1].numpy(), img1_kps[vis, 0].numpy()
    img1_y_patch, img1_x_patch, img1_patch_idx = get_patch_idx(
        args, scale_factor, num_patches, img1_y, img1_x
    )
    
    img2_y, img2_x = img2_kps[vis, 1].numpy(), img2_kps[vis, 0].numpy()
    img2_y_patch, img2_x_patch, img2_patch_idx = get_patch_idx(
        args, scale_factor, num_patches, img2_y, img2_x
    )
    
    # Base loss: CLIP-style contrastive loss on matched patches
    loss = cal_clip_loss(
        img1_desc[0, img1_patch_idx, :],
        img2_desc[0, img2_patch_idx, :],
        aggre_net.logit_scale.exp(),
        self_logit_scale=aggre_net.self_logit_scale.exp()
    )
    
    # Add dense objective loss if enabled
    if args.DENSE_OBJ > 0:
        flow_idx = img1_patch_idx
        flow_idx2 = img2_patch_idx
        
        gt_flow = torch.stack([
            torch.tensor(img2_x_patch) - torch.tensor(img1_x_patch),
            torch.tensor(img2_y_patch) - torch.tensor(img1_y_patch)
        ], dim=-1).to(device)
        
        # Add Gaussian noise augmentation if specified
        if args.GAUSSIAN_AUGMENT > 0:
            std = args.GAUSSIAN_AUGMENT * img2_threshold / 2  # 2 sigma within threshold
            noise = torch.randn_like(gt_flow, dtype=torch.float32) * std
            gt_flow = gt_flow + noise
        
        EPE_loss = get_corr_map_loss(
            img1_desc, img2_desc, corr_map_net,
            flow_idx, gt_flow, num_patches, img2_patch_idx=flow_idx2
        )
        loss += EPE_loss
    
    # Add self-contrastive loss if enabled
    if args.SELF_CONTRAST_WEIGHT > 0:
        contrast_loss1 = self_contrastive_loss(
            img1_desc.permute(0, 2, 1).reshape(-1, args.PROJ_DIM, num_patches, num_patches),
            mask1.unsqueeze(0)
        ) * args.SELF_CONTRAST_WEIGHT
        
        contrast_loss2 = self_contrastive_loss(
            img2_desc.permute(0, 2, 1).reshape(-1, args.PROJ_DIM, num_patches, num_patches),
            mask2.unsqueeze(0)
        ) * args.SELF_CONTRAST_WEIGHT
        
        contrast_loss = (contrast_loss1 + contrast_loss2) / 2 * args.SELF_CONTRAST_WEIGHT
        loss += contrast_loss
    
    return loss

class DinoV2KeypointDetector(nn.Module):
    def __init__(self, model_name='dinov2_vits14', num_keypoints=10, freeze_backbone=True, 
                 features_checkpoint=None):
        super().__init__()
        # Load precomputed features if checkpoint is provided
        self.precomputed_features = None
        self.feature_extractor = None
        
        # Determine embed_dim based on model variant
        embed_dim_map = {
            'dinov2_vits14': 384,
            'dinov2_vitb14': 768,
            'dinov2_vitl14': 1024,
            'dinov2_vitg14': 1536,
        }
        embed_dim = embed_dim_map.get(model_name, 384)
        
        if features_checkpoint is not None:
            print(f"Loading precomputed features from {features_checkpoint}...")
            checkpoint = torch.load(features_checkpoint)
            self.precomputed_features = checkpoint['features']
            print(f"Loaded {len(self.precomputed_features)} precomputed features")
        else:
            # Create feature extractor for on-the-fly extraction
            from src.dinov2_features import DINOv2FeatureExtractor
            print(f"Initializing DINOv2FeatureExtractor for on-the-fly extraction...")
            self.feature_extractor = DINOv2FeatureExtractor(model_name=model_name)
            if freeze_backbone:
                for param in self.feature_extractor.model.parameters():
                    param.requires_grad = False
        
        # 2. Prediction Head (Conv 1x1)
        # Transforms embed_dim channels into N heatmaps (one per keypoint)
        self.head = nn.Conv2d(embed_dim, num_keypoints, kernel_size=1)
        
        # Initialize head weights
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x, image_keys=None):
        """
        x: Immagini (B, 3, H, W) or precomputed features if using checkpoint
        image_keys: List of image keys (e.g., 'aeroplane/2008_000585') when using precomputed features
        Returns: List of heatmaps (one per image) when using precomputed features with different sizes,
                 or batched tensor (B, N, H, W) when all images have same size
        """
        if self.precomputed_features is not None and image_keys is not None:
            # Use precomputed features
            # Since images can have different sizes, we process each separately
            all_heatmaps = []
            for key in image_keys:
                feat_dict = self.precomputed_features[key]
                features = feat_dict['features']  # Shape: (1, 384, H_grid, W_grid)
                input_h, input_w = feat_dict['image_size']  # Original image size
                
                # Move features to the same device as the model
                features = features.to(next(self.parameters()).device)
                
                # Generate heatmaps for this single image
                heatmaps_low_res = self.head(features)
                heatmaps_high_res = F.interpolate(
                    heatmaps_low_res, 
                    size=(input_h, input_w), 
                    mode='bilinear', 
                    align_corners=False
                )
                all_heatmaps.append(heatmaps_high_res)
            
            # Return list of heatmaps (each can have different size)
            return all_heatmaps
        else:
            # Extract features on the fly
            input_h, input_w = x.shape[2], x.shape[3]
            features = self.feature_extractor.extract_features(x)
        
            # features è già in formato (B, C, H_grid, W_grid), pronto per Conv2d
            
            # Genera heatmap a bassa risoluzione (grid size di Dino)
            heatmaps_low_res = self.head(features)
            
            # Upsampling bilineare per matchare la risoluzione originale
            # Questo permette alla loss densa e al soft-argmax di lavorare con precisione pixel
            heatmaps_high_res = F.interpolate(
                heatmaps_low_res, 
                size=(input_h, input_w), 
                mode='bilinear', 
                align_corners=False
            )
            
            return heatmaps_high_res


# Test code
if __name__ == "__main__":
    print("Loss functions from GeoAware-SC successfully imported.")
    print("Available functions:")
    print("  - get_corr_map_loss: Correlation map loss with EPE")
    print("  - self_contrastive_loss: Local smoothness + global diversity")
    print("  - cal_clip_loss: CLIP-style contrastive loss")
    print("  - calculate_patch_indices_and_loss: Patch-based keypoint matching loss")
    print("  - calculate_loss: Main loss function (without flip augmentation)")
    print("\nNote: ADAPT_FLIP, AUGMENT_DOUBLE_FLIP, and AUGMENT_SELF_FLIP have been removed.")