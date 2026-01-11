import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA
from PIL import Image
import os

def plot_training_history(history, save_path=None):
    """
    Plot training history after training is complete.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'epoch_train_losses'
        save_path: Optional path to save the figure
    """


    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Epoch losses
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    val_epochs = range(1, len(history['val_loss']) + 1)
    axes[0].plot(val_epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss per Epoch', fontsize=14)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Batch losses with smoothing
    if history.get('epoch_train_losses'):
        batch_losses = history['epoch_train_losses']
        window = min(100, len(batch_losses) // 5 + 1)
        if window > 1:
            smoothed = np.convolve(batch_losses, np.ones(window)/window, mode='valid')
            axes[1].plot(range(len(smoothed)), smoothed, 'g-', linewidth=2, 
                         label=f'Smoothed (w={window})')
        axes[1].plot(batch_losses, 'b-', alpha=0.2, label='Raw')
        axes[1].set_xlabel('Batch', fontsize=12)
        axes[1].set_ylabel('Loss', fontsize=12)
        axes[1].set_title('Batch-level Loss', fontsize=14)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    

    # Plot 3: PCK (Validation)
    val_epochs = range(1, len(history['val_pck_global']) + 1)
    axes[2].plot(val_epochs, history['val_pck_global'], "g-o", label="Global PCK")
    axes[2].plot(val_epochs, history['val_pck_window'], "m-s", label="Window PCK")
    axes[2].set_title("PCK over Epochs")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("PCK")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    

    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.close(fig)
    return fig

def _update_plots(history, fig, axes, current_epoch):
        """Update training visualization plots."""
        axes[0].clear()
        axes[1].clear()
        
        # Plot 1: Epoch-level losses
        epochs = range(1, len(history['train_loss']) + 1)
        axes[0].plot(epochs, history['train_loss'], 'b-o', 
                     label='Train Loss', linewidth=2, markersize=6)
        if history['val_loss']:
            axes[0].plot(epochs, history['val_loss'], 'r-s', 
                         label='Val Loss', linewidth=2, markersize=6)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].set_title(f'Training Progress (Epoch {current_epoch})', fontsize=14)
        axes[0].legend(loc='upper right', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(0.5, max(current_epoch, 1) + 0.5)
        
        # Plot 2: Batch-level losses (smoothed)
        if len(history['epoch_train_losses']) > 0:
            batch_losses = history['epoch_train_losses']
            
            # Moving average smoothing
            window_size = min(50, len(batch_losses) // 10 + 1)
            if window_size > 1:
                smoothed = np.convolve(batch_losses, 
                                       np.ones(window_size)/window_size, 
                                       mode='valid')
                x_smooth = range(window_size // 2, window_size // 2 + len(smoothed))
                axes[1].plot(x_smooth, smoothed, 'g-', 
                             label=f'Smoothed (window={window_size})', 
                             linewidth=2, alpha=0.9)
            
            # Raw losses (semi-transparent)
            axes[1].plot(batch_losses, 'b-', alpha=0.3, 
                         label='Raw', linewidth=0.5)
            
            axes[1].set_xlabel('Batch', fontsize=12)
            axes[1].set_ylabel('Loss', fontsize=12)
            axes[1].set_title('Batch-level Training Loss', fontsize=14)
            axes[1].legend(loc='upper right', fontsize=10)
            axes[1].grid(True, alpha=0.3)
        
        fig.tight_layout()
        fig.canvas.draw()
        fig.canvas.flush_events()


def plot_block_weights(block, block_idx, save_path):
    """
    Visualize the weights of a transformer block as heatmaps.
    
    Args:
        block: The transformer block module (nn.Module)
        block_idx: Index of the block
        save_path: Path to save the visualization
    """
    # Identify weights to plot (focusing on Attention matrices)
    weights = {}
    
    for name, param in block.named_parameters():
        # process only 2D weights for heatmaps
        if len(param.shape) != 2:
            continue
            
        if 'attn.qkv' in name and 'weight' in name:
            weights[f'Attention QKV'] = param.data.cpu().numpy()
        elif 'attn.proj' in name and 'weight' in name:
            weights[f'Attention Output'] = param.data.cpu().numpy()
            
    if not weights:
        print(f"No 2D attention weights found for block {block_idx}")
        return

    n_weights = len(weights)
    
    fig, axes = plt.subplots(1, n_weights, figsize=(8 * n_weights, 8))
    if n_weights == 1:
        axes = [axes]
    
    for i, (name, data) in enumerate(weights.items()):
        ax = axes[i]
        
        # Plot heatmap
        # Use a diverging colormap centered at 0
        limit = max(abs(data.min()), abs(data.max()))
        im = ax.imshow(data, cmap='seismic', vmin=-limit, vmax=limit, interpolation='nearest', aspect='auto')
        
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
        ax.set_title(f"{name}\nShape: {data.shape}", fontsize=12)
        ax.set_xlabel("Output Dim")
        ax.set_ylabel("Input Dim")
        
    plt.suptitle(f"Attention Weights - Block {block_idx}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    if save_path:
        plt.savefig(save_path, dpi=150)
        
    plt.close(fig)


def compute_pca(feat1, feat2):
    """
    Compute PCA for a pair of feature maps to visualize them in RGB.
    Uses the same PCA basis for both images to ensure comparable colors.
    
    Args:
        feat1, feat2: Feature tensors of shape (1, C, H, W)
        
    Returns:
        pca1, pca2: Numpy arrays (H, W, 3) with values in [0, 1]
    """
    # Remove batch dim and convert to numpy
    f1 = feat1.squeeze(0).cpu().numpy()  # (C, H, W)
    f2 = feat2.squeeze(0).cpu().numpy()  # (C, H, W)
    
    C, H1, W1 = f1.shape
    _, H2, W2 = f2.shape
    
    # Flatten spatial dims and transpose to (N, C)
    f1_flat = f1.reshape(C, -1).T  # (H1*W1, C)
    f2_flat = f2.reshape(C, -1).T  # (H2*W2, C)
    
    # Concatenate to find common PCA basis
    X = np.concatenate([f1_flat, f2_flat], axis=0)  # (N, C)
    
    # Apply PCA to reduce to 3 components (for RGB visualization)
    pca = PCA(n_components=3)
    projected = pca.fit_transform(X)  # (N, 3)
    
    # Normalize to [0, 1] for RGB visualization
    p_min = projected.min(axis=0, keepdims=True)
    p_max = projected.max(axis=0, keepdims=True)
    projected = (projected - p_min) / (p_max - p_min + 1e-6)
    
    # Split back into two images
    pca1 = projected[:H1*W1, :].reshape(H1, W1, 3)
    pca2 = projected[H1*W1:, :].reshape(H2, W2, 3)
    
    return pca1, pca2


def plot_pca_features(src_img_path, trg_img_path, src_pca, trg_pca, title_suffix="", save_path=None):
    """
    Plot source and target images alongside their PCA feature visualizations.
    
    Args:
        src_img_path: Path to source image file
        trg_img_path: Path to target image file
        src_pca: Source PCA features (H, W, 3)
        trg_pca: Target PCA features (H, W, 3)
        title_suffix: Optional suffix for titles
        save_path: Path to save the plot
    """
    # Load images
    src_img = plt.imread(src_img_path)
    trg_img = plt.imread(trg_img_path)
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()
    
    axes[0].imshow(src_img)
    axes[0].set_title("Source Image")
    axes[0].axis('off')
    
    axes[1].imshow(src_pca, interpolation='nearest') # nearest to see grid, or bilinear for smooth
    axes[1].set_title(f"Source Features {title_suffix}")
    axes[1].axis('off')
    
    axes[2].imshow(trg_img)
    axes[2].set_title("Target Image")
    axes[2].axis('off')
    
    axes[3].imshow(trg_pca, interpolation='nearest')
    axes[3].set_title(f"Target Features {title_suffix}")
    axes[3].axis('off')
    
    plt.tight_layout()
    if save_path:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compare_models_pca(models, src_img_path, trg_img_path, save_path=None, model_names=None):
    """
    Compare PCA features of multiple models for a single image pair using final feature outputs.
    
    Args:
        models: List of model adapter instances.
        src_img_path: Path to source image.
        trg_img_path: Path to target image.
        save_path: Path to save result image.
        model_names: Optional list of names/titles for each model.
    """
    n_models = len(models)

    if model_names and len(model_names) != n_models:
        print(f"Warning: Number of model names ({len(model_names)}) does not match number of models ({n_models}). Ignoring custom names.")
        model_names = None
    
    # Load images for display
    src_img = plt.imread(src_img_path)
    trg_img = plt.imread(trg_img_path)
    
    # Setup plot: 2 rows (Original, Models...) x (N+1) columns
    # Row 0: Source Image, Model 0 Src, Model 1 Src...
    # Row 1: Target Image, Model 0 Trg, Model 1 Trg...
    
    fig, axes = plt.subplots(2, n_models + 1, figsize=(4 * (n_models + 1), 8))
    
    # Handle single model case where axes might be 1D or 0D? No multiple cols/rows usually returns 2D array unless squeezed.
    # But n_models >= 1, so cols >= 2. rows=2. So always 2D array.
    if n_models == 0:
        return

    # Plot original images in first column
    axes[0, 0].imshow(src_img)
    axes[0, 0].set_title("Source Image")
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(trg_img)
    axes[1, 0].set_title("Target Image")
    axes[1, 0].axis('off')
    
    for i, model in enumerate(models):
        col = i + 1
        if model_names:
            model_name = model_names[i]
        else:
            model_name = getattr(model, 'model_name', f'Model {i}')
        
        # Preprocess and extract final features
        standard_size = model.standard_size
        src_tensor = model.preprocess_image(str(src_img_path), target_size=(standard_size, standard_size))
        trg_tensor = model.preprocess_image(str(trg_img_path), target_size=(standard_size, standard_size))
        
        with torch.no_grad():
             # Extract intermediate features (output of frozen blocks)
             src_intermediate = model.extract_intermediate_features(src_tensor)
             trg_intermediate = model.extract_intermediate_features(trg_tensor)
             
             # Forward through unfrozen blocks to get final features
             feat1 = model.forward_unfrozen_blocks(src_intermediate)
             feat2 = model.forward_unfrozen_blocks(trg_intermediate)
             
             pca1, pca2 = compute_pca(feat1, feat2)
             
             # Check if center is darker than corners in the source image
             # Convert to grayscale for brightness check
             pca1_gray = pca1.mean(axis=2)
             H, W = pca1_gray.shape
             
             # Define center region (middle 50%)
             center_roi = pca1_gray[H//4 : 3*H//4, W//4 : 3*W//4]
             avg_center = center_roi.mean()
             
             # Define corner regions (outer 10%)
             h_margin = max(1, H // 10)
             w_margin = max(1, W // 10)
             
             tl = pca1_gray[:h_margin, :w_margin]
             tr = pca1_gray[:h_margin, -w_margin:]
             bl = pca1_gray[-h_margin:, :w_margin]
             br = pca1_gray[-h_margin:, -w_margin:]
             
             avg_corners = (tl.mean() + tr.mean() + bl.mean() + br.mean()) / 4.0
             
             if avg_center < avg_corners:
                 pca1 = 1.0 - pca1
                 pca2 = 1.0 - pca2
        
        axes[0, col].imshow(pca1, interpolation='nearest')
        axes[0, col].set_title(f"{model_name}\n(Source)")
        axes[0, col].axis('off')
        
        axes[1, col].imshow(pca2, interpolation='nearest')
        axes[1, col].set_title(f"{model_name}\n(Target)")
        axes[1, col].axis('off')
        
    plt.tight_layout()
    if save_path:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
    else:
        plt.show()


def visualize_heatmap(models, model_names, src_image_path, trg_image_path, save_path, point, temperature=0.02):
    """
    Visualize cosine similarity heatmaps showing correspondence between a source point and target image.
    
    Args:
        models: List of model adapters
        model_names: List of display names for each model
        src_image_path: Path to source image
        trg_image_path: Path to target image
        save_path: Path to save the visualization
        point: Tuple (x, y) of normalized coordinates [0,1] for the query point in the source image
        temperature: Temperature parameter for softmax (lower = sharper heatmaps)
    """
    n_models = len(models)
    
    # Create figure with 1 row and n_models + 2 columns
    # Column 0: Source image with query point
    # Column 1: Target image
    # Columns 2+: Heatmap overlays for each model
    fig, axes = plt.subplots(1, n_models + 2, figsize=(4 * (n_models + 2), 4))
    
    # Ensure axes is always a 1D array
    if n_models + 2 == 1:
        axes = np.array([axes])
    
    # Load images for display
    src_img_display = plt.imread(src_image_path)
    trg_img_display = plt.imread(trg_image_path)
    
    # Normalized point coordinates
    point_x_norm, point_y_norm = point
    
    # Plot source image with query point (Column 0)
    axes[0].imshow(src_img_display)
    src_h, src_w = src_img_display.shape[:2]
    point_x_pixel = point_x_norm * src_w
    point_y_pixel = point_y_norm * src_h
    axes[0].scatter([point_x_pixel], [point_y_pixel], 
                    c='red', s=10, marker='x', linewidths=3)
    axes[0].scatter([point_x_pixel], [point_y_pixel], 
                    c='yellow', s=20, marker='o', facecolors='none', linewidths=2)
    axes[0].set_title("Source\n(Query Point)")
    axes[0].axis('off')
    
    # Plot target image (Column 1)
    axes[1].imshow(trg_img_display)
    axes[1].set_title("Target")
    axes[1].axis('off')
    
    for idx, (model, model_name) in enumerate(zip(models, model_names)):
        # Column index for this model's heatmap (offset by 2 for source and target)
        col_idx = idx + 2
        
        # Get model's standard size
        standard_size = model.standard_size
        
        # Preprocess images
        src_tensor = model.preprocess_image(src_image_path, target_size=(standard_size, standard_size))
        trg_tensor = model.preprocess_image(trg_image_path, target_size=(standard_size, standard_size))
        
        # Extract features (no gradients needed)
        with torch.no_grad():
            # Extract intermediate features (output of frozen blocks)
            src_inter = model.extract_intermediate_features(src_tensor)
            trg_inter = model.extract_intermediate_features(trg_tensor)
            
            # Forward through unfrozen blocks to get final features
            src_feats = model.forward_unfrozen_blocks(src_inter)
            trg_feats = model.forward_unfrozen_blocks(trg_inter)
            
            # Get feature map dimensions
            B, C, H, W = src_feats.shape
            
            # Convert normalized point to feature map coordinates
            # Normalized coords are in [0,1], need to map to feature grid [-1,1] for grid_sample
            point_grid_x = 2.0 * point_x_norm - 1.0
            point_grid_y = 2.0 * point_y_norm - 1.0
            
            # Create grid sample coordinates: (B, 1, 1, 2)
            src_grid = torch.tensor([[[[point_grid_x, point_grid_y]]]], 
                                    dtype=torch.float32, device=src_feats.device)
            
            # Sample source descriptor at the query point
            src_desc = F.grid_sample(src_feats, src_grid, mode='bilinear', align_corners=True)
            src_desc = src_desc.squeeze()  # (C,)
            
            # Normalize features for cosine similarity
            src_vec = F.normalize(src_desc.unsqueeze(0), p=2, dim=-1)  # (1, C)
            trg_vecs = F.normalize(trg_feats.view(B, C, -1), p=2, dim=1)  # (B, C, H*W)
            
            # Compute cosine similarity
            similarity = torch.mm(src_vec, trg_vecs.squeeze(0))  # (1, H*W)
            heatmap = similarity.view(H, W)  # (H, W)
            
            # Apply temperature and softmax for probability distribution
            heatmap_temp = heatmap / temperature
            heatmap_temp = heatmap_temp - heatmap_temp.max()  # Stability
            prob_map = F.softmax(heatmap_temp.flatten(), dim=0).view(H, W)
            
            # Convert to numpy
            heatmap_np = prob_map.cpu().numpy()
        
        # Plot heatmap overlay on target
        axes[col_idx].imshow(trg_img_display)
        
        # Resize heatmap to match target image size for overlay
        trg_h, trg_w = trg_img_display.shape[:2]
        heatmap_resized = torch.nn.functional.interpolate(
            torch.from_numpy(heatmap_np).unsqueeze(0).unsqueeze(0),
            size=(trg_h, trg_w),
            mode='bilinear',
            align_corners=True
        ).squeeze().numpy()
        
        # Overlay heatmap with transparency
        im = axes[col_idx].imshow(heatmap_resized, cmap='viridis', alpha=0.6, 
                                 vmin=0, vmax=heatmap_resized.max())
        axes[col_idx].set_title(f"{model_name}\n(T={temperature})")
        axes[col_idx].axis('off')
        
        # Add colorbar for the heatmap
        plt.colorbar(im, ax=axes[col_idx], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        if os.path.dirname(save_path):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()
    