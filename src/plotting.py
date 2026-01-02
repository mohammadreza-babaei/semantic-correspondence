import matplotlib.pyplot as plt
import numpy as np

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

    