import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_history(history, save_path=None):
    """
    Plot training history after training is complete.
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'epoch_train_losses'
        save_path: Optional path to save the figure
    """
    train_df = pd.read_csv("checkpoints/finetuned_dinov2/training_log_e5_b1.csv")
    val_df   = pd.read_csv("metrics/val_metrics.csv")
    print(train_df.head())
    print(val_df.head())


    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Epoch losses
    epochs = range(1, len(history['train_loss']) + 1)
    axes[0].plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2)
    if history.get('val_loss'):
        axes[0].plot(epochs, history['val_loss'], 'r-s', label='Val Loss', linewidth=2)
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
    # axes[2].plot(val_df["epoch"], val_df["pck"], "g-o", label="PCK")
    axes[2].plot(val_df["epoch"], val_df["val_pck"], "g-o", label="PCK")

    axes[2].set_title("PCK over Epochs")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("PCK")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend()
    

    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
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
    