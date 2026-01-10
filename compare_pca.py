import torch
import matplotlib.pyplot as plt
from src.plotting import compare_models_pca
from src.models import DINOv2Adapter, SAMAdapter, DINOv3Adapter

# Initialize models
device = 'cuda' if torch.cuda.is_available() else 'cpu'

dinov2 = DINOv2Adapter(model_name='dinov2_vits14', device=device)
dinov2_finetuned_augmented = DINOv2Adapter(model_name='dinov2_vits14', weights_path='/home/lollo/Documents/models/finetuned_dinov2_augmented.pt', device=device)

dinov3 = DINOv3Adapter(model_name='dinov3_vits16', weights_path='/home/lollo/Downloads/dinov3_vits16_pretrain_lvd1689m-08c60483.pth', device=device)
dinov3_finetuned_augmented = DINOv3Adapter(model_name='dinov3_vits16', weights_path='/home/lollo/Documents/models/finetuned_dinov3_augmented_temp_0.02.pt', device=device)

sam = SAMAdapter(model_name='vit_b', weights_path='/home/lollo/Downloads/sam_vit_b_01ec64.pth', device=device)
sam_finetuned_augmented = SAMAdapter(model_name='vit_b', weights_path='/home/lollo/Documents/models/finetuned_sam_vit_b_augmented_temp_0.02.pt', device=device)
# Define image paths (you can pick any pair of images)
src_img = 'data/SPair-71k/JPEGImages/cat/2010_004954.jpg'
trg_img = 'data/SPair-71k/JPEGImages/cat/2008_006999.jpg'

# Compare final feature outputs from each model
compare_models_pca(
    models=[dinov2, dinov2_finetuned_augmented, dinov3, dinov3_finetuned_augmented, sam, sam_finetuned_augmented],
    model_names=['DINOv2', 'DINOv2 (finetuned)', 'DINOv3', 'DINOv3 (finetuned)', 'SAM', 'SAM (finetuned)'],
    src_img_path=src_img,
    trg_img_path=trg_img,
    save_path='comparison.png' # Optional, if not provided it shows the plot with plt.show()
)