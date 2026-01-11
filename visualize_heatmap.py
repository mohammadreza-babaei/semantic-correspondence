import torch
from src.plotting import visualize_heatmap
from src.models import DINOv2Adapter, SAMAdapter, DINOv3Adapter

# Setup device
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize models

dinov2 = DINOv2Adapter(model_name='dinov2_vits14', device=device)
dinov2_finetuned_augmented = DINOv2Adapter(model_name='dinov2_vits14', weights_path='/home/lollo/Documents/models/finetuned_dinov2_augmented.pt', device=device)

dinov3 = DINOv3Adapter(model_name='dinov3_vits16', weights_path='/home/lollo/Downloads/dinov3_vits16_pretrain_lvd1689m-08c60483.pth', device=device)
dinov3_finetuned_augmented = DINOv3Adapter(model_name='dinov3_vits16', weights_path='/home/lollo/Documents/models/finetuned_dinov3_augmented_temp_0.02.pt', device=device)

sam = SAMAdapter(model_name='vit_b', weights_path='/home/lollo/Downloads/sam_vit_b_01ec64.pth', device=device)
sam_finetuned_augmented = SAMAdapter(model_name='vit_b', weights_path='/home/lollo/Documents/models/finetuned_sam_vit_b_augmented_temp_0.02.pt', device=device)

# Define image paths and query point
src_img = 'data/SPair-71k/JPEGImages/cat/2010_004954.jpg'
trg_img = 'data/SPair-71k/JPEGImages/cat/2008_006999.jpg'

# Visualize left paw
visualize_heatmap(
    models=[dinov2, dinov2_finetuned_augmented, dinov3, dinov3_finetuned_augmented, sam, sam_finetuned_augmented],
    model_names=['DINOv2 (baseline)', 'DINOv2 (finetuned)', 'DINOv3 (baseline)', 'DINOv3 (finetuned)', 'SAM (baseline)', 'SAM (finetuned)'],
    src_image_path=src_img,
    trg_image_path=trg_img,
    save_path='evaluations/heatmaps/left_paw_comparison.png',
    point=(0.57, 0.8),
    temperature=0.02
)

# Visualize right paw
visualize_heatmap(
    models=[dinov2, dinov2_finetuned_augmented, dinov3, dinov3_finetuned_augmented, sam, sam_finetuned_augmented],
    model_names=['DINOv2 (baseline)', 'DINOv2 (finetuned)', 'DINOv3 (baseline)', 'DINOv3 (finetuned)', 'SAM (baseline)', 'SAM (finetuned)'],
    src_image_path=src_img,
    trg_image_path=trg_img,
    save_path='evaluations/heatmaps/right_paw_comparison.png',
    point=(0.63, 0.78),
    temperature=0.02
)
