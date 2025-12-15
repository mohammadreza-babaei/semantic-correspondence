from torchvision import transforms

class ModelConfig:
    def __init__(self, name, size, norm_mean, norm_std, needs_prompt=False, interpolation=3):
        self.name = name
        self.size = size
        self.mean = norm_mean
        self.std = norm_std
        self.needs_prompt = needs_prompt
        self.interpolation = interpolation # 3 is BICUBIC, 2 is BILINEAR

MODEL_CONFIGS = {
    # Paper 2 & 3: DINOv2 requires 840x840 (multiple of 14) and ImageNet Norm
    'dinov2': ModelConfig(
        name='dinov2',
        size=840,
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
        needs_prompt=False
    ),

    # ADD THIS FOR DINOv3
    'dinov3': ModelConfig(
        name='dinov3',
        size=848, # 848 / 16 = 53 patches. (Closest valid size to 840)
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
        needs_prompt=False
    ),

    'sam': ModelConfig(
    name='sam',
    size=1024,  # Standard for SAM
    norm_mean=[0.485, 0.456, 0.406],
    norm_std=[0.229, 0.224, 0.225],
    needs_prompt=False
    ),
    
    # Paper 2 & 3 (Telling Left from Right): Uses 960x960 for SD features
    'sd_paper3': ModelConfig(
        name='sd_paper3',
        size=960,
        norm_mean=[0.5, 0.5, 0.5],
        norm_std=[0.5, 0.5, 0.5],
        needs_prompt=True
    ),
    
    # Future proofing for SAM (Example)
    'sam': ModelConfig(
        name='sam',
        size=1024, # SAM standard
        norm_mean=[0.485, 0.456, 0.406],
        norm_std=[0.229, 0.224, 0.225],
        needs_prompt=False
    )
}