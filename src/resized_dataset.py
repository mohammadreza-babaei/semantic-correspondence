# Here we define all the data loaders used in the project
import os
import json
from torchvision import transforms
from torchvision.transforms import functional as TF
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class ResizedSPairDataset(Dataset):
    """
    Wrapper around an existing SPair dataset that applies resizing and scaling transformations.
    Takes an existing dataset (from HuggingFace or custom loader) and resizes images/metadata
    according to the model config.
    """
    def __init__(self, base_dataset, config=None):
        """
        Args:
            base_dataset: An existing SPair dataset to wrap and resize
            config (ModelConfig): A configuration object containing size, norm, etc.
        """
        self.base_dataset = base_dataset
        self.config = config

        # Build Transform based on Config
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.config.mean, std=self.config.std)
        ])

    def resize_and_scale(self, image, kps, bbox):
        """Resize with aspect-ratio preservation and edge-padding to square."""
        if isinstance(image, Image.Image):
            w, h = image.size
        else:
            # Handle tensor images
            h, w = image.shape[-2:]
            image = transforms.ToPILImage()(image)

        target_size = self.config.size

        # Fit the longest side to target_size, keep aspect ratio
        scale = target_size / max(w, h)
        new_w = max(1, int(round(w * scale)))
        new_h = max(1, int(round(h * scale)))

        img_resized = image.resize((new_w, new_h), self.config.interpolation)

        # Symmetric padding to reach target_size using edge replication
        pad_w = max(0, target_size - new_w)
        pad_h = max(0, target_size - new_h)
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top

        # Apply edge padding without altering the resized content
        pad_params = (pad_left, pad_top, pad_right, pad_bottom)
        if any(pad_params):
            img_final = TF.pad(img_resized, pad_params, padding_mode='edge')
        else:
            img_final = img_resized

        # Scale and shift metadata to match resize + padding
        scale_x = scale
        scale_y = scale

        if len(kps) > 0:
            kps_tensor = kps.float().clone() if isinstance(kps, torch.Tensor) else torch.tensor(kps).float()
            kps_tensor[:, 0] = kps_tensor[:, 0] * scale_x + pad_left
            kps_tensor[:, 1] = kps_tensor[:, 1] * scale_y + pad_top
        else:
            kps_tensor = torch.tensor([])

        if bbox is not None and len(bbox) > 0:
            bbox_scaled = bbox.float().clone() if isinstance(bbox, torch.Tensor) else torch.tensor(bbox).float()
            bbox_scaled[0] = bbox_scaled[0] * scale_x + pad_left
            bbox_scaled[1] = bbox_scaled[1] * scale_y + pad_top
            bbox_scaled[2] = bbox_scaled[2] * scale_x + pad_left
            bbox_scaled[3] = bbox_scaled[3] * scale_y + pad_top
        else:
            bbox_scaled = torch.zeros(4)

        return img_final, kps_tensor, bbox_scaled

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        # Get raw data from base dataset
        sample = self.base_dataset[idx]
        
        # Extract source and target images and metadata
        src_img_raw = sample['src_img'] if isinstance(sample['src_img'], Image.Image) else Image.fromarray(sample['src_img'])
        trg_img_raw = sample['trg_img'] if isinstance(sample['trg_img'], Image.Image) else Image.fromarray(sample['trg_img'])
        
        src_kps = sample['src_kps'] if 'src_kps' in sample else []
        trg_kps = sample['trg_kps'] if 'trg_kps' in sample else []
        src_bbox = sample['src_bndbox'] if 'src_bndbox' in sample else None
        trg_bbox = sample['trg_bndbox'] if 'trg_bndbox' in sample else None
        category = sample.get('category', 'unknown')
        
        # Preprocess (Resize & Scale)
        src_img, src_kps_scaled, src_bbox_scaled = self.resize_and_scale(src_img_raw, src_kps, src_bbox)
        trg_img, trg_kps_scaled, trg_bbox_scaled = self.resize_and_scale(trg_img_raw, trg_kps, trg_bbox)
        
        # Prepare Output Dict
        output = {
            'src_img': self.transform(src_img),
            'trg_img': self.transform(trg_img),
            'src_kps': src_kps_scaled,
            'trg_kps': trg_kps_scaled,
            'src_bbox': src_bbox_scaled,
            'trg_bbox': trg_bbox_scaled,
            'category': category,
            'original_size': torch.tensor(src_img_raw.size) # Keep track of original size just in case
        }
        
        # Condition Prompt (Only for models that need it)
        if self.config.needs_prompt:
            output['prompt'] = f"a photo of a {category}"
            
        return output
