"""SPair-71k: A Large-scale Benchmark for Semantic Correspondence

Pure PyTorch Dataset implementation without HuggingFace dependencies.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, List
from PIL import Image
import torch
from torch.utils.data import Dataset


_CITATION = """\
@article{min2019spair,
   title={SPair-71k: A Large-scale Benchmark for Semantic Correspondence},
   author={Juhong Min and Jongmin Lee and Jean Ponce and Minsu Cho},
   journal={arXiv prepreint arXiv:1908.10543},
   year={2019}
}
"""

_DESCRIPTION = """\
Establishing visual correspondences under large intra-class variations, which is often referred to as 
semantic correspondence or semantic matching, remains a challenging problem in computer vision. 
SPair-71k contains 70,958 image pairs with diverse variations in viewpoint and scale.
"""

_HOMEPAGE = "https://cvlab.postech.ac.kr/research/SPair-71k/"

CATEGORIES = ['cat', 'pottedplant', 'train', 'bicycle', 'car', 'bus', 'aeroplane', 
              'dog', 'bird', 'chair', 'motorbike', 'cow', 'bottle', 'person', 
              'boat', 'sheep', 'horse', 'tvmonitor']

POSES = ['Unspecified', 'Frontal', 'Left', 'Rear', 'Right']


class SPair71kPairs(Dataset):
    """SPair-71k dataset for image pairs with correspondence annotations.
    
    Args:
        root: Path to the SPair-71k dataset root directory (containing SPair-71k folder)
        split: One of 'train', 'val', or 'test'
        transform: Optional transform to be applied on images
    """
    
    def __init__(
        self, 
        root: str, 
        split: str = 'train',
        transform: Optional[Any] = None
    ):
        self.root = Path(root) / 'SPair-71k'
        self.split = split
        self.transform = transform
        
        # Map split names
        split_map = {'train': 'trn', 'val': 'val', 'test': 'test'}
        if split not in split_map:
            raise ValueError(f"Split must be one of {list(split_map.keys())}, got {split}")
        self.split_dir = split_map[split]
        
        # Load all pair annotations
        self.pairs = {}
        pair_dir = self.root / 'PairAnnotation' / self.split_dir
        if not pair_dir.exists():
            raise FileNotFoundError(f"Pair annotation directory not found: {pair_dir}")
        
        for pair_file in sorted(pair_dir.glob('*.json')):
            with open(pair_file, 'r') as f:
                data = json.load(f)
                self.pairs[data['pair_id']] = data
        
        # Store sorted pair_ids for efficient indexing
        self.pair_ids = sorted(self.pairs.keys())
    
    def get_image_path(self, img_name: str) -> Path:
        """Get the absolute path to an image by its name."""
        return self.root / 'JPEGImages' / f'{img_name}.jpg'

    def get_segmentation_path(self, img_name: str) -> Path:
        """Get the absolute path to a segmentation mask by its name."""
        return self.root / 'Segmentation' / f'{img_name}.png'

    def get_images_dataset(self, transform: Optional[Any] = None) -> 'SPair71kImages':
        """Factory method to get an SPair71kImages dataset sharing the same root.
        
        Args:
            transform: Optional transform to be applied on images
            
        Returns:
            SPair71kImages instance
        """
        # self.root is Path(root) / 'SPair-71k', we need the parent for the __init__ call
        return SPair71kImages(str(self.root.parent), transform=transform)

    def __len__(self) -> int:
        return len(self.pairs.keys())
    
    def get_pair_by_id(self, pair_id: str) -> Dict[str, Any]:
        """Get a pair by its pair_id.
        
        Args:
            pair_id: The pair_id to retrieve (e.g., 'trn:aeroplane-0001:aeroplane-0010')
        
        Returns:
            Dictionary containing all pair data (same format as __getitem__)
            
        Raises:
            KeyError: If pair_id is not found in the dataset
        """
        if pair_id not in self.pairs:
            raise KeyError(f"Pair ID '{pair_id}' not found in {self.split} split. "
                          f"Available pair IDs: {len(self.pairs)} pairs")
        
        data = self.pairs[pair_id]
        category = data['category']
        
        # Load source image and segmentation
        src_name = f"{category}/{data['src_imname'][:-4]}"
        src_img_path = self.get_image_path(src_name)
        src_seg_path = self.get_segmentation_path(src_name)
        
        src_img = Image.open(src_img_path).convert('RGB')
        src_seg = Image.open(src_seg_path)
        
        # Load target image and segmentation
        trg_name = f"{category}/{data['trg_imname'][:-4]}"
        trg_img_path = self.get_image_path(trg_name)
        trg_seg_path = self.get_segmentation_path(trg_name)
        
        trg_img = Image.open(trg_img_path).convert('RGB')
        trg_seg = Image.open(trg_seg_path)
        
        # Apply transforms if provided
        if self.transform:
            src_img = self.transform(src_img)
            trg_img = self.transform(trg_img)
            src_seg = self.transform(src_seg)
            trg_seg = self.transform(trg_seg)
        
        # Convert keypoints and boxes to torch tensors (ensure types)
        src_kps = torch.tensor(data['src_kps'], dtype=torch.float32)
        trg_kps = torch.tensor(data['trg_kps'], dtype=torch.float32)
        # Keep keypoint ids as-is (can be strings); downstream code treats them as labels
        kps_ids = list(data['kps_ids'])
        
        return {
            'pair_id': data['pair_id'],
            'src_img': src_img,
            'src_segmentation': src_seg,
            'src_name': src_name,
            'src_imsize': torch.tensor(data['src_imsize'], dtype=torch.int64),
            'src_bndbox': torch.tensor(data['src_bndbox'], dtype=torch.int64),
            'src_pose': data['src_pose'],
            'src_kps': src_kps,
            'trg_img': trg_img,
            'trg_segmentation': trg_seg,
            'trg_name': trg_name,
            'trg_imsize': torch.tensor(data['trg_imsize'], dtype=torch.int64),
            'trg_bndbox': torch.tensor(data['trg_bndbox'], dtype=torch.int64),
            'trg_pose': data['trg_pose'],
            'trg_kps': trg_kps,
            'kps_ids': kps_ids,
            'category': category,
            'category_id': CATEGORIES.index(category),
            'viewpoint_variation': data['viewpoint_variation'],
            'scale_variation': data['scale_variation'],
            'truncation': data['truncation'],
            'occlusion': data['occlusion'],
        }
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        pair_id = self.pair_ids[idx]
        return self.get_pair_by_id(pair_id)


class SPair71kImages(Dataset):
    """SPair-71k dataset for individual images with annotations.
    
    Args:
        root: Path to the SPair-71k dataset root directory (containing SPair-71k folder)
        transform: Optional transform to be applied on images
    """
    
    def __init__(
        self, 
        root: str,
        transform: Optional[Any] = None
    ):
        self.root = Path(root) / 'SPair-71k'
        self.transform = transform
        
        # Build image index by listing all images
        self.images = self.list_all_images()
    
    def get_image_path(self, img_name: str) -> Path:
        """Get the absolute path to an image by its name."""
        return self.root / 'JPEGImages' / f'{img_name}.jpg'

    def get_segmentation_path(self, img_name: str) -> Path:
        """Get the absolute path to a segmentation mask by its name."""
        return self.root / 'Segmentation' / f'{img_name}.png'

    def get_annotation_path(self, img_name: str) -> Path:
        """Get the absolute path to an image's annotation by its name."""
        return self.root / 'ImageAnnotation' / f'{img_name}.json'

    def list_all_images(self) -> List[str]:
        """List all images in the SPair-71k dataset across all splits.
        
        Returns:
            Sorted list of image names in format 'category/image_stem'
        """
        jpeg_dir = self.root / 'JPEGImages'
        if not jpeg_dir.exists():
            raise FileNotFoundError(f"JPEGImages directory not found: {jpeg_dir}")
        
        images = []
        for category_dir in sorted(jpeg_dir.glob('*')):
            if not category_dir.is_dir():
                continue
            category = category_dir.name
            for img_file in sorted(category_dir.glob('*.jpg')):
                img_name = f"{category}/{img_file.stem}"
                images.append(img_name)
        
        return images



    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        img_name = self.images[idx]
        
        # Load image and segmentation
        img_path = self.get_image_path(img_name)
        seg_path = self.get_segmentation_path(img_name)
        annot_path = self.get_annotation_path(img_name)
        
        img = Image.open(img_path).convert('RGB')
        seg = Image.open(seg_path)
        
        # Load annotation
        with open(annot_path, 'r') as f:
            annotation = json.load(f)
        
        # Apply transforms if provided
        if self.transform:
            img = self.transform(img)
            seg = self.transform(seg)
        
        # Convert keypoints
        kps = []
        for kp in annotation['kps'].values():
            if kp:
                kps.append(kp)
            else:
                kps.append([-1, -1])
        kps = torch.tensor(kps, dtype=torch.float32)
        
        return {
            'img': img,
            'name': img_name,
            'segmentation': seg,
            'filename': annotation['filename'],
            'src_database': annotation['src_database'],
            'src_annotation': annotation['src_annotation'],
            'src_image': annotation['src_image'],
            'image_width': annotation['image_width'],
            'image_height': annotation['image_height'],
            'image_depth': annotation['image_depth'],
            'category': annotation['category'],
            'category_id': CATEGORIES.index(annotation['category']),
            'pose': annotation['pose'],
            'truncated': annotation['truncated'],
            'occluded': annotation['occluded'],
            'difficult': annotation['difficult'],
            'bndbox': torch.tensor(annotation['bndbox'], dtype=torch.int64),
            'kps': kps,
            'azimuth_id': annotation['azimuth_id'],
        }

