"""
DUTS Dataset Loader with Advanced Augmentation
"""

import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config


class DUTSDataset(Dataset):
    """DUTS Dataset for Salient Object Detection with Advanced Augmentation"""
    
    def __init__(self, root_dir=None, split='train', transform=None, image_dir=None, mask_dir=None):
        """
        Args:
            root_dir: Root directory containing DUTS-TR and DUTS-TE folders (optional if image_dir/mask_dir provided)
            split: 'train' or 'test' (used with root_dir)
            transform: Albumentations transform pipeline
            image_dir: Direct path to image directory (for Colab)
            mask_dir: Direct path to mask directory (for Colab)
        """
        self.split = split
        self.transform = transform
        
        # Support direct paths (for Colab) or root_dir + split (for local)
        if image_dir is not None and mask_dir is not None:
            self.img_dir = image_dir
            self.mask_dir = mask_dir
        elif root_dir is not None:
            # Auto-detect nested structure (handles both normal and double-nested DUTS)
            if split == 'train':
                split_prefix = 'DUTS-TR'
                img_suffix = 'DUTS-TR-Image'
                mask_suffix = 'DUTS-TR-Mask'
            else:
                split_prefix = 'DUTS-TE'
                img_suffix = 'DUTS-TE-Image'
                mask_suffix = 'DUTS-TE-Mask'
            
            # Try possible paths (nested or normal structure)
            possible_img_paths = [
                os.path.join(root_dir, split_prefix, split_prefix, img_suffix),  # Nested
                os.path.join(root_dir, split_prefix, img_suffix),  # Normal
            ]
            possible_mask_paths = [
                os.path.join(root_dir, split_prefix, split_prefix, mask_suffix),  # Nested
                os.path.join(root_dir, split_prefix, mask_suffix),  # Normal
            ]
            
            # Find existing paths
            self.img_dir = None
            self.mask_dir = None
            for path in possible_img_paths:
                if os.path.exists(path):
                    self.img_dir = path
                    break
            for path in possible_mask_paths:
                if os.path.exists(path):
                    self.mask_dir = path
                    break
            
            if self.img_dir is None or self.mask_dir is None:
                raise ValueError(f"Could not find DUTS {split} directories in {root_dir}")
        else:
            raise ValueError("Either provide root_dir or both image_dir and mask_dir")
        
        # Get all image files
        self.images = sorted([f for f in os.listdir(self.img_dir) 
                             if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
        
        print(f"Dataset: {len(self.images)} images from {self.img_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.img_dir, img_name)
        
        # Read image using OpenCV (faster than PIL for large datasets)
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask (ground truth)
        mask_name = os.path.splitext(img_name)[0] + '.png'
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        # Read mask as grayscale (faster single channel read)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        # Normalize mask to 0-1 (optimized with in-place division)
        mask = mask.astype(np.float32) / 255.0
        
        # Apply transforms
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Add channel dimension to mask if needed (convert numpy to tensor first if needed)
        if len(mask.shape) == 2:
            if isinstance(mask, np.ndarray):
                mask = torch.from_numpy(mask).unsqueeze(0)
            else:
                mask = mask.unsqueeze(0)
        
        return image, mask


def get_train_transform(img_size=416):  # Increased to 416 for 90%+ accuracy
    """
    Get training transforms with MAXIMUM augmentation for 90%+ F1 accuracy
    
    Args:
        img_size: Target image size (default 416 for higher detail)
    
    Returns:
        Albumentations transform pipeline
    """
    if Config.USE_AUGMENTATION:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.0625, 
                scale_limit=0.15,
                rotate_limit=15,
                p=0.4,
                border_mode=cv2.BORDER_CONSTANT
            ),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.4
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=15,
                val_shift_limit=10,
                p=0.3
            ),
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
            ], p=0.3),  # Various blur and noise types
            A.CLAHE(clip_limit=2.0, p=0.3),  # Contrast enhancement
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=1.0),
                A.GridDistortion(p=1.0),
                A.OpticalDistortion(distort_limit=0.2, shift_limit=0.2, p=1.0),
            ], p=0.25),  # Geometric distortions for better boundary learning
            A.Cutout(num_holes=8, max_h_size=32, max_w_size=32, fill_value=0, p=0.2),  # Random erasing
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
            ToTensorV2(),
        ])


def get_test_transform(img_size=416):  # Match training resolution
    """
    Get test/validation transforms (no augmentation)
    
    Args:
        img_size: Target image size (416 for 90%+ accuracy)
    
    Returns:
        Albumentations transform pipeline
    """
    return A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
        ToTensorV2(),
    ])


# Alias for validation transform (same as test)
def get_val_transform(img_size=320):
    """Get validation transforms (alias for get_test_transform)"""
    return get_test_transform(img_size)


def get_dataloaders(config=None):
    """
    Create train and test dataloaders
    
    Args:
        config: Configuration object (defaults to Config if None)
    
    Returns:
        train_loader, test_loader
    """
    if config is None:
        config = Config
    
    # Create datasets
    train_dataset = DUTSDataset(
        root_dir=config.DATA_ROOT,
        split='train',
        transform=get_train_transform(config.IMG_HEIGHT)
    )
    
    test_dataset = DUTSDataset(
        root_dir=config.DATA_ROOT,
        split='test',
        transform=get_test_transform(config.IMG_HEIGHT)
    )
    
    # Create dataloaders with memory-safe settings
    train_loader_kwargs = {
        'batch_size': config.BATCH_SIZE,
        'shuffle': True,
        'num_workers': config.NUM_WORKERS,
        'pin_memory': config.PIN_MEMORY,
        'drop_last': True
    }
    
    test_loader_kwargs = {
        'batch_size': config.EVAL_BATCH_SIZE,
        'shuffle': False,
        'num_workers': config.NUM_WORKERS,
        'pin_memory': config.PIN_MEMORY
    }
    
    # Add prefetch_factor only if num_workers > 0
    if config.NUM_WORKERS > 0:
        prefetch_factor = getattr(config, 'PREFETCH_FACTOR', 2)
        train_loader_kwargs['prefetch_factor'] = prefetch_factor
        train_loader_kwargs['persistent_workers'] = True
        test_loader_kwargs['prefetch_factor'] = prefetch_factor
        test_loader_kwargs['persistent_workers'] = True
    
    train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    test_loader = DataLoader(test_dataset, **test_loader_kwargs)
    
    return train_loader, test_loader


if __name__ == '__main__':
    # Test the dataset
    print("Testing DUTS Dataset...")
    
    train_loader, test_loader = get_dataloaders()
    
    print(f"\nTrain batches: {len(train_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Get a sample batch
    images, masks = next(iter(train_loader))
    print(f"\nBatch shapes:")
    print(f"Images: {images.shape}")
    print(f"Masks: {masks.shape}")
