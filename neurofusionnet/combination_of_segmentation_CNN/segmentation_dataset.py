"""
BRISC 2025 Segmentation Dataset — paired image + mask loading.

Handles:
  - Loading from brisc2025/segmentation_task/train/ and test/
  - Joint augmentation (same transform applied to both image and mask)
  - Train/validation split
  - DataLoader factory with proper workers and pinning
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import functional as TF
from PIL import Image
import random


# ── Joint Augmentation ───────────────────────────────────────────────────────

class JointTransform:
    """Apply identical random transforms to both image and mask.
    
    Ensures spatial transforms (flip, rotation, crop) are synchronized
    between the MRI image and its segmentation mask.
    """

    def __init__(self, img_size=224, is_train=True):
        self.img_size = img_size
        self.is_train = is_train
        # Image-only normalization
        self.normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

    def __call__(self, image, mask):
        # Resize both to target + padding for crop
        if self.is_train:
            resize_to = self.img_size + 32
        else:
            resize_to = self.img_size

        image = TF.resize(image, [resize_to, resize_to])
        mask = TF.resize(mask, [resize_to, resize_to], interpolation=transforms.InterpolationMode.NEAREST)

        if self.is_train:
            # Random crop (same params for both)
            i, j, h, w = transforms.RandomCrop.get_params(
                image, output_size=(self.img_size, self.img_size)
            )
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)

            # Random horizontal flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Random vertical flip
            if random.random() > 0.8:
                image = TF.vflip(image)
                mask = TF.vflip(mask)

            # Random rotation (-15° to +15°)
            if random.random() > 0.5:
                angle = random.uniform(-15, 15)
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

            # Color jitter (image only — don't distort mask)
            if random.random() > 0.5:
                image = transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.1
                )(image)

        # Convert to tensors
        image = TF.to_tensor(image)  # [0, 1]
        mask = TF.to_tensor(mask)    # [0, 1]

        # Binarize mask (threshold at 0.5)
        mask = (mask > 0.5).float()

        # Take only first channel of mask (make it single-channel)
        if mask.shape[0] > 1:
            mask = mask[0:1]

        # Normalize image (ImageNet stats)
        image = self.normalize(image)

        return image, mask


# ── Dataset ──────────────────────────────────────────────────────────────────

class BRISCSegDataset(Dataset):
    """BRISC 2025 Segmentation Dataset.
    
    Loads paired (image, mask) from:
      images/: brisc2025_train_XXXXX_XX_XX_t1.jpg
      masks/:  brisc2025_train_XXXXX_XX_XX_t1.png
    
    Args:
        root_dir: Path to segmentation_task directory
        split: 'train' or 'test'
        transform: JointTransform instance
    """

    def __init__(self, root_dir: str, split: str = "train",
                 transform: Optional[JointTransform] = None):
        self.root_dir = Path(root_dir) / split
        self.images_dir = self.root_dir / "images"
        self.masks_dir = self.root_dir / "masks"
        self.transform = transform

        # Build paired file list
        self.samples = []
        if self.images_dir.exists():
            for img_path in sorted(self.images_dir.glob("*.jpg")):
                # Corresponding mask has same name but .png extension
                mask_name = img_path.stem + ".png"
                mask_path = self.masks_dir / mask_name
                if mask_path.exists():
                    self.samples.append((str(img_path), str(mask_path)))

        if len(self.samples) == 0:
            print(f"⚠️  No image-mask pairs found in {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img_path, mask_path = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Grayscale

        if self.transform:
            image, mask = self.transform(image, mask)
        else:
            image = TF.to_tensor(image)
            mask = TF.to_tensor(mask)
            mask = (mask > 0.5).float()

        return image, mask

    def get_tumor_type(self, idx: int) -> str:
        """Extract tumor type from filename."""
        img_path = self.samples[idx][0]
        fname = Path(img_path).stem
        parts = fname.split("_")
        if len(parts) >= 4:
            code = parts[3]
            mapping = {"gl": "Glioma", "me": "Meningioma", "pi": "Pituitary", "nt": "No Tumor"}
            return mapping.get(code, code)
        return "Unknown"


# ── DataLoader Factory ───────────────────────────────────────────────────────

def get_seg_dataloaders(
    data_dir: str,
    batch_size: int = 16,
    val_split: float = 0.2,
    img_size: int = 224,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, validation, and test DataLoaders for segmentation.
    
    Args:
        data_dir: Path to segmentation_task directory
        batch_size: Batch size
        val_split: Fraction for validation split
        img_size: Target image size
        num_workers: Data loading workers
    
    Returns:
        (train_loader, val_loader, test_loader)
    """
    # Training set with augmentation
    train_full = BRISCSegDataset(
        data_dir, split="train",
        transform=JointTransform(img_size=img_size, is_train=True),
    )

    # Split into train/val
    total = len(train_full)
    val_size = int(total * val_split)
    train_size = total - val_size

    train_dataset, val_dataset_raw = random_split(
        train_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    # Test set (no augmentation)
    test_dataset = BRISCSegDataset(
        data_dir, split="test",
        transform=JointTransform(img_size=img_size, is_train=False),
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset_raw, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "brisc2025", "segmentation_task"
    )
    train_dl, val_dl, test_dl = get_seg_dataloaders(data_dir, batch_size=4, num_workers=0)
    print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}, Test: {len(test_dl)}")
    imgs, masks = next(iter(train_dl))
    print(f"Images: {imgs.shape}, Masks: {masks.shape}")
    print(f"Mask unique values: {masks.unique().tolist()}")
