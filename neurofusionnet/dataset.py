"""
Dataset loading and augmentation for BRISC 2025 Brain Tumor MRI dataset.

Provides:
  - BrainTumorDataset: Custom PyTorch dataset
  - get_dataloaders: Factory function for train/val/test dataloaders
  - Augmentation pipelines for training and evaluation
"""

import os
from pathlib import Path
from typing import Tuple, Optional

import torch
from torch.utils.data import Dataset, DataLoader, random_split, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
import numpy as np

from .utils import CLASS_NAMES, IMG_SIZE


# ── Augmentation Pipelines ────────────────────────────────────────────────────

def get_train_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Training augmentation pipeline — diverse transforms for robustness."""
    return transforms.Compose([
        transforms.Resize((img_size + 32, img_size + 32)),
        transforms.RandomCrop(img_size),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.RandomGrayscale(p=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.15)),
    ])


def get_eval_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Evaluation transforms — deterministic resize and normalize."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


def get_raw_transforms(img_size: int = IMG_SIZE) -> transforms.Compose:
    """Raw transforms for visualization — resize only, no normalization."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class BrainTumorDataset(Dataset):
    """BRISC 2025 Brain Tumor MRI Dataset.
    
    Args:
        root_dir: Path to classification_task directory
        split: 'train' or 'test'
        transform: Optional transform to apply to images
    """

    def __init__(self, root_dir: str, split: str = "train",
                 transform: Optional[transforms.Compose] = None):
        self.root_dir = Path(root_dir) / split
        self.transform = transform
        self.classes = sorted([d.name for d in self.root_dir.iterdir() if d.is_dir()])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Build file list
        self.samples = []
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            for img_path in sorted(cls_dir.glob("*.jpg")):
                self.samples.append((str(img_path), self.class_to_idx[cls_name]))
            for img_path in sorted(cls_dir.glob("*.png")):
                self.samples.append((str(img_path), self.class_to_idx[cls_name]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label

    def get_class_distribution(self) -> dict:
        """Get count of samples per class."""
        dist = {cls: 0 for cls in self.classes}
        for _, label in self.samples:
            cls_name = self.classes[label]
            dist[cls_name] += 1
        return dist


# ── DataLoader Factory ────────────────────────────────────────────────────────

def get_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    val_split: float = 0.15,
    num_workers: int = 4,
    img_size: int = IMG_SIZE,
    use_weighted_sampling: bool = True,
) -> Tuple[DataLoader, DataLoader, DataLoader, list]:
    """Create train, validation, and test DataLoaders.
    
    Args:
        data_dir: Path to classification_task directory
        batch_size: Batch size
        val_split: Fraction of training data for validation
        num_workers: Number of data loading workers
        img_size: Image size for transforms
        use_weighted_sampling: Use weighted random sampling for class balance
    
    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    # Create datasets
    train_full = BrainTumorDataset(data_dir, split="train",
                                   transform=get_train_transforms(img_size))
    test_dataset = BrainTumorDataset(data_dir, split="test",
                                     transform=get_eval_transforms(img_size))

    # Split training into train + validation
    total = len(train_full)
    val_size = int(total * val_split)
    train_size = total - val_size

    train_dataset, val_dataset_raw = random_split(
        train_full, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )

    # Create a validation dataset with eval transforms
    # (We wrap the raw split to override transforms)
    val_dataset = ValidationSubset(train_full, val_dataset_raw.indices,
                                   get_eval_transforms(img_size))

    # Weighted sampling for training
    sampler = None
    shuffle = True
    if use_weighted_sampling:
        train_labels = [train_full.samples[i][1] for i in train_dataset.indices]
        class_counts = np.bincount(train_labels, minlength=len(train_full.classes))
        class_weights = 1.0 / (class_counts + 1e-6)
        sample_weights = [class_weights[label] for label in train_labels]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        shuffle = False

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=shuffle,
        sampler=sampler, num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader, test_loader, train_full.classes


class ValidationSubset(Dataset):
    """Subset wrapper that applies separate transforms (eval transforms for val)."""

    def __init__(self, full_dataset: BrainTumorDataset, indices: list,
                 transform: transforms.Compose):
        self.full_dataset = full_dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        img_path, label = self.full_dataset.samples[real_idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


if __name__ == "__main__":
    # Quick test
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "brisc2025", "classification_task")
    train_dl, val_dl, test_dl, classes = get_dataloaders(data_dir, batch_size=4, num_workers=0)
    print(f"Classes: {classes}")
    print(f"Train batches: {len(train_dl)}, Val batches: {len(val_dl)}, Test batches: {len(test_dl)}")
    batch = next(iter(train_dl))
    print(f"Batch image shape: {batch[0].shape}, Labels shape: {batch[1].shape}")
