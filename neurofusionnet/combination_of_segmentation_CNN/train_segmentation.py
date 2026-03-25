"""
Training script for Lightweight U-Net segmentation on BRISC 2025.

Features:
  - Combined Dice + BCE loss
  - CosineAnnealingLR scheduler
  - Mixed precision training (AMP)
  - Early stopping (patience=7)
  - Saves best model checkpoint
  - Logs training history as JSON
"""

import os
import sys
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from neurofusionnet.combination_of_segmentation_CNN.segmentation_model import (
    create_unet, CombinedSegLoss, dice_score, iou_score,
)
from neurofusionnet.combination_of_segmentation_CNN.segmentation_dataset import (
    get_seg_dataloaders,
)


def train_segmentation(
    data_dir: str = None,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-4,
    patience: int = 7,
    save_dir: str = None,
    num_workers: int = 4,
    img_size: int = 224,
):
    """Train the U-Net segmentation model.
    
    Args:
        data_dir: Path to brisc2025/segmentation_task/
        epochs: Maximum training epochs
        batch_size: Training batch size
        lr: Initial learning rate
        patience: Early stopping patience
        save_dir: Directory for checkpoints
        num_workers: DataLoader workers
        img_size: Input image size
    """
    # Defaults
    if data_dir is None:
        data_dir = str(PROJECT_ROOT / "brisc2025" / "segmentation_task")
    if save_dir is None:
        save_dir = str(PROJECT_ROOT / "checkpoints")
    os.makedirs(save_dir, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Device: {device}")

    # Data
    print(f"📂 Loading data from: {data_dir}")
    train_loader, val_loader, test_loader = get_seg_dataloaders(
        data_dir, batch_size=batch_size, img_size=img_size, num_workers=num_workers,
    )
    print(f"   Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # Model
    model = create_unet(pretrained=True, freeze_encoder=0.5).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"🧠 U-Net: {total_params / 1e6:.2f}M params ({trainable / 1e6:.2f}M trainable)")

    # Loss, optimizer, scheduler
    criterion = CombinedSegLoss(dice_weight=0.5, bce_weight=0.5)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr, weight_decay=1e-5,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    # Mixed precision
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None
    use_amp = device.type == 'cuda'

    # Training state
    best_dice = 0.0
    no_improve = 0
    history = {
        "train_loss": [], "val_loss": [],
        "train_dice": [], "val_dice": [],
        "train_iou": [], "val_iou": [],
        "lr": [],
    }

    print(f"\n{'='*60}")
    print(f"🚀 Starting Training — {epochs} epochs")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        start = time.time()

        # ── Train ──
        model.train()
        train_loss, train_dice_sum, train_iou_sum = 0, 0, 0
        for batch_idx, (images, masks) in enumerate(train_loader):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            if use_amp:
                with torch.amp.autocast('cuda'):
                    preds = model(images)
                    loss = criterion(preds, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(images)
                loss = criterion(preds, masks)
                loss.backward()
                optimizer.step()

            train_loss += loss.item()
            with torch.no_grad():
                train_dice_sum += dice_score(preds, masks).item()
                train_iou_sum += iou_score(preds, masks).item()

        n_train = len(train_loader)
        avg_train_loss = train_loss / n_train
        avg_train_dice = train_dice_sum / n_train
        avg_train_iou = train_iou_sum / n_train

        # ── Validate ──
        model.eval()
        val_loss, val_dice_sum, val_iou_sum = 0, 0, 0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                preds = model(images)
                loss = criterion(preds, masks)
                val_loss += loss.item()
                val_dice_sum += dice_score(preds, masks).item()
                val_iou_sum += iou_score(preds, masks).item()

        n_val = max(len(val_loader), 1)
        avg_val_loss = val_loss / n_val
        avg_val_dice = val_dice_sum / n_val
        avg_val_iou = val_iou_sum / n_val

        # Update scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Log
        elapsed = time.time() - start
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["train_dice"].append(avg_train_dice)
        history["val_dice"].append(avg_val_dice)
        history["train_iou"].append(avg_train_iou)
        history["val_iou"].append(avg_val_iou)
        history["lr"].append(current_lr)

        # Print progress
        improved = ""
        if avg_val_dice > best_dice:
            best_dice = avg_val_dice
            no_improve = 0
            improved = " ★ BEST"
            # Save best model
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "dice_score": best_dice,
            }, os.path.join(save_dir, "unet_segmentation.pth"))
        else:
            no_improve += 1

        print(f"Epoch {epoch:3d}/{epochs} │ "
              f"Loss: {avg_train_loss:.4f}/{avg_val_loss:.4f} │ "
              f"Dice: {avg_train_dice:.4f}/{avg_val_dice:.4f} │ "
              f"IoU: {avg_train_iou:.4f}/{avg_val_iou:.4f} │ "
              f"LR: {current_lr:.2e} │ "
              f"{elapsed:.1f}s{improved}")

        # Early stopping
        if no_improve >= patience:
            print(f"\n⏹️  Early stopping at epoch {epoch} (no improvement for {patience} epochs)")
            break

    # Save training history
    with open(os.path.join(save_dir, "seg_training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*60}")
    print(f"✅ Training complete — Best Val Dice: {best_dice:.4f}")
    print(f"   Model saved to: {os.path.join(save_dir, 'unet_segmentation.pth')}")
    print(f"   History saved to: {os.path.join(save_dir, 'seg_training_history.json')}")
    print(f"{'='*60}")

    return model, history


if __name__ == "__main__":
    train_segmentation(epochs=30, batch_size=16, num_workers=2)
