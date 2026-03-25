"""
Training pipeline for NeuroFusionNet.

Features:
  - AdamW optimizer with cosine annealing + warmup
  - Label smoothing cross-entropy loss
  - Mixed precision training (AMP)
  - Early stopping with patience
  - Model checkpointing
  - Progress bars with tqdm
"""

import os
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import numpy as np

from .model import NeuroFusionNet, create_model
from .dataset import get_dataloaders
from .utils import (
    set_seed, get_device, count_parameters,
    compute_metrics, save_checkpoint, CLASS_NAMES,
)


# ── Learning Rate Scheduler with Warmup ───────────────────────────────────────

class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """Cosine annealing with linear warmup."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            factor = self.last_epoch / max(1, self.warmup_epochs)
            return [base_lr * factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            progress = (self.last_epoch - self.warmup_epochs) / max(
                1, self.total_epochs - self.warmup_epochs
            )
            factor = 0.5 * (1 + np.cos(np.pi * progress))
            return [
                self.min_lr + (base_lr - self.min_lr) * factor
                for base_lr in self.base_lrs
            ]


# ── Training Engine ──────────────────────────────────────────────────────────

class Trainer:
    """Training engine for NeuroFusionNet."""

    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        device: torch.device,
        epochs: int = 50,
        lr: float = 3e-4,
        weight_decay: float = 0.01,
        label_smoothing: float = 0.1,
        warmup_epochs: int = 5,
        patience: int = 10,
        save_dir: str = "checkpoints",
        use_amp: bool = True,
    ):
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self.patience = patience
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_amp = use_amp and device.type == "cuda"

        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

        # Optimizer: different LR for backbone vs new layers
        backbone_params = list(model.cnn.parameters())
        new_params = [p for n, p in model.named_parameters()
                      if not n.startswith("cnn") and p.requires_grad]

        self.optimizer = torch.optim.AdamW([
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": new_params, "lr": lr},
        ], weight_decay=weight_decay)

        # Scheduler
        self.scheduler = CosineWarmupScheduler(
            self.optimizer, warmup_epochs, epochs
        )

        # AMP scaler
        self.scaler = GradScaler(enabled=self.use_amp)

        # Tracking
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "lr": [],
        }
        self.best_val_acc = 0.0
        self.best_epoch = 0
        self.patience_counter = 0

    def train_epoch(self) -> tuple:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, labels in pbar:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item() * images.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{correct/total:.4f}")

        return total_loss / total, correct / total

    @torch.no_grad()
    def validate(self, loader=None) -> tuple:
        """Validate the model."""
        if loader is None:
            loader = self.val_loader
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds, all_labels, all_probs = [], [], []

        for images, labels in loader:
            images = images.to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True)

            with autocast(device_type=self.device.type, enabled=self.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = probs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

        avg_loss = total_loss / total
        accuracy = correct / total
        metrics = compute_metrics(
            np.array(all_labels), np.array(all_preds), np.array(all_probs)
        )
        return avg_loss, accuracy, metrics

    def train(self) -> dict:
        """Full training loop with early stopping."""
        print(f"\n{'='*60}")
        print(f"🧠 NeuroFusionNet Training")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        params = count_parameters(self.model)
        print(f"Parameters: {params['total_millions']:.2f}M total, "
              f"{params['trainable_millions']:.2f}M trainable")
        print(f"Epochs: {self.epochs}, Patience: {self.patience}")
        print(f"{'='*60}\n")

        start_time = time.time()

        for epoch in range(1, self.epochs + 1):
            print(f"\nEpoch {epoch}/{self.epochs}")
            print("-" * 40)

            # Train
            train_loss, train_acc = self.train_epoch()

            # Validate
            val_loss, val_acc, val_metrics = self.validate()

            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[1]["lr"]

            # Record history
            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            print(f"  Train — Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"  Val   — Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            print(f"  LR: {current_lr:.2e}")

            # Checkpointing
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.patience_counter = 0
                save_checkpoint(
                    self.model, self.optimizer, epoch, val_metrics,
                    str(self.save_dir / "best_model.pth"),
                )
                print(f"  ✅ Best model saved! (Acc: {val_acc:.4f})")
            else:
                self.patience_counter += 1
                print(f"  ⏳ Patience: {self.patience_counter}/{self.patience}")

            # Early stopping
            if self.patience_counter >= self.patience:
                print(f"\n🛑 Early stopping at epoch {epoch}")
                break

        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"✅ Training complete in {elapsed/60:.1f} minutes")
        print(f"Best validation accuracy: {self.best_val_acc:.4f} (epoch {self.best_epoch})")

        # Final test evaluation
        print(f"\n📊 Evaluating on test set...")
        from .utils import load_checkpoint
        load_checkpoint(self.model, str(self.save_dir / "best_model.pth"), self.device)
        test_loss, test_acc, test_metrics = self.validate(self.test_loader)
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test Macro F1: {test_metrics['macro_f1']:.4f}")
        print(test_metrics["classification_report"])

        # Save training history
        history_path = str(self.save_dir / "training_history.json")
        with open(history_path, "w") as f:
            serializable_history = {
                k: [float(v) for v in vals] for k, vals in self.history.items()
            }
            serializable_history["test_accuracy"] = float(test_acc)
            serializable_history["test_metrics"] = {
                k: v for k, v in test_metrics.items()
                if k not in ("confusion_matrix", "classification_report")
            }
            json.dump(serializable_history, f, indent=2)

        return {
            "history": self.history,
            "best_val_acc": self.best_val_acc,
            "test_acc": test_acc,
            "test_metrics": test_metrics,
        }


# ── CLI Entry Point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train NeuroFusionNet")
    parser.add_argument("--data-dir", type=str, default="brisc2025/classification_task",
                        help="Path to BRISC 2025 classification_task directory")
    parser.add_argument("--variant", type=str, default="base",
                        choices=["tiny", "base", "large"])
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--save-dir", type=str, default="checkpoints")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Data
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        args.data_dir, batch_size=args.batch_size,
    )
    print(f"Classes: {classes}")
    print(f"Train batches: {len(train_loader)}, Val: {len(val_loader)}, Test: {len(test_loader)}")

    # Model
    model = create_model(args.variant, num_classes=len(classes))

    # Train
    trainer = Trainer(
        model, train_loader, val_loader, test_loader, device,
        epochs=args.epochs, lr=args.lr, patience=args.patience,
        save_dir=args.save_dir, use_amp=not args.no_amp,
    )
    results = trainer.train()


if __name__ == "__main__":
    main()
