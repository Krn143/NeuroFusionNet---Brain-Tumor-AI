"""
Evaluation module for NeuroFusionNet.

Generates:
  - Confusion matrix plots
  - ROC curves with AUC
  - Per-class metrics tables
  - Model comparison charts
"""

import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve,
    auc,
    confusion_matrix,
    precision_recall_curve,
)
from tqdm import tqdm

from .model import NeuroFusionNet
from .utils import CLASS_NAMES, NUM_CLASSES, compute_metrics, get_device


# ── Evaluation Engine ─────────────────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model, dataloader, device=None):
    """Run full evaluation and return predictions, labels, probabilities."""
    if device is None:
        device = get_device()
    model = model.to(device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []

    for images, labels in tqdm(dataloader, desc="Evaluating", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with autocast(device_type=device.type, enabled=device.type == "cuda"):
            outputs = model(images)

        probs = torch.softmax(outputs, dim=1)
        _, predicted = probs.max(1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    return (
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
    )


# ── Visualization Functions ──────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, save_path=None, normalize=True):
    """Plot confusion matrix heatmap."""
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=".2f" if normalize else "d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=ax,
        cbar_kws={"shrink": 0.8},
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix — NeuroFusionNet", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_roc_curves(y_true, y_prob, save_path=None):
    """Plot ROC curves for each class."""
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4"]

    for i, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        y_bin = (np.array(y_true) == i).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, lw=2,
                label=f"{cls_name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — NeuroFusionNet", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_training_history(history, save_path=None):
    """Plot training curves (loss, accuracy, learning rate)."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss
    axes[0].plot(epochs, history["train_loss"], "b-", label="Train", linewidth=2)
    axes[0].plot(epochs, history["val_loss"], "r-", label="Val", linewidth=2)
    axes[0].set_title("Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history["train_acc"], "b-", label="Train", linewidth=2)
    axes[1].plot(epochs, history["val_acc"], "r-", label="Val", linewidth=2)
    axes[1].set_title("Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Learning Rate
    axes[2].plot(epochs, history["lr"], "g-", linewidth=2)
    axes[2].set_title("Learning Rate", fontsize=13, fontweight="bold")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("LR")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.3)

    plt.suptitle("NeuroFusionNet Training History", fontsize=15, fontweight="bold", y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_per_class_metrics(metrics, save_path=None):
    """Plot per-class precision, recall, F1 as grouped bar chart."""
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(CLASS_NAMES))
    width = 0.25

    precision = [metrics["per_class_precision"][c] for c in CLASS_NAMES]
    recall = [metrics["per_class_recall"][c] for c in CLASS_NAMES]
    f1 = [metrics["per_class_f1"][c] for c in CLASS_NAMES]

    bars1 = ax.bar(x - width, precision, width, label="Precision", color="#FF6B6B", alpha=0.85)
    bars2 = ax.bar(x, recall, width, label="Recall", color="#4ECDC4", alpha=0.85)
    bars3 = ax.bar(x + width, f1, width, label="F1-Score", color="#45B7D1", alpha=0.85)

    ax.set_xlabel("Class", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Metrics — NeuroFusionNet", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis="y")

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points", ha="center", va="bottom",
                        fontsize=8)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
