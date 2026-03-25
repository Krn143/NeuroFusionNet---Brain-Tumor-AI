"""
Utility functions for NeuroFusionNet.
Includes seed setting, device detection, metrics, and checkpoint helpers.
"""

import os
import random
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    classification_report,
)


# ── Class labels ──────────────────────────────────────────────────────────────
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224

# ── Tumor info for reports ────────────────────────────────────────────────────
TUMOR_INFO = {
    "Glioma": {
        "description": "Gliomas arise from glial cells in the brain or spine. They are the most common type of primary brain tumor.",
        "severity": "High",
        "common_treatments": ["Surgery", "Radiation therapy", "Chemotherapy (Temozolomide)", "Targeted therapy"],
        "typical_location": "Cerebral hemispheres, brainstem, cerebellum",
        "prognosis": "Variable — depends on grade (I-IV). Low-grade gliomas have better prognosis.",
    },
    "Meningioma": {
        "description": "Meningiomas develop from the meninges, the membranes surrounding the brain and spinal cord. Most are benign.",
        "severity": "Low to Moderate",
        "common_treatments": ["Observation (watch and wait)", "Surgery", "Radiation therapy"],
        "typical_location": "Convexity, parasagittal, sphenoid wing, posterior fossa",
        "prognosis": "Generally favorable — most are benign (WHO Grade I).",
    },
    "Pituitary": {
        "description": "Pituitary adenomas are tumors of the pituitary gland. Most are benign and slow-growing.",
        "severity": "Low to Moderate",
        "common_treatments": ["Medication (dopamine agonists)", "Transsphenoidal surgery", "Radiation therapy"],
        "typical_location": "Sella turcica (pituitary fossa)",
        "prognosis": "Generally excellent with appropriate treatment.",
    },
    "No Tumor": {
        "description": "No tumorous growth detected in the MRI scan. The brain parenchyma appears within normal limits.",
        "severity": "None",
        "common_treatments": ["No treatment required", "Routine follow-up if symptomatic"],
        "typical_location": "N/A",
        "prognosis": "N/A — No pathological findings.",
    },
}


def set_seed(seed: int = 42):
    """Set random seed for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_parameters(model: torch.nn.Module) -> dict:
    """Count model parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = total - trainable
    return {
        "total": total,
        "trainable": trainable,
        "frozen": frozen,
        "total_millions": total / 1e6,
        "trainable_millions": trainable / 1e6,
    }


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Compute comprehensive classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(NUM_CLASSES)
    )
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro"
    )
    cm = confusion_matrix(y_true, y_pred, labels=range(NUM_CLASSES))

    metrics = {
        "accuracy": acc,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "per_class_precision": dict(zip(CLASS_NAMES, precision.tolist())),
        "per_class_recall": dict(zip(CLASS_NAMES, recall.tolist())),
        "per_class_f1": dict(zip(CLASS_NAMES, f1.tolist())),
        "confusion_matrix": cm,
        "classification_report": classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, digits=4
        ),
    }

    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            per_class_auc = roc_auc_score(
                y_true, y_prob, multi_class="ovr", average=None
            )
            metrics["macro_auc"] = auc
            metrics["per_class_auc"] = dict(zip(CLASS_NAMES, per_class_auc.tolist()))
        except ValueError:
            metrics["macro_auc"] = None
            metrics["per_class_auc"] = None

    return metrics


def save_checkpoint(model, optimizer, epoch, metrics, path):
    """Save training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(model, path, device=None, optimizer=None):
    """Load training checkpoint."""
    if device is None:
        device = get_device()
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("epoch", 0), checkpoint.get("metrics", {})
