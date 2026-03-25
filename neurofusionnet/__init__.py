"""
NeuroFusionNet — Hybrid CNN-ViT for Brain Tumor MRI Classification
with MedGemma LLM Integration and Comprehensive XAI Suite.

A lightweight hybrid architecture that combines MobileNetV2 CNN backbone
with a 4-layer Vision Transformer encoder for accurate brain tumor
classification from T1-weighted MRI scans.

Classes: Glioma, Meningioma, Pituitary Tumor, No Tumor
Dataset: BRISC 2025 (6,000 images)
"""

__version__ = "1.0.0"
__author__ = "NeuroFusionNet Team"

from .model import NeuroFusionNet
from .dataset import BrainTumorDataset, get_dataloaders

__all__ = [
    "NeuroFusionNet",
    "BrainTumorDataset",
    "get_dataloaders",
]
