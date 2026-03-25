"""
Combination of Segmentation + CNN Classification Pipeline.

Modules:
  - segmentation_model: Lightweight U-Net with MobileNetV2 encoder
  - segmentation_dataset: BRISC 2025 segmentation data loader
  - train_segmentation: Training script for U-Net
  - pipeline: End-to-end Seg → Classification → XAI → LLM pipeline
  - llm_engine: Free LLM integration (Google Gemini) for clinical reports
"""

from .segmentation_model import LightweightUNet, create_unet
from .pipeline import SegClassPipeline
