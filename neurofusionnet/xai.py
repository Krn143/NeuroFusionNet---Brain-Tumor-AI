"""
Comprehensive XAI (Explainable AI) Suite for NeuroFusionNet.

Five complementary techniques:
  1. Grad-CAM       — CNN gradient-based activation heatmap
  2. Grad-CAM++     — Improved localization with weighted gradients
  3. Attention Rollout — Transformer attention flow across layers
  4. SHAP           — Per-pixel Shapley value contributions
  5. LIME           — Superpixel importance explanation

All methods generate visual heatmaps overlaid on the input MRI.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from .utils import CLASS_NAMES


# ── Grad-CAM ──────────────────────────────────────────────────────────────────

class GradCAM:
    """Gradient-weighted Class Activation Mapping.
    
    Highlights which spatial regions the CNN focuses on for a prediction.
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        # Default: last convolutional layer of MobileNetV2
        if target_layer is None:
            self.target_layer = model.cnn.features[-1]
        else:
            self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM heatmap.
        
        Args:
            input_tensor: (1, 3, H, W) normalized input image
            target_class: Target class index (None = predicted class)
        
        Returns:
            heatmap: (H, W) numpy array in [0, 1]
        """
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass for target class
        target = output[0, target_class]
        target.backward(retain_graph=True)

        # Weighted combination of feature maps
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize and resize
        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ── Grad-CAM++ ────────────────────────────────────────────────────────────────

class GradCAMPlusPlus:
    """Grad-CAM++ — improved localization with higher-order gradients."""

    def __init__(self, model, target_layer=None):
        self.model = model
        self.model.eval()
        if target_layer is None:
            self.target_layer = model.cnn.features[-1]
        else:
            self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        """Generate Grad-CAM++ heatmap."""
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        target = output[0, target_class]
        target.backward(retain_graph=True)

        grads = self.gradients
        acts = self.activations

        # Grad-CAM++ weighting (alpha coefficients)
        grads_power_2 = grads ** 2
        grads_power_3 = grads_power_2 * grads
        sum_acts = acts.sum(dim=[2, 3], keepdim=True)

        alpha = grads_power_2 / (2 * grads_power_2 + sum_acts * grads_power_3 + 1e-7)
        weights = (alpha * F.relu(grads)).sum(dim=[2, 3], keepdim=True)

        cam = (weights * acts).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


# ── Attention Rollout ─────────────────────────────────────────────────────────

class AttentionRollout:
    """Attention Rollout — visualizes cumulative transformer attention flow.
    
    Shows how information from each spatial position flows through all
    transformer layers to the [CLS] token.
    """

    def __init__(self, model, head_fusion="mean", discard_ratio=0.1):
        self.model = model
        self.head_fusion = head_fusion
        self.discard_ratio = discard_ratio

    def generate(self, input_tensor):
        """Generate attention rollout heatmap.
        
        Returns:
            heatmap: (H, W) numpy array in [0, 1]
        """
        self.model.eval()
        with torch.no_grad():
            _ = self.model(input_tensor)

        attention_maps = self.model.get_attention_maps()
        if not attention_maps:
            return np.zeros((224, 224))

        # Process attention maps
        result = None
        for attention in attention_maps:
            # Fuse heads
            if self.head_fusion == "mean":
                attention_heads_fused = attention.mean(dim=1)
            elif self.head_fusion == "max":
                attention_heads_fused = attention.max(dim=1)[0]
            elif self.head_fusion == "min":
                attention_heads_fused = attention.min(dim=1)[0]
            else:
                raise ValueError(f"Unknown head fusion: {self.head_fusion}")

            # Discard low-attention values
            flat = attention_heads_fused.view(attention_heads_fused.size(0), -1)
            _, indices = flat.topk(int(flat.size(-1) * self.discard_ratio), dim=-1, largest=False)
            flat.scatter_(-1, indices, 0)

            # Add identity (residual connection)
            I = torch.eye(attention_heads_fused.size(-1)).to(attention_heads_fused.device)
            a = (attention_heads_fused + I) / 2
            a = a / a.sum(dim=-1, keepdim=True)

            if result is None:
                result = a
            else:
                result = torch.bmm(a, result)

        # Extract [CLS] token attention to patches
        mask = result[0, 0, 1:]  # Skip CLS-to-CLS attention
        num_patches = int(mask.shape[0] ** 0.5)
        mask = mask.reshape(num_patches, num_patches).cpu().numpy()
        mask = cv2.resize(mask, (224, 224))
        mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
        return mask


# ── SHAP Explainer ────────────────────────────────────────────────────────────

class SHAPExplainer:
    """SHAP-based explanation using DeepExplainer.
    
    Computes per-pixel Shapley values showing each pixel's contribution
    to the prediction.
    """

    def __init__(self, model, background_data=None):
        self.model = model
        self.model.eval()
        self.background_data = background_data

    def generate(self, input_tensor, target_class=None):
        """Generate SHAP explanation heatmap.
        
        Falls back to gradient-based approximation if SHAP library
        is not available or background data is insufficient.
        """
        try:
            import shap

            if self.background_data is not None and len(self.background_data) >= 10:
                bg = self.background_data[:50]
            else:
                # Use random samples as background
                bg = torch.randn(20, 3, 224, 224).to(input_tensor.device) * 0.1

            explainer = shap.DeepExplainer(self.model, bg)
            shap_values = explainer.shap_values(input_tensor)

            if target_class is None:
                with torch.no_grad():
                    output = self.model(input_tensor)
                target_class = output.argmax(dim=1).item()

            # Get SHAP values for target class
            if isinstance(shap_values, list):
                sv = np.array(shap_values[target_class])
            else:
                sv = shap_values

            # Aggregate across channels
            heatmap = np.abs(sv[0]).mean(axis=0)
            heatmap = cv2.resize(heatmap, (224, 224))
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            return heatmap

        except Exception:
            # Fallback: gradient-based attribution
            return self._gradient_attribution(input_tensor, target_class)

    def _gradient_attribution(self, input_tensor, target_class=None):
        """Fallback gradient × input attribution."""
        input_tensor = input_tensor.clone().requires_grad_(True)
        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        output[0, target_class].backward()
        grads = input_tensor.grad.data

        # Gradient × Input
        attr = (grads * input_tensor.data).abs().sum(dim=1, keepdim=True)
        attr = attr.squeeze().cpu().numpy()
        attr = cv2.resize(attr, (224, 224))
        attr = (attr - attr.min()) / (attr.max() - attr.min() + 1e-8)
        return attr


# ── LIME Explainer ────────────────────────────────────────────────────────────

class LIMEExplainer:
    """LIME — Local Interpretable Model-agnostic Explanations.
    
    Segments the image into superpixels and identifies which regions
    are most important for the prediction.
    """

    def __init__(self, model, device=None):
        self.model = model
        self.model.eval()
        self.device = device

    def generate(self, input_image_pil, input_tensor, target_class=None, num_samples=300):
        """Generate LIME explanation.
        
        Args:
            input_image_pil: PIL Image (original, unnormalized)
            input_tensor: Normalized tensor for model
            target_class: Target class index
            num_samples: Number of perturbation samples
        
        Returns:
            heatmap: (H, W) numpy array in [0, 1]
        """
        try:
            from lime import lime_image
            from .dataset import get_eval_transforms

            explainer = lime_image.LimeImageExplainer()

            transform = get_eval_transforms()

            def predict_fn(images):
                """Batch prediction for LIME."""
                batch = []
                for img in images:
                    pil_img = Image.fromarray(img.astype(np.uint8))
                    tensor = transform(pil_img)
                    batch.append(tensor)
                batch = torch.stack(batch).to(
                    self.device if self.device else next(self.model.parameters()).device
                )
                with torch.no_grad():
                    outputs = self.model(batch)
                    probs = torch.softmax(outputs, dim=1)
                return probs.cpu().numpy()

            # Resize PIL image to 224x224
            img_array = np.array(input_image_pil.resize((224, 224)))

            explanation = explainer.explain_instance(
                img_array, predict_fn,
                top_labels=4, num_samples=num_samples,
                hide_color=0, random_seed=42,
            )

            if target_class is None:
                with torch.no_grad():
                    output = self.model(input_tensor.to(
                        self.device if self.device else next(self.model.parameters()).device
                    ))
                target_class = output.argmax(dim=1).item()

            # Get explanation mask
            temp, mask = explanation.get_image_and_mask(
                target_class, positive_only=True,
                num_features=10, hide_rest=False,
            )

            # Convert mask to heatmap
            heatmap = mask.astype(np.float32)
            heatmap = cv2.resize(heatmap, (224, 224))
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
            return heatmap

        except Exception:
            # Fallback: return uniform heatmap
            return np.ones((224, 224)) * 0.5


# ── Unified XAI Engine ────────────────────────────────────────────────────────

class XAIEngine:
    """Unified XAI engine that runs all explanation methods.
    
    Provides a single interface to generate explanations from all 5 methods
    and create comparison visualizations.
    """

    def __init__(self, model, device=None, background_data=None):
        self.model = model
        self.device = device or next(model.parameters()).device
        self.model = self.model.to(self.device)

        # Initialize all explainers
        self.gradcam = GradCAM(model)
        self.gradcam_pp = GradCAMPlusPlus(model)
        self.attention_rollout = AttentionRollout(model)
        self.shap_explainer = SHAPExplainer(model, background_data)
        self.lime_explainer = LIMEExplainer(model, device)

    def explain(self, input_tensor, input_image_pil=None, target_class=None,
                methods=None):
        """Generate explanations using specified methods.
        
        Args:
            input_tensor: (1, 3, 224, 224) normalized tensor
            input_image_pil: Original PIL image (needed for LIME)
            target_class: Target class (None = predicted)
            methods: List of methods to use (None = all)
        
        Returns:
            dict of method_name → heatmap (224×224 numpy array)
        """
        all_methods = ["gradcam", "gradcam++", "attention_rollout", "shap", "lime"]
        if methods is None:
            methods = all_methods

        input_tensor = input_tensor.to(self.device)
        results = {}

        # Get prediction info
        with torch.no_grad():
            output = self.model(input_tensor)
            probs = torch.softmax(output, dim=1)
            pred_class = probs.argmax(dim=1).item()
            confidence = probs[0, pred_class].item()

        if target_class is None:
            target_class = pred_class

        results["prediction"] = {
            "class": CLASS_NAMES[pred_class],
            "class_idx": pred_class,
            "confidence": confidence,
            "all_probs": {CLASS_NAMES[i]: probs[0, i].item() for i in range(len(CLASS_NAMES))},
        }

        # Generate explanations
        if "gradcam" in methods:
            try:
                results["gradcam"] = self.gradcam.generate(input_tensor, target_class)
            except Exception:
                results["gradcam"] = np.zeros((224, 224))

        if "gradcam++" in methods:
            try:
                results["gradcam++"] = self.gradcam_pp.generate(input_tensor, target_class)
            except Exception:
                results["gradcam++"] = np.zeros((224, 224))

        if "attention_rollout" in methods:
            try:
                results["attention_rollout"] = self.attention_rollout.generate(input_tensor)
            except Exception:
                results["attention_rollout"] = np.zeros((224, 224))

        if "shap" in methods:
            try:
                results["shap"] = self.shap_explainer.generate(input_tensor, target_class)
            except Exception:
                results["shap"] = np.zeros((224, 224))

        if "lime" in methods and input_image_pil is not None:
            try:
                results["lime"] = self.lime_explainer.generate(
                    input_image_pil, input_tensor, target_class
                )
            except Exception:
                results["lime"] = np.zeros((224, 224))

        return results

    def create_comparison_figure(self, input_tensor, input_image_pil=None,
                                 target_class=None, save_path=None):
        """Create a comparison figure with all XAI methods side by side.
        
        Returns:
            matplotlib Figure
        """
        results = self.explain(input_tensor, input_image_pil, target_class)

        # Get original image for overlay
        img = input_tensor[0].cpu().numpy().transpose(1, 2, 0)
        # Denormalize
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = img * std + mean
        img = np.clip(img, 0, 1)

        # Build figure
        method_keys = [k for k in results if k != "prediction"]
        n_methods = len(method_keys)
        fig, axes = plt.subplots(1, n_methods + 1, figsize=(4 * (n_methods + 1), 4))

        # Original image
        axes[0].imshow(img)
        pred = results["prediction"]
        axes[0].set_title(
            f"Original\n{pred['class']} ({pred['confidence']:.1%})",
            fontsize=10, fontweight="bold",
        )
        axes[0].axis("off")

        # XAI overlays
        method_titles = {
            "gradcam": "Grad-CAM\n(CNN Focus)",
            "gradcam++": "Grad-CAM++\n(Improved)",
            "attention_rollout": "Attention Rollout\n(Transformer Focus)",
            "shap": "SHAP\n(Pixel Importance)",
            "lime": "LIME\n(Superpixel)",
        }

        for idx, method in enumerate(method_keys):
            heatmap = results[method]
            colored_heatmap = cv2.applyColorMap(
                (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
            )
            colored_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2RGB) / 255.0
            overlay = 0.5 * img + 0.5 * colored_heatmap

            axes[idx + 1].imshow(np.clip(overlay, 0, 1))
            axes[idx + 1].set_title(
                method_titles.get(method, method), fontsize=10, fontweight="bold"
            )
            axes[idx + 1].axis("off")

        plt.suptitle(
            "🔬 NeuroFusionNet XAI Report Card",
            fontsize=14, fontweight="bold", y=1.02,
        )
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        return fig


def overlay_heatmap(image_array, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
    """Overlay a heatmap on an image.
    
    Args:
        image_array: (H, W, 3) numpy array in [0, 1]
        heatmap: (H, W) numpy array in [0, 1]
        alpha: Overlay transparency
        colormap: OpenCV colormap
    
    Returns:
        overlay: (H, W, 3) numpy array in [0, 1]
    """
    colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), colormap)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB) / 255.0

    if image_array.shape[:2] != heatmap.shape[:2]:
        colored = cv2.resize(colored, (image_array.shape[1], image_array.shape[0]))

    overlay = (1 - alpha) * image_array + alpha * colored
    return np.clip(overlay, 0, 1)
