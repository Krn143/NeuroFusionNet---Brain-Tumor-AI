"""
NeuroFusionNet — Hybrid CNN-ViT Architecture for Brain Tumor Classification.

Architecture:
  MobileNetV2 (CNN) → PatchTokenizer → TransformerEncoder (4 layers)
  → FusionHead (CNN global pool + Transformer [CLS] + SE attention) → Classifier

~8-10M parameters — lightweight yet powerful.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import math


# ── CNN Backbone ──────────────────────────────────────────────────────────────

class CNNBackbone(nn.Module):
    """MobileNetV2 feature extractor (pretrained on ImageNet).
    
    Extracts spatial feature maps from input images.
    Output: (B, 1280, 7, 7) for 224×224 input.
    """

    def __init__(self, pretrained: bool = True, freeze_ratio: float = 0.7):
        super().__init__()
        mobilenet = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        self.features = mobilenet.features  # All convolutional layers
        self.out_channels = 1280

        # Freeze early layers (lower-level features are universal)
        total_layers = len(self.features)
        freeze_until = int(total_layers * freeze_ratio)
        for i, layer in enumerate(self.features):
            if i < freeze_until:
                for param in layer.parameters():
                    param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)


# ── Patch Tokenizer ──────────────────────────────────────────────────────────

class PatchTokenizer(nn.Module):
    """Converts CNN feature maps into a sequence of patch tokens.
    
    Reshapes (B, C, H, W) → (B, H*W, embed_dim) via learned linear projection.
    """

    def __init__(self, in_channels: int = 1280, embed_dim: int = 256):
        super().__init__()
        self.projection = nn.Sequential(
            nn.LayerNorm(in_channels),
            nn.Linear(in_channels, embed_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        # Reshape: (B, C, H, W) → (B, H*W, C)
        x = x.flatten(2).transpose(1, 2)
        # Project: (B, H*W, C) → (B, H*W, embed_dim)
        return self.projection(x)


# ── Multi-Head Self-Attention ─────────────────────────────────────────────────

class MultiHeadSelfAttention(nn.Module):
    """Standard multi-head self-attention with scaled dot-product."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # Store attention weights for XAI visualization
        self.attention_weights = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        # Store for attention visualization
        self.attention_weights = attn.detach()

        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj_drop(self.proj(x))
        return x


# ── Transformer Block ────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Pre-norm Transformer block: LayerNorm → MHSA → LayerNorm → FFN."""

    def __init__(self, embed_dim: int = 256, num_heads: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ── Transformer Encoder ──────────────────────────────────────────────────────

class TransformerEncoder(nn.Module):
    """Transformer encoder with learnable [CLS] token and positional embeddings.
    
    Processes patch tokens through N transformer blocks.
    Returns the [CLS] token representation for classification.
    """

    def __init__(self, num_patches: int = 49, embed_dim: int = 256,
                 depth: int = 4, num_heads: int = 8, mlp_ratio: float = 4.0,
                 dropout: float = 0.1):
        super().__init__()
        # Learnable [CLS] token
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        # Positional embeddings (learnable)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim) * 0.02
        )
        self.pos_drop = nn.Dropout(dropout)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        # Prepend [CLS] token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        # Add positional embeddings
        x = self.pos_drop(x + self.pos_embed)
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        # Return [CLS] token output
        return x[:, 0]


# ── Squeeze-and-Excitation Channel Attention ──────────────────────────────────

class SqueezeExcitation(nn.Module):
    """SE block for adaptive feature weighting in the fusion head."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.fc(x)
        return x * scale


# ── Fusion Head ───────────────────────────────────────────────────────────────

class FusionHead(nn.Module):
    """Dual-stream fusion: CNN global pool + Transformer [CLS] + SE attention.
    
    Combines local CNN features and global Transformer features with
    learned attention gating for adaptive feature weighting.
    """

    def __init__(self, cnn_dim: int = 1280, transformer_dim: int = 256,
                 hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        fused_dim = cnn_dim + transformer_dim
        self.se = SqueezeExcitation(fused_dim)
        self.fc = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.GELU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )

    def forward(self, cnn_feat: torch.Tensor, transformer_feat: torch.Tensor) -> torch.Tensor:
        # Concatenate both feature streams
        fused = torch.cat([cnn_feat, transformer_feat], dim=-1)
        # Apply channel attention
        fused = self.se(fused)
        # Process through MLP
        return self.fc(fused)


# ── NeuroFusionNet (Main Model) ──────────────────────────────────────────────

class NeuroFusionNet(nn.Module):
    """Hybrid CNN-ViT Architecture for Brain Tumor Classification.
    
    Combines MobileNetV2 (local spatial features) with a 4-layer
    Transformer encoder (global context) via SE-gated fusion.
    
    Args:
        num_classes: Number of output classes (default: 4)
        embed_dim: Transformer embedding dimension (default: 256)
        depth: Number of transformer layers (default: 4)
        num_heads: Number of attention heads (default: 8)
        pretrained: Use pretrained CNN backbone (default: True)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(
        self,
        num_classes: int = 4,
        embed_dim: int = 256,
        depth: int = 4,
        num_heads: int = 8,
        pretrained: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_classes = num_classes

        # CNN backbone: extracts (B, 1280, 7, 7) feature maps
        self.cnn = CNNBackbone(pretrained=pretrained)

        # CNN global pooling for fusion
        self.cnn_pool = nn.AdaptiveAvgPool2d(1)

        # Patch tokenizer: maps CNN features to transformer tokens
        self.tokenizer = PatchTokenizer(self.cnn.out_channels, embed_dim)

        # Transformer encoder (49 patches from 7×7 CNN output)
        self.transformer = TransformerEncoder(
            num_patches=49, embed_dim=embed_dim, depth=depth,
            num_heads=num_heads, dropout=dropout,
        )

        # Fusion head: combines CNN + Transformer features
        self.fusion = FusionHead(
            cnn_dim=self.cnn.out_channels, transformer_dim=embed_dim, dropout=dropout * 3,
        )

        # Classification head
        self.classifier = nn.Linear(256, num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize non-pretrained weights."""
        for m in [self.tokenizer, self.transformer, self.fusion, self.classifier]:
            for p in m.modules():
                if isinstance(p, nn.Linear):
                    nn.init.trunc_normal_(p.weight, std=0.02)
                    if p.bias is not None:
                        nn.init.zeros_(p.bias)
                elif isinstance(p, nn.LayerNorm):
                    nn.init.ones_(p.weight)
                    nn.init.zeros_(p.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1. CNN feature extraction
        cnn_features = self.cnn(x)  # (B, 1280, 7, 7)

        # 2. CNN global pooling (for fusion)
        cnn_global = self.cnn_pool(cnn_features).flatten(1)  # (B, 1280)

        # 3. Convert CNN features to patch tokens
        tokens = self.tokenizer(cnn_features)  # (B, 49, embed_dim)

        # 4. Transformer encoding (returns [CLS] token)
        transformer_out = self.transformer(tokens)  # (B, embed_dim)

        # 5. Fusion: combine CNN and Transformer features
        fused = self.fusion(cnn_global, transformer_out)  # (B, 256)

        # 6. Classification
        logits = self.classifier(fused)  # (B, num_classes)
        return logits

    def get_attention_maps(self) -> list:
        """Get attention weights from all transformer layers for XAI."""
        attention_maps = []
        for block in self.transformer.blocks:
            if block.attn.attention_weights is not None:
                attention_maps.append(block.attn.attention_weights)
        return attention_maps

    def get_cnn_features(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw CNN feature maps (for Grad-CAM)."""
        return self.cnn(x)


# ── Model factory ─────────────────────────────────────────────────────────────

def create_model(variant: str = "tiny", num_classes: int = 4,
                 pretrained: bool = True) -> NeuroFusionNet:
    """Create a NeuroFusionNet model with predefined configurations.
    
    Variants:
        "tiny"  — 4 layers, 128 dim, 4 heads (~5M params)
        "base"  — 4 layers, 256 dim, 8 heads (~8M params)
        "large" — 6 layers, 384 dim, 8 heads (~15M params)
    """
    configs = {
        "tiny": {"embed_dim": 128, "depth": 4, "num_heads": 4, "dropout": 0.1},
        "base": {"embed_dim": 256, "depth": 4, "num_heads": 8, "dropout": 0.1},
        "large": {"embed_dim": 384, "depth": 6, "num_heads": 8, "dropout": 0.15},
    }
    if variant not in configs:
        raise ValueError(f"Unknown variant '{variant}'. Choose from {list(configs.keys())}")

    return NeuroFusionNet(num_classes=num_classes, pretrained=pretrained, **configs[variant])


if __name__ == "__main__":
    # Quick test
    model = create_model("base")
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.2f}M")
    print(f"Trainable params: {trainable_params / 1e6:.2f}M")
