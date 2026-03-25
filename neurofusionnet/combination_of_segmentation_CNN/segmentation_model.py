"""
Lightweight U-Net with MobileNetV2 Encoder for Brain Tumor Segmentation.

Architecture:
  Encoder: MobileNetV2 pretrained (frozen early layers)
  Bridge:  ASPP (Atrous Spatial Pyramid Pooling) for multi-scale context
  Decoder: 4 upsampling blocks with skip connections
  Output:  Binary mask (sigmoid) — tumor vs. background

~4.5M parameters — suitable for real-time medical imaging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


# ── ASPP (Atrous Spatial Pyramid Pooling) ────────────────────────────────────

class ASPPModule(nn.Module):
    """Captures multi-scale context through parallel dilated convolutions."""

    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        # 1×1 convolution
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        # Dilated convolutions at different rates
        self.conv3x3_r6 = self._dilated_conv(in_channels, out_channels, rate=6)
        self.conv3x3_r12 = self._dilated_conv(in_channels, out_channels, rate=12)
        self.conv3x3_r18 = self._dilated_conv(in_channels, out_channels, rate=18)
        # Global average pooling branch (no normalization — 1×1 spatial)
        self.global_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=True),
            nn.ReLU(inplace=True),
        )
        # Fusion
        self.project = nn.Sequential(
            nn.Conv2d(out_channels * 5, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def _dilated_conv(self, in_ch, out_ch, rate):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=rate, dilation=rate, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        size = x.shape[2:]
        features = [
            self.conv1x1(x),
            self.conv3x3_r6(x),
            self.conv3x3_r12(x),
            self.conv3x3_r18(x),
            F.interpolate(self.global_pool(x), size=size, mode='bilinear', align_corners=False),
        ]
        return self.project(torch.cat(features, dim=1))


# ── Decoder Block ────────────────────────────────────────────────────────────

class DecoderBlock(nn.Module):
    """Upsampling block with skip connection from encoder."""

    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        total_ch = in_channels // 2 + skip_channels
        self.conv_block = nn.Sequential(
            nn.Conv2d(total_ch, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        # Handle size mismatch
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv_block(x)


# ── Lightweight U-Net ────────────────────────────────────────────────────────

class LightweightUNet(nn.Module):
    """U-Net with MobileNetV2 encoder for brain tumor segmentation.

    Encoder stages (MobileNetV2 features):
      Stage 0: features[0:2]   → 16 channels,  H/2
      Stage 1: features[2:4]   → 24 channels,  H/4  
      Stage 2: features[4:7]   → 32 channels,  H/8
      Stage 3: features[7:14]  → 96 channels,  H/16
      Stage 4: features[14:18] → 320 channels, H/32
    
    Args:
        pretrained: Use ImageNet pretrained MobileNetV2 encoder
        freeze_encoder: Fraction of encoder layers to freeze (0.0 to 1.0)
        out_channels: Number of output channels (1 for binary segmentation)
    """

    def __init__(self, pretrained=True, freeze_encoder=0.5, out_channels=1):
        super().__init__()

        # ── Encoder (MobileNetV2) ──
        mobilenet = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        )
        features = mobilenet.features

        self.enc0 = features[0:2]    # 16 ch, stride 2
        self.enc1 = features[2:4]    # 24 ch, stride 4
        self.enc2 = features[4:7]    # 32 ch, stride 8
        self.enc3 = features[7:14]   # 96 ch, stride 16
        self.enc4 = features[14:18]  # 320 ch, stride 32

        # Freeze early layers
        if freeze_encoder > 0:
            all_enc = [self.enc0, self.enc1, self.enc2, self.enc3, self.enc4]
            freeze_count = int(len(all_enc) * freeze_encoder)
            for stage in all_enc[:freeze_count]:
                for param in stage.parameters():
                    param.requires_grad = False

        # ── Bridge (ASPP) ──
        self.bridge = ASPPModule(320, 256)

        # ── Decoder ──
        self.dec4 = DecoderBlock(256, 96, 128)    # 256→128 + 96 skip
        self.dec3 = DecoderBlock(128, 32, 64)     # 128→64 + 32 skip
        self.dec2 = DecoderBlock(64, 24, 32)      # 64→32 + 24 skip
        self.dec1 = DecoderBlock(32, 16, 16)      # 32→16 + 16 skip

        # ── Final output ──
        self.final_upsample = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_channels, 1),
        )

    def forward(self, x):
        """Forward pass.
        
        Args:
            x: Input image tensor (B, 3, H, W)
        
        Returns:
            Logits tensor (B, 1, H, W) — apply sigmoid for probabilities
        """
        # Encoder
        e0 = self.enc0(x)     # (B, 16, H/2, W/2)
        e1 = self.enc1(e0)    # (B, 24, H/4, W/4)
        e2 = self.enc2(e1)    # (B, 32, H/8, W/8)
        e3 = self.enc3(e2)    # (B, 96, H/16, W/16)
        e4 = self.enc4(e3)    # (B, 320, H/32, W/32)

        # Bridge
        bridge = self.bridge(e4)  # (B, 256, H/32, W/32)

        # Decoder with skip connections
        d4 = self.dec4(bridge, e3)  # (B, 128, H/16, W/16)
        d3 = self.dec3(d4, e2)      # (B, 64, H/8, W/8)
        d2 = self.dec2(d3, e1)      # (B, 32, H/4, W/4)
        d1 = self.dec1(d2, e0)      # (B, 16, H/2, W/2)

        # Final upsample to original size
        out = self.final_upsample(d1)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:], mode='bilinear', align_corners=False)
        out = self.final_conv(out)
        return out


# ── Loss Functions ───────────────────────────────────────────────────────────

class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = torch.sigmoid(pred)
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1 - (2. * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )


class CombinedSegLoss(nn.Module):
    """Combined Dice + BCE loss for stable segmentation training."""

    def __init__(self, dice_weight=0.5, bce_weight=0.5):
        super().__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight

    def forward(self, pred, target):
        return self.dice_weight * self.dice(pred, target) + \
               self.bce_weight * self.bce(pred, target)


# ── Metrics ──────────────────────────────────────────────────────────────────

def dice_score(pred, target, threshold=0.5):
    """Compute Dice coefficient."""
    pred = (torch.sigmoid(pred) > threshold).float()
    smooth = 1.0
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)


def iou_score(pred, target, threshold=0.5):
    """Compute IoU (Jaccard Index)."""
    pred = (torch.sigmoid(pred) > threshold).float()
    smooth = 1.0
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)


# ── Factory ──────────────────────────────────────────────────────────────────

def create_unet(pretrained=True, freeze_encoder=0.5):
    """Create a lightweight U-Net model."""
    return LightweightUNet(pretrained=pretrained, freeze_encoder=freeze_encoder)


if __name__ == "__main__":
    model = create_unet()
    x = torch.randn(2, 3, 224, 224)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params:     {total / 1e6:.2f}M")
    print(f"Trainable params: {trainable / 1e6:.2f}M")
