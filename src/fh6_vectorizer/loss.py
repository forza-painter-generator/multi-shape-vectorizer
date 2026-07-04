"""
Loss functions for the differentiable vectorizer.

Currently implemented:
  - MSE (pixel-wise RGB loss)
  - Perceptual Loss (VGG16-based feature loss)
  - Combined loss with configurable weights

Reference: diffbmp/pydiffbmp/util/loss_functions.py
"""

import torch
import torch.nn.functional as F
from torch import nn

# ImageNet normalization stats
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


class VGGFeatureExtractor(nn.Module):
    """
    VGG16 feature extractor for perceptual loss.
    Uses layers up to relu3_3 (16 layers) as recommended by diffbmp.
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except ImportError:
            raise ImportError(
                "torchvision is required for perceptual loss. "
                "Install with: pip install torchvision"
            )
        # Use layers up to relu3_3 (index 16 exclusive)
        self.slice = nn.Sequential(*[vgg[i] for i in range(16)])
        for p in self.parameters():
            p.requires_grad = False
        self.to(device)
        self.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] in [0, 1] range
        Returns:
            features: [B, C, H', W']
        """
        # Normalize to ImageNet stats
        x = (x - IMAGENET_MEAN.to(x.device)) / IMAGENET_STD.to(x.device)
        return self.slice(x)


def mse_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Pixel-wise MSE loss in RGB space.

    Args:
        rendered: [B, 3, H, W] in [0, 1]
        target:   [B, 3, H, W] in [0, 1]
    """
    return F.mse_loss(rendered, target)


def perceptual_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    vgg: VGGFeatureExtractor,
) -> torch.Tensor:
    """
    Perceptual loss in VGG feature space.

    Compares relu3_3 features between rendered and target images.

    Args:
        rendered: [B, 3, H, W] in [0, 1]
        target:   [B, 3, H, W] in [0, 1]
        vgg:      VGGFeatureExtractor instance
    """
    with torch.no_grad():
        feat_target = vgg(target)
    feat_rendered = vgg(rendered)
    return F.mse_loss(feat_rendered, feat_target) / 100.0


def combined_loss(
    rendered: torch.Tensor,
    target: torch.Tensor,
    vgg: VGGFeatureExtractor,
    mse_weight: float = 1.0,
    perceptual_weight: float = 0.2,
) -> tuple[torch.Tensor, dict]:
    """
    Combined loss: MSE + perceptual.

    Returns:
        total_loss, {"mse": mse_val, "perceptual": perc_val}
    """
    mse_val = mse_loss(rendered, target)
    perc_val = perceptual_loss(rendered, target, vgg)
    total = mse_weight * mse_val + perceptual_weight * perc_val
    return total, {"mse": mse_val.item(), "perceptual": perc_val.item()}
