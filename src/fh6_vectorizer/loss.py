"""
Loss functions for the differentiable vectorizer.

Implemented:
  - MSE (pixel-wise RGB loss)
  - L1 / Huber (robust alternatives to MSE)
  - Perceptual Loss (VGG16-based feature loss)
  - Grayscale MSE (luminance-only loss)
  - Alpha / Opacity Regularization
  - sRGB ↔ Linear color space conversion utilities
  - Combined loss with configurable weights

Unavailable (verified by diffbmp):
  - SSIM: loss diverges (+21.7%), conflicts with gradient descent
  - Edge Loss (Sobel): gradients conflict with other losses

Reference: diffbmp/pydiffbmp/util/loss_functions.py
"""

import torch
import torch.nn.functional as F
from torch import nn

# ImageNet normalization stats
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# sRGB ↔ Linear conversion (for physically correct color blending)
# Reference: vinylizer/src/cuda/color_utils.cuh

def srgb_to_linear(srgb: torch.Tensor) -> torch.Tensor:
    """
    Convert sRGB [0,1] to linear space.
    Uses the standard sRGB transfer function. Clamps input to [0,1].

    Args:
        srgb: tensor of any shape in [0, 1]
    Returns:
        linear: same shape in [0, 1]
    """
    srgb = srgb.clamp(0.0, 1.0)
    low_mask = srgb <= 0.04045
    out = torch.where(
        low_mask,
        srgb / 12.92,
        torch.pow((srgb + 0.055) / 1.055, 2.4),
    )
    return out


def linear_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    """
    Convert linear [0,1] to sRGB space. Clamps input to [0,1].

    Args:
        linear: tensor of any shape in [0, 1]
    Returns:
        srgb: same shape in [0, 1]
    """
    # Clamp to avoid pow backward NaN near 0
    linear = linear.clamp(min=1e-8, max=1.0)
    low_mask = linear <= 0.0031308
    out = torch.where(
        low_mask,
        linear * 12.92,
        1.055 * torch.pow(linear, 1.0 / 2.4) - 0.055,
    )
    return out


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


def l1_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Pixel-wise L1 (MAE) loss in RGB space.

    More robust to outliers than MSE — a single wildly-wrong shape
    won't produce a disproportionately large gradient.

    Args:
        rendered: [B, 3, H, W] or [3, H, W] in [0, 1]
        target:   same shape in [0, 1]
    """
    return F.l1_loss(rendered, target)


def huber_loss(
    rendered: torch.Tensor, target: torch.Tensor, delta: float = 0.1
) -> torch.Tensor:
    """
    Huber (smooth L1) loss — quadratic near zero, linear for large errors.

    Args:
        rendered: [B, 3, H, W] or [3, H, W] in [0, 1]
        target:   same shape in [0, 1]
        delta:    threshold between L2 and L1 behavior (default: 0.1)

    Reference: diffbmp loss_functions.py
    """
    return F.smooth_l1_loss(rendered, target, beta=delta)


def grayscale_mse_loss(rendered: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Luminance-only MSE — reduces sensitivity to color deviations.

    Converts to grayscale using BT.601 luma coefficients, then MSE.

    Args:
        rendered: [..., 3, H, W] in [0, 1]
        target:   same shape in [0, 1]
    """
    lum_weights = torch.tensor([0.299, 0.587, 0.114], device=rendered.device)
    # Handle both 3-dim and 4-dim inputs
    if rendered.ndim == 3:
        r_gray = (rendered * lum_weights.view(3, 1, 1)).sum(dim=0)
        t_gray = (target * lum_weights.view(3, 1, 1)).sum(dim=0)
    else:
        r_gray = (rendered * lum_weights.view(1, 3, 1, 1)).sum(dim=1)
        t_gray = (target * lum_weights.view(1, 3, 1, 1)).sum(dim=1)
    return F.mse_loss(r_gray, t_gray)


def alpha_regularization(opacity: torch.Tensor, target_mean: float = 0.5) -> torch.Tensor:
    """
    Encourage shapes to have meaningful opacity.

    Penalizes:
      - Too-low opacity (invisible/useless shapes)
      - Too-high opacity (single shape domination)

    Args:
        opacity: [N] tensor of shape opacities in [0, 1]
        target_mean: desired mean opacity (default 0.5)
    """
    return F.mse_loss(opacity, torch.full_like(opacity, target_mean))


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
    l1_weight: float = 0.0,
    huber_weight: float = 0.0,
    grayscale_weight: float = 0.0,
) -> tuple[torch.Tensor, dict]:
    """
    Combined loss: weighted sum of available loss components.

    Recommended weights (from diffbmp configs/default.json):
        mse: 1.0, perceptual: 0.2

    Args:
        rendered: [B, 3, H, W] in [0, 1]
        target:   [B, 3, H, W] in [0, 1]
        vgg:      VGGFeatureExtractor instance (required for perceptual)
        mse_weight, perceptual_weight, l1_weight, huber_weight, grayscale_weight:
            weights for each loss component

    Returns:
        total_loss, {"mse": ..., "perceptual": ..., "l1": ..., "huber": ..., "grayscale": ...}
    """
    total = torch.tensor(0.0, device=rendered.device)
    loss_dict = {}

    if mse_weight > 0:
        mse_val = mse_loss(rendered, target)
        total = total + mse_weight * mse_val
        loss_dict["mse"] = mse_val.item()

    if perceptual_weight > 0 and vgg is not None:
        perc_val = perceptual_loss(rendered, target, vgg)
        total = total + perceptual_weight * perc_val
        loss_dict["perceptual"] = perc_val.item()

    if l1_weight > 0:
        l1_val = l1_loss(rendered, target)
        total = total + l1_weight * l1_val
        loss_dict["l1"] = l1_val.item()

    if huber_weight > 0:
        hub_val = huber_loss(rendered, target)
        total = total + huber_weight * hub_val
        loss_dict["huber"] = hub_val.item()

    if grayscale_weight > 0:
        gray_val = grayscale_mse_loss(rendered, target)
        total = total + grayscale_weight * gray_val
        loss_dict["grayscale"] = gray_val.item()

    return total, loss_dict
