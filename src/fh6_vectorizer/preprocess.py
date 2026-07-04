"""
Image preprocessing for the differentiable vectorizer.

Generates importance maps that guide shape initialization and relocation:
  1. K-means color quantization — reduce color space complexity
  2. Canny edge detection — identify structural boundaries
  3. Visual saliency — highlight regions the human eye focuses on
  4. Fused importance map — weighted combination of the above

Reference:
  - vinylizer/src/preprocess/preprocessor.h + .cpp
  - IMPLEMENTATION_PLAN.md §7
"""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch


def kmeans_quantize(
    image: np.ndarray,
    k: int = 16,
    attempts: int = 3,
) -> np.ndarray:
    """
    Reduce image to K dominant colors using K-means.

    This helps the optimizer focus on structural reconstruction
    rather than precisely matching every subtle color variation.

    Args:
        image: [H, W, 3] uint8 image
        k: number of color clusters (default 16)
        attempts: K-means attempts for robustness

    Returns:
        quantized: [H, W, 3] uint8 quantized image
    """
    h, w = image.shape[:2]
    pixels = image.reshape(-1, 3).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    _, labels, centers = cv2.kmeans(
        pixels, k, None, criteria, attempts, cv2.KMEANS_RANDOM_CENTERS
    )

    centers = centers.astype(np.uint8)
    quantized = centers[labels.flatten()].reshape(h, w, 3)
    return quantized


def canny_edge_map(
    image: np.ndarray,
    low_threshold: float = 50,
    high_threshold: float = 150,
    blur_sigma: float = 1.0,
) -> np.ndarray:
    """
    Generate edge importance map using Canny edge detection.

    Edges indicate structural boundaries that need more shapes.

    Args:
        image: [H, W, 3] uint8 image
        low_threshold, high_threshold: Canny thresholds
        blur_sigma: Gaussian blur before edge detection

    Returns:
        edge_map: [H, W] float32 in [0, 1]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), sigmaX=blur_sigma)

    edges = cv2.Canny(gray, low_threshold, high_threshold)
    # Dilate edges slightly for importance spreading
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

    # Distance transform: pixels near edges get higher importance
    dist = cv2.distanceTransform(
        (255 - edges).astype(np.uint8), cv2.DIST_L2, 5
    )
    # Invert so edges = high importance
    max_dist = dist.max() + 1e-8
    edge_map = 1.0 - dist / max_dist
    return edge_map.astype(np.float32)


def variance_map(image: np.ndarray, kernel_size: int = 16) -> np.ndarray:
    """
    Compute local color variance as a proxy for detail/complexity.

    Regions with high color variance need more shapes.

    Args:
        image: [H, W, 3] uint8
        kernel_size: size of the local window

    Returns:
        variance: [H, W] float32 in [0, 1]
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    mean = cv2.blur(gray, (kernel_size, kernel_size))
    mean_sq = cv2.blur(gray * gray, (kernel_size, kernel_size))
    var = mean_sq - mean * mean
    # Normalize
    var_max = var.max() + 1e-8
    return (var / var_max).astype(np.float32)


def compute_importance_map(
    image: np.ndarray,
    edge_weight: float = 0.5,
    variance_weight: float = 0.3,
    uniform_weight: float = 0.2,
    kmeans_k: Optional[int] = None,
) -> torch.Tensor:
    """
    Fuse multiple cues into a single importance map.

    The importance map guides:
      - Shape initialization: weighted random sampling
      - Shape relocation: weighted random sampling of new positions

    Args:
        image: [H, W, 3] uint8 image
        edge_weight: weight for edge map (structural boundaries)
        variance_weight: weight for color variance (detail regions)
        uniform_weight: weight for uniform baseline (coverage everywhere)
        kmeans_k: if set, quantize before computing importance

    Returns:
        importance: [H, W] float32 tensor in [0, 1]

    Example:
        img = cv2.imread("photo.jpg")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imp = compute_importance_map(img)
        # Use imp for weighted sampling:
        flat = imp.flatten()
        idx = torch.multinomial(flat / flat.sum(), num_samples)
    """
    # Optional color quantization
    if kmeans_k is not None:
        image = kmeans_quantize(image, k=kmeans_k)

    # Edge map
    edge = canny_edge_map(image)

    # Variance map
    variance = variance_map(image)

    # Uniform baseline (all pixels equal importance)
    uniform = np.ones((image.shape[0], image.shape[1]), dtype=np.float32)

    # Weighted fusion
    importance = (
        edge_weight * edge
        + variance_weight * variance
        + uniform_weight * uniform
    )

    # Normalize to [0, 1]
    imp_max = importance.max() + 1e-8
    importance = importance / imp_max

    return torch.from_numpy(importance)


def importance_weighted_sample(
    importance: torch.Tensor,
    num_samples: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Sample positions from an importance map using weighted random sampling.

    Args:
        importance: [H, W] importance map
        num_samples: number of positions to sample

    Returns:
        (cx, cy): two [num_samples] tensors of pixel coordinates
    """
    H, W = importance.shape
    flat = importance.flatten().clamp(min=0)
    flat_sum = flat.sum()
    if flat_sum < 1e-8:
        flat = torch.ones_like(flat)
        flat_sum = flat.sum()
    flat = flat / flat_sum

    indices = torch.multinomial(flat, num_samples, replacement=True)
    cy = (indices // W).float()
    cx = (indices % W).float()
    return cx, cy


def color_from_target(
    target: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    noise_std: float = 0.05,
) -> torch.Tensor:
    """
    Sample colors from the target image at given positions.

    Adds small Gaussian noise to prevent degenerate identical colors.

    Args:
        target: [3, H, W] target image in [0, 1]
        cx, cy: [N] sample positions (pixel coordinates)
        noise_std: standard deviation of additive noise

    Returns:
        colors: [N, 3] sampled colors in [0, 1]
    """
    H, W = target.shape[1], target.shape[2]
    # Clamp coordinates to valid range
    cx = cx.clamp(0, W - 1)
    cy = cy.clamp(0, H - 1)

    # Bilinear interpolation at sample points
    cx_norm = cx / (W - 1) * 2 - 1  # → [-1, 1]
    cy_norm = cy / (H - 1) * 2 - 1
    grid = torch.stack([cx_norm, cy_norm], dim=-1).unsqueeze(0).unsqueeze(0)  # [1, 1, N, 2]

    target_batched = target.unsqueeze(0)  # [1, 3, H, W]
    colors = F.grid_sample(
        target_batched, grid,
        mode="bilinear", padding_mode="border", align_corners=True,
    )  # [1, 3, 1, N]
    colors = colors.squeeze(0).squeeze(1).T  # [N, 3]

    # Add noise and re-clamp
    if noise_std > 0:
        colors = colors + torch.randn_like(colors) * noise_std
    return colors.clamp(0.0, 1.0)


# Import here to avoid circular dependency at module level
import torch.nn.functional as F
