"""
Template Generator: Mathematical gradient shape templates.

Produces:
  - hard templates: binary {0,1} bitmaps (forward rendering)
  - soft templates: continuous gradient [0,1] bitmaps (for gradient flow)

All templates use mathematically-defined gradients — no Gaussian blur needed.
"""

from pathlib import Path

import cv2
import numpy as np
import torch

# --- Config ---
TEMPLATE_SIZE = 128  # px, template texture resolution

# Synthetic gradient shape indices → FH6 type_code mapping
# These 5 mathematical gradient shapes approximate FH6 Gradient_Shapes.
# Like vinylizer's alpha_228, the mathematical definition differs slightly
# from the game engine rendering, but the visual impact is negligible.
# See IMPLEMENTATION_PLAN.md §3.7 for rationale.
GRADIENT_SHAPE_INDICES = {
    0: 1048777 + 28 - 1,  # gradient_ellipse1 → FH6 Gradient_Shapes #28
    1: 1048777 + 11 - 1,  # gradient_rect1    → FH6 Gradient_Shapes #11
    2: 1048777 + 12 - 1,  # gradient_rect2    → FH6 Gradient_Shapes #12
    3: 1048777 + 13 - 1,  # gradient_rect3    → FH6 Gradient_Shapes #13
    4: 1048777 + 16 - 1,  # gradient_rect4    → FH6 Gradient_Shapes #16
}


def generate_synthetic_templates(
    num_types: int = 5,
    template_size: int = TEMPLATE_SIZE,
    device: str = "cpu",
) -> dict:
    """
    Generate 5 mathematical gradient templates (no FH6 data dependency).

    All use continuous gradient — no Gaussian blur needed.
    """
    hard = []
    soft = []
    names = []
    s = template_size
    center = s // 2

    y_idx, x_idx = np.mgrid[:s, :s]
    tx = x_idx.astype(np.float32) / (s - 1)
    ty = y_idx.astype(np.float32) / (s - 1)
    cx_n = center / (s - 1)
    cy_n = center / (s - 1)

    def _grad_shape(name, hard_mask, soft_grad):
        names.append(name)
        hard.append(torch.from_numpy(hard_mask.astype(np.float32)))
        soft.append(torch.from_numpy(soft_grad.astype(np.float32)))

    # 0: Gradient Ellipse 1 (center→edge radial, like alpha_228 / FH6 #28)
    d = np.sqrt(((tx - cx_n) / 0.45) ** 2 + ((ty - cy_n) / 0.45) ** 2)
    ellipse_mask = (d <= 1.0).astype(np.float32)
    grad = np.clip(1.0 - d, 0.0, 1.0)
    _grad_shape("gradient_ellipse1", ellipse_mask, grad)

    # Shared rect mask for gradient rects 1-4
    rect_mask = np.zeros((s, s), dtype=np.float32)
    m = int(s * 0.1)
    cv2.rectangle(rect_mask, (m, m), (s - m, s - m), 1.0, -1)

    # 1: Gradient Rect 1 (center→right edge, FH6 #11)
    grad = np.clip(1.0 - 2.0 * np.maximum(tx - 0.5, 0.0), 0.0, 1.0) * rect_mask
    _grad_shape("gradient_rect1", rect_mask, grad)

    # 2: Gradient Rect 2 (top-left→bottom-right, FH6 #12)
    grad = np.clip(1.0 - (tx + ty) * 0.5, 0.0, 1.0) * rect_mask
    _grad_shape("gradient_rect2", rect_mask, grad)

    # 3: Gradient Rect 3 (left edge→right edge, FH6 #13)
    grad = np.clip(1.0 - tx, 0.0, 1.0) * rect_mask
    _grad_shape("gradient_rect3", rect_mask, grad)

    # 4: Gradient Rect 4 (three corners→bottom-right, FH6 #16)
    grad = np.clip((tx + ty) * 0.5, 0.0, 1.0) * rect_mask
    _grad_shape("gradient_rect4", rect_mask, grad)

    # Truncate to requested number
    hard = hard[:num_types]
    soft = soft[:num_types]
    names = names[:num_types]

    return {
        "hard": torch.stack(hard).to(device),
        "soft": torch.stack(soft).to(device),
        "type_map": {i: i for i in range(len(names))},
        "names": names,
    }


def save_template_library(library: dict, path: Path) -> None:
    """Save template library to disk."""
    torch.save(library, path)
    print(f"Saved template library to {path}")


def load_template_library(path: Path, device: str = "cpu") -> dict:
    """Load template library from disk."""
    lib = torch.load(path, map_location=device, weights_only=False)
    return lib
