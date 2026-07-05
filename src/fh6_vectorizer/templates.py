"""
Template Generator: Render FH6 polygon shapes to bitmap templates.

Produces:
  - hard templates: binary {0,1} bitmaps (match in-game rendering)
  - soft templates: Gaussian-blurred continuous [0,1] bitmaps (for gradient flow)

Uses the STE (Straight-Through Estimator) strategy:
  - Forward pass: sample from hard templates
  - Backward pass: sample from soft templates → gradients flow
"""

import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# --- Config ---
TEMPLATE_SIZE = 128  # px, template texture resolution
DEFAULT_SIGMA = 2.0  # Wider soft edge for better gradient reach (was 0.5)
FH6_COORD_RANGE = 128.0  # FH6 vertices are in [-128, 128] range

# Family name → type_code base (from forza-painter-fh6)
VINYL_TYPE_BASES = {
    "Primitives": 1048677,
    "Gradient_Shapes": 1048777,
    "Stripes": 1048877,
    "Tears": 1048977,
    "Racing_Icons": 1049077,
    "Flames": 1049177,
    "Paint_Splats": 1049277,
    "Tribal": 1049377,
    "Nature": 1049477,
    "Community_Vinyls_1": 1050677,
    "Community_Vinyls_2": 1050777,
    "Community_Vinyls_3": 1050877,
    "Community_Vinyls_4": 1050977,
}

# Synthetic gradient shape indices → FH6 type_code mapping
# These 8 mathematical gradient shapes approximate FH6 Gradient_Shapes.
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

# Recommended subset of shapes for the PoC (from the implementation plan)
RECOMMENDED_SHAPES = [
    ("Primitives", 2),   # Circle (index 2 = circle)
    ("Primitives", 3),   # Square/Rectangle
    ("Primitives", 4),   # Triangle
    ("Primitives", 16),  # Ellipse (rx ≠ ry capable)
    ("Primitives", 6),   # Another polygon variant
    ("Primitives", 7),   # Star-like
    ("Primitives", 8),   # Diamond
    ("Stripes", 1),      # Stripes pattern
]


def render_fh6_shape(
    vertices: list[dict],
    indices: list[int],
    size: int = TEMPLATE_SIZE,
) -> np.ndarray:
    """
    Render an FH6 polygon shape to a bitmap.

    FH6 vertices are in [-128, 128] coordinate space.
    We normalize to [0, size] for the template texture.

    Args:
        vertices: list of {"X": float, "Y": float}
        indices: flat list of triangle vertex indices
        size: output texture size in pixels

    Returns:
        np.ndarray of shape (size, size), dtype float32, values in [0, 1]
    """
    pts = np.array([[v["X"], v["Y"]] for v in vertices], dtype=np.float32)

    # Normalize: FH6 [-128, 128] → [0, size]
    pts[:, 0] = (pts[:, 0] / FH6_COORD_RANGE + 1.0) * 0.5 * size
    pts[:, 1] = (pts[:, 1] / FH6_COORD_RANGE + 1.0) * 0.5 * size

    triangles = np.array(indices, dtype=np.int32).reshape(-1, 3)
    canvas = np.zeros((size, size), dtype=np.float32)
    cv2.fillPoly(canvas, [pts[tri].astype(np.int32)], 1.0)

    return np.clip(canvas, 0.0, 1.0)


def gaussian_blur(tensor: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply Gaussian blur to a 2D tensor using OpenCV."""
    arr = tensor.cpu().numpy()
    if sigma > 0:
        arr = cv2.GaussianBlur(arr, (0, 0), sigmaX=sigma)
    return torch.from_numpy(arr)


def load_shape_json(json_path: Path) -> dict:
    """Load an FH6 shape JSON file."""
    with open(json_path, "r") as f:
        return json.load(f)


def _load_gradient_shape_png(
    family_dir: Path,
    index: int,
    template_size: int = TEMPLATE_SIZE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load FH6 Gradient Shape from its pre-rendered PNG preview.

    FH6 Gradient_Shapes are rendered by the game engine with directional
    gradient alpha (center→edge, corner→corner, etc.). The .png preview
    already contains this baked-in gradient, so we can use it directly
    as the soft template — no Gaussian blur needed.

    Args:
        family_dir: path to Gradient_Shapes/ directory
        index: shape index within the family (e.g., 28)
        template_size: output texture resolution

    Returns:
        (hard_template, soft_template):
          - hard: binary {0, 1} — thresholded PNG for FH6 match
          - soft: continuous [0, 1] — raw PNG gradient
    """
    png_path = family_dir / f"{index}.png"
    if not png_path.exists():
        raise FileNotFoundError(f"Gradient Shape PNG not found: {png_path}")

    png = cv2.imread(str(png_path), cv2.IMREAD_UNCHANGED)
    if png is None:
        raise RuntimeError(f"Failed to load PNG: {png_path}")

    # Extract alpha channel (4th channel in RGBA)
    if len(png.shape) == 3 and png.shape[2] == 4:
        alpha_ch = png[:, :, 3]  # [0, 255] alpha
    elif len(png.shape) == 2:
        alpha_ch = png  # grayscale, use directly
    else:
        raise RuntimeError(f"Unexpected PNG shape: {png.shape} for {png_path}")

    if alpha_ch.shape[0] != template_size or alpha_ch.shape[1] != template_size:
        alpha_ch = cv2.resize(alpha_ch, (template_size, template_size),
                              interpolation=cv2.INTER_LANCZOS4)

    soft = alpha_ch.astype(np.float32) / 255.0
    # Normalize to [0, 1] — FH6 PNG alpha may not reach 1.0 or 0.0
    a_min, a_max = soft.min(), soft.max()
    if a_max > a_min + 0.01:
        soft = (soft - a_min) / (a_max - a_min)
    # Hard = threshold at 0.5 for binary FH6 game rendering match
    hard = (soft >= 0.5).astype(np.float32)

    return hard, soft


def build_template_library(
    vinyls_root: Path,
    shape_list: Optional[list[tuple[str, int]]] = None,
    template_size: int = TEMPLATE_SIZE,
    sigma: float = DEFAULT_SIGMA,
    device: str = "cpu",
) -> dict:
    """
    Build a library of hard + soft templates from FH6 vinyl resources.

    Args:
        vinyls_root: path to Vinyls/ directory
        shape_list: list of (family_name, index) to include.
                    If None, uses RECOMMENDED_SHAPES.
        template_size: texture resolution
        sigma: Gaussian blur sigma for soft templates
        device: torch device

    Returns:
        dict with:
            - "hard": torch.Tensor [num_types, template_size, template_size]
            - "soft": torch.Tensor [num_types, template_size, template_size]
            - "type_map": dict {type_code: template_index}
            - "names": list of (family, index) strings
    """
    if shape_list is None:
        shape_list = RECOMMENDED_SHAPES

    hard_templates = []
    soft_templates = []
    type_map = {}
    names = []

    for idx, (family, shape_index) in enumerate(shape_list):
        shape_dir = vinyls_root / family / str(shape_index)
        if not shape_dir.exists():
            print(f"  [WARN] Shape not found: {family}/{shape_index}, skipping")
            continue

        data = load_shape_json(shape_dir)

        # Render hard template
        hard = render_fh6_shape(
            data["Vertices"], data["Indices"], template_size
        )
        hard_tensor = torch.from_numpy(hard)

        # Render soft template (Gaussian blurred)
        soft_tensor = gaussian_blur(hard_tensor, sigma)

        hard_templates.append(hard_tensor)
        soft_templates.append(soft_tensor)

        type_code = data["Info"]["Type"]
        type_map[type_code] = idx
        names.append(f"{family}/{shape_index}")

    return {
        "hard": torch.stack(hard_templates).to(device) if hard_templates else torch.empty(0),
        "soft": torch.stack(soft_templates).to(device) if soft_templates else torch.empty(0),
        "type_map": type_map,
        "names": names,
    }


def generate_synthetic_templates(
    num_types: int = 5,
    template_size: int = TEMPLATE_SIZE,
    sigma: float = DEFAULT_SIGMA,
    device: str = "cpu",
) -> dict:
    """
    Generate 5 mathematical gradient templates (no FH6 data dependency).

    All use continuous gradient — no Gaussian blur needed.
    Previously included 16 geometric shapes removed: Gaussian blur edge-only
    gradient (~40% coverage) vs mathematical gradients (61-67% coverage).
    """
    hard = []
    soft = []
    names = []
    s = template_size
    center = s // 2

    # === 5 Mathematical Gradient Shapes ===
    y_idx, x_idx = np.mgrid[:s, :s]

    # === 5 Mathematical Gradient Shapes ===
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

    # 1: Gradient Rect 1 (center→right edge, FH6 #11)
    rect_mask = np.zeros((s, s), dtype=np.float32)
    m = int(s * 0.1); cv2.rectangle(rect_mask, (m, m), (s - m, s - m), 1.0, -1)
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

    # --- Removed shapes ---
    # 16 geometric shapes (circle, square, etc.): Gaussian blur edge-only gradient
    #   → limited gradient (~40% coverage) vs mathematical gradients (61-67%)
    # gradient_ellipse2 (#17): center 50% solid → only 15% non-zero gradient area
    # gradient_rect5 (#22): center line → left & right, redundant with rect1+rect3
    # gradient_triangle1 (#23): tip→base, only 33% gradient in triangle area

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
