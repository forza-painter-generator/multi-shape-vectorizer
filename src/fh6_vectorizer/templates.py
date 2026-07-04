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
    num_types: int = 16,
    template_size: int = TEMPLATE_SIZE,
    sigma: float = DEFAULT_SIGMA,
    device: str = "cpu",
) -> dict:
    """
    Generate synthetic geometric templates (no FH6 data dependency).
    Useful for testing without FH6 resources.

    Produces up to 16 shapes:
      0: circle, 1: square, 2: triangle, 3: ellipse, 4: diamond,
      5: star5, 6: cross, 7: ring, 8: pentagon, 9: hexagon,
      10: crescent, 11: heart, 12: arrow_right, 13: droplet,
      14: chevron, 15: star4
    """
    hard = []
    soft = []
    names = []
    s = template_size
    center = s // 2

    def _make_shape(name: str, canvas: np.ndarray) -> None:
        names.append(name)
        t = torch.from_numpy(canvas.astype(np.float32))
        hard.append(t)
        soft.append(gaussian_blur(t, sigma))

    # 0: Filled circle
    canvas = np.zeros((s, s), dtype=np.float32)
    cv2.circle(canvas, (center, center), int(s * 0.45), 1.0, -1)
    _make_shape("circle", canvas)

    # 1: Filled square
    canvas = np.zeros((s, s), dtype=np.float32)
    margin = int(s * 0.1)
    cv2.rectangle(canvas, (margin, margin), (s - margin, s - margin), 1.0, -1)
    _make_shape("square", canvas)

    # 2: Filled triangle
    canvas = np.zeros((s, s), dtype=np.float32)
    pts = np.array([[center, s * 0.1], [s * 0.1, s * 0.9], [s * 0.9, s * 0.9]], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], 1.0)
    _make_shape("triangle", canvas)

    # 3: Filled ellipse
    canvas = np.zeros((s, s), dtype=np.float32)
    cv2.ellipse(canvas, (center, center), (int(s * 0.4), int(s * 0.25)), 0, 0, 360, 1.0, -1)
    _make_shape("ellipse", canvas)

    # 4: Filled diamond
    canvas = np.zeros((s, s), dtype=np.float32)
    pts = np.array([[center, s * 0.05], [s * 0.95, center], [center, s * 0.95], [s * 0.05, center]], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], 1.0)
    _make_shape("diamond", canvas)

    # 5: Star (5-point)
    canvas = np.zeros((s, s), dtype=np.float32)
    outer_r = int(s * 0.45)
    inner_r = int(s * 0.2)
    star_pts = []
    for i in range(10):
        r = outer_r if i % 2 == 0 else inner_r
        angle = math.pi / 2 + i * math.pi / 5
        star_pts.append([int(center + r * math.cos(angle)), int(center - r * math.sin(angle))])
    cv2.fillPoly(canvas, [np.array(star_pts, dtype=np.int32)], 1.0)
    _make_shape("star", canvas)

    # 6: Cross
    canvas = np.zeros((s, s), dtype=np.float32)
    arm_w = int(s * 0.2)
    cv2.rectangle(canvas, (center - arm_w, 0), (center + arm_w, s), 1.0, -1)
    cv2.rectangle(canvas, (0, center - arm_w), (s, center + arm_w), 1.0, -1)
    _make_shape("cross", canvas)

    # 7: Ring (hollow circle)
    canvas = np.zeros((s, s), dtype=np.float32)
    cv2.circle(canvas, (center, center), int(s * 0.45), 1.0, -1)
    cv2.circle(canvas, (center, center), int(s * 0.25), 0.0, -1)
    _make_shape("ring", canvas)

    # 8: Pentagon (regular 5-sided)
    canvas = np.zeros((s, s), dtype=np.float32)
    r = int(s * 0.45)
    pent_pts = []
    for i in range(5):
        ang = -math.pi / 2 + i * 2 * math.pi / 5
        pent_pts.append([int(center + r * math.cos(ang)), int(center + r * math.sin(ang))])
    cv2.fillPoly(canvas, [np.array(pent_pts, dtype=np.int32)], 1.0)
    _make_shape("pentagon", canvas)

    # 9: Hexagon (regular 6-sided)
    canvas = np.zeros((s, s), dtype=np.float32)
    r = int(s * 0.45)
    hex_pts = []
    for i in range(6):
        ang = i * 2 * math.pi / 6
        hex_pts.append([int(center + r * math.cos(ang)), int(center + r * math.sin(ang))])
    cv2.fillPoly(canvas, [np.array(hex_pts, dtype=np.int32)], 1.0)
    _make_shape("hexagon", canvas)

    # 10: Crescent (moon shape — difference of two offset circles)
    canvas = np.zeros((s, s), dtype=np.float32)
    cx1, cy1 = int(s * 0.42), center
    cx2, cy2 = int(s * 0.58), int(s * 0.55)
    r_moon = int(s * 0.42)
    cv2.circle(canvas, (cx1, cy1), r_moon, 1.0, -1)
    cv2.circle(canvas, (cx2, cy2), r_moon, 0.0, -1)
    _make_shape("crescent", canvas)

    # 11: Heart
    canvas = np.zeros((s, s), dtype=np.float32)
    # Two circles top + triangle bottom
    r_h = int(s * 0.22)
    cv2.circle(canvas, (int(s * 0.33), int(s * 0.32)), r_h, 1.0, -1)
    cv2.circle(canvas, (int(s * 0.67), int(s * 0.32)), r_h, 1.0, -1)
    heart_pts = np.array([
        [s * 0.10, s * 0.40], [s * 0.90, s * 0.40],
        [center, s * 0.92],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [heart_pts], 1.0)
    _make_shape("heart", canvas)

    # 12: Arrow (right-pointing)
    canvas = np.zeros((s, s), dtype=np.float32)
    arrow_pts = np.array([
        [s * 0.05, s * 0.35], [s * 0.55, s * 0.35],
        [s * 0.55, s * 0.10], [s * 0.95, center],
        [s * 0.55, s * 0.90], [s * 0.55, s * 0.65],
        [s * 0.05, s * 0.65],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [arrow_pts], 1.0)
    _make_shape("arrow", canvas)

    # 13: Droplet / teardrop
    canvas = np.zeros((s, s), dtype=np.float32)
    drop_pts = np.array([
        [center, s * 0.05],
        [int(s * 0.85), int(s * 0.55)],
        [int(s * 0.68), s - 10],
        [center, int(s * 0.88)],
        [int(s * 0.32), s - 10],
        [int(s * 0.15), int(s * 0.55)],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [drop_pts], 1.0)
    _make_shape("droplet", canvas)

    # 14: Chevron (>> shape)
    canvas = np.zeros((s, s), dtype=np.float32)
    chev_pts = np.array([
        [s * 0.05, s * 0.20], [s * 0.50, center],
        [s * 0.05, s * 0.80], [s * 0.30, s * 0.80],
        [s * 0.65, center], [s * 0.30, s * 0.20],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [chev_pts], 1.0)
    _make_shape("chevron", canvas)

    # 15: Star (4-point / sparkle)
    canvas = np.zeros((s, s), dtype=np.float32)
    star4_pts = np.array([
        [center, s * 0.05], [int(s * 0.58), int(s * 0.38)],
        [s * 0.95, center], [int(s * 0.58), int(s * 0.62)],
        [center, s * 0.95], [int(s * 0.42), int(s * 0.62)],
        [s * 0.05, center], [int(s * 0.42), int(s * 0.38)],
    ], dtype=np.int32)
    cv2.fillPoly(canvas, [star4_pts], 1.0)
    _make_shape("star4", canvas)

    # 16: Gradient ellipse (radial gradient, continuous falloff)
    # Key difference from regular ellipse (#3):
    #   - Hard: threshold at 0.5 (binary, same STE forward as others)
    #   - Soft: raw gradient values [0,1] (gradient over ENTIRE area, not just edges!)
    # This gives much better gradient signal during optimization.
    y_idx, x_idx = np.ogrid[:s, :s]
    rx_px = s * 0.45
    ry_px = s * 0.45
    dist = np.sqrt(((x_idx - center) / rx_px) ** 2 + ((y_idx - center) / ry_px) ** 2)
    grad = np.clip(1.0 - dist, 0.0, 1.0)
    names.append("gradient_ellipse")
    hard.append(torch.from_numpy((grad >= 0.5).astype(np.float32)))
    soft.append(torch.from_numpy(grad.astype(np.float32)))

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
