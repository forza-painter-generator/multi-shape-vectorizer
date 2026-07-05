"""
STE (Straight-Through Estimator) Differentiable Over-Compositing Renderer.

Core rendering pipeline:
  1. For each shape (back-to-front z-order):
     a. Transform pixel coords → template-local coords
     b. Sample hard template → binary alpha (forward)
     c. Sample soft template → continuous alpha (backward gradient source)
     d. STE: alpha = hard.detach() + soft - soft.detach()
     e. Over composite in LINEAR space: C_lin += T * alpha * srgb_to_linear(color)
     f. Final output: linear_to_srgb(C_lin)

The STE trick allows:
  - Forward: exact FH6 hard-edge rendering (binary alpha)
  - Backward: gradients flow through the soft (blurred) template

Color pipeline (physically correct):
  sRGB input → Linear blend → sRGB output

Reference:
  - diffbmp: Gaussian blur for soft rasterization (CVPR 2026)
  - vinylizer: Over compositing + STE + sRGB↔Linear (src/cuda/color_utils.cuh)
  - IGS: Chunked processing pattern (for future Triton port)
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .loss import srgb_to_linear, linear_to_srgb


# Template fill ratio: shape edge maps to this coordinate in [-1,1] grid_sample space
# The shape occupies ~90% of the template texture
TEMPLATE_FILL_RATIO = 0.9

# Tile size for tile-based rendering (px)
DEFAULT_TILE_SIZE = 128

# Minimum canvas dimension to trigger tiled rendering
TILE_THRESHOLD = 256


def _make_canvas_grid(
    height: int, width: int, device: str = "cpu"
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Create pixel coordinate grids for the canvas.
    Returns (px_grid, py_grid) each of shape [H, W].
    Pixel centers are at integer coordinates (0-indexed).
    """
    py = torch.arange(height, dtype=torch.float32, device=device)
    px = torch.arange(width, dtype=torch.float32, device=device)
    py_grid, px_grid = torch.meshgrid(py, px, indexing="ij")
    return px_grid, py_grid


def compute_template_coords(
    px_grid: torch.Tensor,
    py_grid: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    rx: torch.Tensor,
    ry: torch.Tensor,
    angle_deg: torch.Tensor,
) -> torch.Tensor:
    """
    Compute grid_sample coordinates for one shape.

    Transforms canvas pixel coords → template coords in [-1, 1]:
      1. Translate: dx = px - cx, dy = py - cy
      2. Rotate by -angle
      3. Scale: divide by rx, ry and multiply by TEMPLATE_FILL_RATIO

    Args:
        px_grid: [H, W] pixel x coordinates
        py_grid: [H, W] pixel y coordinates
        cx, cy: [1] shape center (scalar tensors)
        rx, ry: [1] shape scale radii (scalar tensors)
        angle_deg: [1] rotation in degrees (scalar tensor)

    Returns:
        grid: [1, H, W, 2] in [-1, 1] range for F.grid_sample
    """
    angle_rad = torch.deg2rad(angle_deg)
    cos_a = torch.cos(-angle_rad)
    sin_a = torch.sin(-angle_rad)

    dx = px_grid - cx
    dy = py_grid - cy

    # Rotate
    dx_rot = dx * cos_a - dy * sin_a
    dy_rot = dx * sin_a + dy * cos_a

    # Scale to template space [-1, 1]
    # rx/ry in canvas pixels → template units
    tx = dx_rot / (rx + 1e-8) * TEMPLATE_FILL_RATIO
    ty = dy_rot / (ry + 1e-8) * TEMPLATE_FILL_RATIO

    # grid_sample expects (x, y) order = (tx, ty) for the last dim
    grid = torch.stack([tx, ty], dim=-1)  # [H, W, 2]
    return grid.unsqueeze(0)  # [1, H, W, 2]


def over_composite_render(
    hard_templates: torch.Tensor,
    soft_templates: torch.Tensor,
    type_indices: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    rx: torch.Tensor,
    ry: torch.Tensor,
    angle: torch.Tensor,
    colors: torch.Tensor,
    opacity: torch.Tensor,
    canvas_height: int,
    canvas_width: int,
    background: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Differentiable Over-compositing renderer with STE.

    Args:
        hard_templates: [num_types, T, T] binary templates
        soft_templates: [num_types, T, T] blurred templates
        type_indices: [N] long tensor, template index for each shape
        cx, cy: [N] center positions (canvas pixels)
        rx, ry: [N] scale radii (canvas pixels)
        angle: [N] rotation in degrees
        colors: [N, 3] RGB colors in [0, 1]
        opacity: [N] opacity in [0, 1]
        canvas_height, canvas_width: output size
        background: [3] RGB background color
        device: torch device

    Returns:
        rendered: [3, H, W] rendered image in [0, 1]
    """
    N = cx.shape[0]
    px_grid, py_grid = _make_canvas_grid(canvas_height, canvas_width, device)

    hard = hard_templates.unsqueeze(1)  # [num_types, 1, T, T]
    soft = soft_templates.unsqueeze(1)  # [num_types, 1, T, T]

    # Initialize canvas with background (in linear space)
    bg_linear = srgb_to_linear(background)
    C = bg_linear.view(3, 1, 1).expand(3, canvas_height, canvas_width).clone()
    T = torch.ones(canvas_height, canvas_width, device=device)

    DEG2RAD = 3.141592653589793 / 180.0

    for i in range(N):
        # Inline coordinate transform (avoid allocating per-shape grid tensor)
        shape_ang = angle[i] * DEG2RAD
        cos_a = torch.cos(-shape_ang)
        sin_a = torch.sin(-shape_ang)
        shape_rx = rx[i] + 1e-8
        shape_ry = ry[i] + 1e-8

        dx = px_grid - cx[i]
        dy = py_grid - cy[i]
        dx_rot = dx * cos_a - dy * sin_a
        dy_rot = dx * sin_a + dy * cos_a
        tx = dx_rot / shape_rx * TEMPLATE_FILL_RATIO
        ty = dy_rot / shape_ry * TEMPLATE_FILL_RATIO
        grid = torch.stack([tx, ty], dim=-1).unsqueeze(0)  # [1, H, W, 2]

        tidx = type_indices[i].item()

        # Continuous soft alpha directly (no STE, like vinylizer alpha_228)
        alpha = F.grid_sample(
            soft[tidx:tidx + 1], grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        ).squeeze(0).squeeze(0) * opacity[i]  # [H, W]
        alpha = torch.clamp(alpha, 0.0, 1.0)

        # Over composite in LINEAR space
        w = alpha * T  # [H, W]
        color_linear = srgb_to_linear(colors[i]).view(3, 1, 1)  # [3, 1, 1]
        C = C + w.unsqueeze(0) * color_linear
        T = T * (1.0 - alpha)

        if T.max() < 1e-4:
            break

    result = linear_to_srgb(C)
    result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
    return result.clamp(0.0, 1.0)


# ============================================================
# Tile-based rendering (performance optimization)
# ============================================================

def _compute_shape_aabb(
    cx: torch.Tensor,
    cy: torch.Tensor,
    rx: torch.Tensor,
    ry: torch.Tensor,
    angle_deg: torch.Tensor,
    blur_pad: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute conservative AABB for a rotated shape.

    Handles rotation by computing the maximum extent in x and y
    after applying the rotation to the bounding ellipse.

    Args:
        cx, cy: [N] shape centers
        rx, ry: [N] scale radii in canvas pixels
        angle_deg: [N] rotation in degrees
        blur_pad: extra padding for soft template blur (pixels)

    Returns:
        (x0, y0, x1, y1): [N] each, AABB corner coordinates
    """
    angle_rad = torch.deg2rad(angle_deg)
    cos_a = torch.abs(torch.cos(angle_rad))
    sin_a = torch.abs(torch.sin(angle_rad))

    # Maximum extent in x and y after rotation
    # The shape boundary in template space is at ±TEMPLATE_FILL_RATIO
    # In canvas pixels: extent_x = rx, extent_y = ry (by definition)
    # After rotation: max_x = |rx*cos| + |ry*sin|, max_y = |rx*sin| + |ry*cos|
    half_w = rx * cos_a + ry * sin_a + blur_pad
    half_h = rx * sin_a + ry * cos_a + blur_pad

    x0 = cx - half_w
    y0 = cy - half_h
    x1 = cx + half_w
    y1 = cy + half_h

    return x0, y0, x1, y1


def over_composite_render_tiled(
    hard_templates: torch.Tensor,
    soft_templates: torch.Tensor,
    type_indices: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    rx: torch.Tensor,
    ry: torch.Tensor,
    angle: torch.Tensor,
    colors: torch.Tensor,
    opacity: torch.Tensor,
    canvas_height: int,
    canvas_width: int,
    background: torch.Tensor,
    device: str = "cpu",
    tile_size: int = DEFAULT_TILE_SIZE,
) -> torch.Tensor:
    """
    Tile-based differentiable Over-compositing renderer with STE.

    Splits canvas into tiles, computes per-shape AABB, and only renders
    shapes that overlap each tile. This dramatically reduces computation
    for large canvases where most shapes only cover a small area.

    Args:
        (same as over_composite_render)
        tile_size: tile dimension in pixels (default 128)

    Returns:
        rendered: [3, H, W] rendered image in [0, 1]
    """
    N = cx.shape[0]
    H, W = canvas_height, canvas_width

    # Compute AABBs for all shapes
    x0, y0, x1, y1 = _compute_shape_aabb(cx, cy, rx, ry, angle)

    # Add batch/channel dims to templates
    hard = hard_templates.unsqueeze(1)  # [num_types, 1, T, T]
    soft = soft_templates.unsqueeze(1)

    # Collect tile contributions as full-canvas tensors, then sum.
    # This avoids in-place assignment on leaf tensors that breaks autograd.
    bg_linear = srgb_to_linear(background)
    all_tiles = []

    # Compute tile grid
    num_tiles_y = (H + tile_size - 1) // tile_size
    num_tiles_x = (W + tile_size - 1) // tile_size

    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            t_y0 = ty * tile_size
            t_y1 = min(t_y0 + tile_size, H)
            t_x0 = tx * tile_size
            t_x1 = min(t_x0 + tile_size, W)
            th = t_y1 - t_y0
            tw = t_x1 - t_x0

            overlaps = (
                (x1 > t_x0) & (x0 < t_x1) &
                (y1 > t_y0) & (y0 < t_y1)
            )
            active_indices = torch.where(overlaps)[0]

            # Build tile result
            C_tile = bg_linear.view(3, 1, 1).expand(3, th, tw).clone()
            T_tile = torch.ones(th, tw, device=device)

            if len(active_indices) > 0:
                px_tile, py_tile = _make_canvas_grid(th, tw, device)
                px_tile = px_tile + t_x0
                py_tile = py_tile + t_y0

                for idx in active_indices:
                    i = idx.item()
                    grid = compute_template_coords(
                        px_tile, py_tile,
                        cx[i], cy[i], rx[i], ry[i], angle[i],
                    )
                    tidx = type_indices[i].item()

                    hard_alpha = F.grid_sample(
                        hard[tidx:tidx + 1], grid,
                        mode="bilinear", padding_mode="zeros", align_corners=True,
                    ).squeeze(0).squeeze(0)
                    soft_alpha = F.grid_sample(
                        soft[tidx:tidx + 1], grid,
                        mode="bilinear", padding_mode="zeros", align_corners=True,
                    ).squeeze(0).squeeze(0)

                    hard_alpha = (hard_alpha > 0.5).float()
                    alpha = hard_alpha.detach() + soft_alpha - soft_alpha.detach()
                    alpha = alpha * opacity[i]
                    alpha = torch.clamp(alpha, 0.0, 1.0)

                    w = alpha * T_tile
                    color_lin = srgb_to_linear(colors[i]).view(3, 1, 1)
                    C_tile = C_tile + w.unsqueeze(0) * color_lin
                    T_tile = T_tile * (1.0 - alpha)

                    if T_tile.max() < 1e-4:
                        break

            # Place tile into full-canvas contribution tensor
            # Using a fresh zeros tensor + __setitem__ means the backward
            # correctly flows from C_out -> full -> C_tile via the + operator.
            full = torch.zeros(3, H, W, device=device)
            full[:, t_y0:t_y1, t_x0:t_x1] = C_tile
            all_tiles.append(full)

    # Sum all tile contributions (non-in-place, autograd-safe)
    if len(all_tiles) == 0:
        C_out = bg_linear.view(3, 1, 1).expand(3, H, W).clone()
    elif len(all_tiles) == 1:
        C_out = all_tiles[0]
    else:
        C_out = torch.stack(all_tiles, dim=0).sum(dim=0)

def over_composite_render_aabb(
    hard_templates: torch.Tensor,
    soft_templates: torch.Tensor,
    type_indices: torch.Tensor,
    cx: torch.Tensor,
    cy: torch.Tensor,
    rx: torch.Tensor,
    ry: torch.Tensor,
    angle: torch.Tensor,
    colors: torch.Tensor,
    opacity: torch.Tensor,
    canvas_height: int,
    canvas_width: int,
    background: torch.Tensor,
    device: str = "cpu",
) -> torch.Tensor:
    """
    AABB-clipped differentiable Over-compositing renderer with STE.

    For each shape, grid_sample is computed only within the shape's
    AABB region, then padded to full canvas using F.pad (autograd-safe).
    Much faster than full-canvas grid_sample for shapes that cover small areas.
    """
    N = cx.shape[0]
    H, W = canvas_height, canvas_width

    x0, y0, x1, y1 = _compute_shape_aabb(cx, cy, rx, ry, angle)

    hard = hard_templates.unsqueeze(1)
    soft = soft_templates.unsqueeze(1)

    bg_linear = srgb_to_linear(background)
    C = bg_linear.view(3, 1, 1).expand(3, H, W).clone()
    T_full = torch.ones(H, W, device=device)

    for i in range(N):
        # Clamp AABB to valid range (NaN protection)
        _x0 = float(x0[i].item()) if not torch.isnan(x0[i]) else 0.0
        _y0 = float(y0[i].item()) if not torch.isnan(y0[i]) else 0.0
        _x1 = float(x1[i].item()) if not torch.isnan(x1[i]) else 1.0
        _y1 = float(y1[i].item()) if not torch.isnan(y1[i]) else 1.0

        ax0 = max(0, int(_x0))
        ay0 = max(0, int(_y0))
        ax1 = min(W, int(_x1) + 1)
        ay1 = min(H, int(_y1) + 1)

        if ax1 <= ax0 or ay1 <= ay0:
            continue

        aH, aW = ay1 - ay0, ax1 - ax0
        px_aabb, py_aabb = _make_canvas_grid(aH, aW, device)
        px_aabb = px_aabb + ax0
        py_aabb = py_aabb + ay0

        grid = compute_template_coords(
            px_aabb, py_aabb,
            cx[i], cy[i], rx[i], ry[i], angle[i],
        )

        tidx = type_indices[i].item()
        hard_alpha = F.grid_sample(
            hard[tidx:tidx + 1], grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        ).squeeze(0).squeeze(0)
        soft_alpha = F.grid_sample(
            soft[tidx:tidx + 1], grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        ).squeeze(0).squeeze(0)

        hard_alpha = (hard_alpha > 0.5).float()
        alpha = hard_alpha.detach() + soft_alpha - soft_alpha.detach()
        alpha = alpha * opacity[i]
        alpha = torch.clamp(alpha, 0.0, 1.0)

        T_local = T_full[ay0:ay1, ax0:ax1]
        w = alpha * T_local
        color_lin = srgb_to_linear(colors[i]).view(3, 1, 1)
        contrib = w.unsqueeze(0) * color_lin  # [3, aH, aW]

        # Pad to full canvas (autograd-safe, no in-place slice assignment)
        contrib_padded = F.pad(contrib, (ax0, W - ax1, ay0, H - ay1))
        C = C + contrib_padded

        T_local_new = T_local * (1.0 - alpha)
        T_full = T_full.clone()
        T_full[ay0:ay1, ax0:ax1] = T_local_new

    result = linear_to_srgb(C)
    result = torch.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
    return result.clamp(0.0, 1.0)


class STEVectorRenderer(nn.Module):
    """
    PyTorch Module wrapper around the STE Over-compositing renderer.

    Manages optimizable parameters and template library.
    """

    def __init__(
        self,
        num_shapes: int,
        num_types: int,
        hard_templates: torch.Tensor,
        soft_templates: torch.Tensor,
        canvas_height: int = 512,
        canvas_width: int = 512,
        background: tuple = (0.0, 0.0, 0.0),
        device: str = "cpu",
    ):
        """
        Args:
            num_shapes: N, number of shapes to optimize
            num_types: number of template types available
            hard_templates: [num_types, T, T]
            soft_templates: [num_types, T, T]
            canvas_height, canvas_width: output resolution
            background: RGB background color
            device: torch device
        """
        super().__init__()
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        self.device = device

        self.register_buffer("hard_templates", hard_templates)
        self.register_buffer("soft_templates", soft_templates)
        self.register_buffer(
            "background",
            torch.tensor(background, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "type_indices",
            torch.zeros(num_shapes, dtype=torch.long, device=device),
        )

        # Optimizable continuous parameters
        self.cx = nn.Parameter(torch.rand(num_shapes, device=device) * canvas_width)
        self.cy = nn.Parameter(torch.rand(num_shapes, device=device) * canvas_height)
        self.rx = nn.Parameter(torch.rand(num_shapes, device=device) * 40 + 10)
        self.ry = nn.Parameter(torch.rand(num_shapes, device=device) * 40 + 10)
        self.angle = nn.Parameter(torch.rand(num_shapes, device=device) * 360.0)
        self.colors = nn.Parameter(torch.rand(num_shapes, 3, device=device))
        self.opacity = nn.Parameter(torch.ones(num_shapes, device=device))

    def forward(self, return_alpha: bool = False) -> torch.Tensor:
        """
        Render current shapes to canvas.
        GPU: Triton tile-based kernel. CPU: PyTorch fallback.

        Returns: [3, H, W] or [4, H, W] rendered image in sRGB [0, 1].
        If return_alpha=True, 4th channel is total opacity (1 - transmittance).
        """
        if self.device == "cuda":
            result = self._triton_forward()
        else:
            result = self._pytorch_forward()
        if return_alpha:
            alpha = self._render_alpha_map()
            result = torch.cat([result, alpha.unsqueeze(0)], dim=0)
        return result

    def _render_alpha_map(self) -> torch.Tensor:
        """Render alpha channel: render all shapes in white on black background."""
        with torch.no_grad():
            if self.device == "cuda":
                from .triton_kernels_v2 import TritonV2Soft
                from .loss import srgb_to_linear
                white_colors = torch.ones_like(self.colors)
                black_bg = torch.zeros(3, device=self.device)
                bg_lin = srgb_to_linear(black_bg)
                alpha_lin = TritonV2Soft.apply(
                    self.hard_templates, self.soft_templates, self.type_indices,
                    self.cx, self.cy, self.rx, self.ry, self.angle,
                    white_colors, self.opacity,
                    self.canvas_height, self.canvas_width, bg_lin,
                )
                # alpha = luminance of rendering white shapes on black
                from .loss import linear_to_srgb
                alpha = linear_to_srgb(alpha_lin.permute(2, 0, 1))
                return alpha.mean(dim=0).clamp(0.0, 1.0)
            else:
                return over_composite_render(
                    hard_templates=self.hard_templates,
                    soft_templates=self.soft_templates,
                    type_indices=self.type_indices,
                    cx=self.cx, cy=self.cy, rx=self.rx, ry=self.ry, angle=self.angle,
                    colors=torch.ones_like(self.colors),
                    opacity=self.opacity,
                    canvas_height=self.canvas_height, canvas_width=self.canvas_width,
                    background=torch.zeros(3, device=self.device),
                    device=self.device,
                ).mean(dim=0)

    def _triton_forward(self) -> torch.Tensor:
        """Triton tile-based forward + backward (via autograd Function)."""
        from .triton_kernels_v2 import TritonV2STE
        from .loss import srgb_to_linear, linear_to_srgb
        bg_linear = srgb_to_linear(self.background)
        # Triton operates in linear space, returns [H, W, 3]
        rendered_lin = TritonV2STE.apply(
            self.hard_templates, self.soft_templates, self.type_indices,
            self.cx, self.cy, self.rx, self.ry, self.angle,
            self.colors, self.opacity,
            self.canvas_height, self.canvas_width, bg_linear,
        )
        # Convert to sRGB [3, H, W]
        result = linear_to_srgb(rendered_lin.permute(2, 0, 1))
        return result.clamp(0.0, 1.0)

    def _pytorch_forward(self, use_tiling: bool = None) -> torch.Tensor:
        """Pure PyTorch forward path — non-tiled, simplest and most reliable."""
        return over_composite_render(
            hard_templates=self.hard_templates,
            soft_templates=self.soft_templates,
            type_indices=self.type_indices,
            cx=self.cx, cy=self.cy, rx=self.rx, ry=self.ry, angle=self.angle,
            colors=self.colors, opacity=self.opacity,
            canvas_height=self.canvas_height, canvas_width=self.canvas_width,
            background=self.background, device=self.device,
        )

    def clamp_params(self):
        """Clamp parameters to valid ranges (called after optimizer step)."""
        with torch.no_grad():
            self.cx.data.clamp_(0, self.canvas_width)
            self.cy.data.clamp_(0, self.canvas_height)
            self.rx.data.clamp_(2, self.canvas_width)
            self.ry.data.clamp_(2, self.canvas_height)
            self.angle.data.clamp_(0, 360)
            self.colors.data.clamp_(0, 1)
            self.opacity.data.clamp_(0.01, 1.0)

    def get_params_dict(self) -> dict:
        """Return all parameters as a dict for external use."""
        return {
            "cx": self.cx,
            "cy": self.cy,
            "rx": self.rx,
            "ry": self.ry,
            "angle": self.angle,
            "colors": self.colors,
            "opacity": self.opacity,
            "type_indices": self.type_indices,
        }
