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

    # Add batch/channel dims to templates for grid_sample
    # grid_sample expects [N, C, H, W] templates
    hard = hard_templates.unsqueeze(1)  # [num_types, 1, T, T]
    soft = soft_templates.unsqueeze(1)  # [num_types, 1, T, T]

    # Initialize canvas with background (in linear space)
    bg_linear = srgb_to_linear(background)
    C = bg_linear.view(3, 1, 1).expand(3, canvas_height, canvas_width).clone()
    T = torch.ones(canvas_height, canvas_width, device=device)  # transmittance

    # Render shapes back-to-front (index 0 = back, N-1 = front)
    for i in range(N):
        # Compute template-space coordinates
        grid = compute_template_coords(
            px_grid, py_grid,
            cx[i], cy[i], rx[i], ry[i], angle[i],
        )  # [1, H, W, 2]

        tidx = type_indices[i].item()

        # Sample hard template (forward)
        hard_alpha = F.grid_sample(
            hard[tidx:tidx + 1], grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        ).squeeze(0).squeeze(0)  # [H, W]

        # Sample soft template (for backward gradient)
        soft_alpha = F.grid_sample(
            soft[tidx:tidx + 1], grid,
            mode="bilinear", padding_mode="zeros", align_corners=True,
        ).squeeze(0).squeeze(0)  # [H, W]

        # Threshold hard alpha (binary) for exact FH6 rendering
        hard_alpha = (hard_alpha > 0.5).float()

        # STE trick: forward=hard, backward=soft
        # alpha = hard.detach() + soft - soft.detach()
        # → value = hard, gradient = d(soft)/d(params)
        alpha = hard_alpha.detach() + soft_alpha - soft_alpha.detach()
        alpha = alpha * opacity[i]  # [H, W]
        alpha = torch.clamp(alpha, 0.0, 1.0)

        # Over composite in LINEAR space (physically correct)
        w = alpha * T  # [H, W]
        color_linear = srgb_to_linear(colors[i]).view(3, 1, 1)  # [3, 1, 1]
        C = C + w.unsqueeze(0) * color_linear
        T = T * (1.0 - alpha)

        # Early termination: stop if transmittance is negligible
        if T.max() < 1e-4:
            break

    # Convert back to sRGB for output
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

    # Initialize output canvas
    bg_linear = srgb_to_linear(background)
    C_out = bg_linear.view(3, 1, 1).expand(3, H, W).clone()
    T_out = torch.ones(H, W, device=device)

    # Compute tile grid
    num_tiles_y = (H + tile_size - 1) // tile_size
    num_tiles_x = (W + tile_size - 1) // tile_size

    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            # Tile boundaries
            t_y0 = ty * tile_size
            t_y1 = min(t_y0 + tile_size, H)
            t_x0 = tx * tile_size
            t_x1 = min(t_x0 + tile_size, W)
            th = t_y1 - t_y0
            tw = t_x1 - t_x0

            # Find shapes overlapping this tile
            overlaps = (
                (x1 > t_x0) & (x0 < t_x1) &
                (y1 > t_y0) & (y0 < t_y1)
            )
            active_indices = torch.where(overlaps)[0]

            if len(active_indices) == 0:
                continue

            # Create tile-local grids
            px_tile, py_tile = _make_canvas_grid(th, tw, device)
            # Offset tile coords to global canvas coords
            px_tile = px_tile + t_x0
            py_tile = py_tile + t_y0

            # Tile-local buffers
            C_tile = bg_linear.view(3, 1, 1).expand(3, th, tw).clone()
            T_tile = torch.ones(th, tw, device=device)

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

            # Write tile back to output
            C_out[:, t_y0:t_y1, t_x0:t_x1] = C_tile
            T_out[t_y0:t_y1, t_x0:t_x1] = T_tile

    result = linear_to_srgb(C_out)
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

    def forward(self, use_tiling: bool = None) -> torch.Tensor:
        """
        Render current shapes to canvas.

        On GPU, uses torch.compile (Triton backend) or Triton STE kernel
        for acceleration. Falls back to tiled/non-tiled PyTorch otherwise.

        Args:
            use_tiling: if None, auto-selects tiled rendering when
                        canvas >= TILE_THRESHOLD (256px) on either axis.
        """
        # On GPU: try torch.compile (uses Triton backend automatically)
        if self.device == "cuda":
            try:
                if not hasattr(self, "_compiled_forward"):
                    self._compiled_forward = torch.compile(
                        self._pytorch_forward, mode="default"
                    )
                return self._compiled_forward(use_tiling=use_tiling)
            except Exception:
                pass  # Fall through

        return self._pytorch_forward(use_tiling=use_tiling)

    def _pytorch_forward(self, use_tiling: bool = None) -> torch.Tensor:
        """Pure PyTorch forward path (tiled or non-tiled)."""
        if use_tiling is None:
            use_tiling = (
                self.canvas_height >= TILE_THRESHOLD
                or self.canvas_width >= TILE_THRESHOLD
            )

        if use_tiling:
            return over_composite_render_tiled(
                hard_templates=self.hard_templates,
                soft_templates=self.soft_templates,
                type_indices=self.type_indices,
                cx=self.cx, cy=self.cy, rx=self.rx, ry=self.ry, angle=self.angle,
                colors=self.colors, opacity=self.opacity,
                canvas_height=self.canvas_height, canvas_width=self.canvas_width,
                background=self.background, device=self.device,
            )
        else:
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
