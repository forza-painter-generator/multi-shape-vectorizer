"""
No-STE Differentiable Over-Compositing Renderer.

Core rendering pipeline:
  1. For each shape (back-to-front z-order):
     a. Transform pixel coords → template-local coords
     b. Sample template → continuous alpha (soft gradient)
     c. Over composite in LINEAR space: C += T * alpha * srgb_to_linear(color)
     d. Final output: linear_to_srgb(C)

Continuous alpha in both forward and backward — no straight-through estimator.

Color pipeline (physically correct):
  sRGB input → Linear blend → sRGB output

Reference:
  - diffbmp: Gaussian blur for soft rasterization (CVPR 2026)
  - vinylizer: Over compositing + sRGB↔Linear (src/cuda/color_utils.cuh)
"""

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from .loss import srgb_to_linear, linear_to_srgb


# Template fill ratio: shape edge maps to this coordinate in [-1,1] grid_sample space
# The shape occupies ~90% of the template texture
TEMPLATE_FILL_RATIO = 0.9


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
                from .triton_kernels_v2 import TritonV2STE
                from .loss import srgb_to_linear
                white_colors = torch.ones_like(self.colors)
                black_bg = torch.zeros(3, device=self.device)
                bg_lin = srgb_to_linear(black_bg)
                alpha_lin = TritonV2STE.apply(
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
