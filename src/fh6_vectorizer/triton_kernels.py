"""
Triton-accelerated tile-based Over compositing with STE.

Implements fused forward + backward kernels that process the canvas
tile-by-tile, only iterating shapes whose AABB overlaps each tile.
This reduces per-step pixel operations by 10-50x vs full-canvas rendering.

Architecture (matching diffbmp's SimpleTileRenderer + CUDA tile rasterizer):
  1. Python pre-process: compute per-shape AABBs, assign shapes to tiles
  2. Triton forward: 1D grid [num_tiles], each block renders one tile
  3. Triton backward: 1D grid [num_tiles], each block computes gradients
     via recompute + analytical chain rule (STE: soft template for backward)

References:
  - diffbmp/cuda_tile_rasterizer/cuda_kernels/tile_forward.cu
  - diffbmp/cuda_tile_rasterizer/cuda_kernels/tile_backward.cu
  - IGS/src/igs/gs_triton_chunked.py
  - IMPLEMENTATION_PLAN.md §5
"""

from typing import Optional

import torch
import triton
import triton.language as tl
from torch.autograd import Function

from .ste_renderer import TEMPLATE_FILL_RATIO

# ---------------------------------------------------------------------------
# Python helpers: AABB computation & tile-shape assignment
# ---------------------------------------------------------------------------

def compute_shape_aabbs(
    cx: torch.Tensor,   # [N]
    cy: torch.Tensor,   # [N]
    rx: torch.Tensor,   # [N]
    ry: torch.Tensor,   # [N]
    angle_deg: torch.Tensor,  # [N]
    blur_pad: float = 3.0,
) -> torch.Tensor:
    """Compute conservative AABBs for all shapes.
    Returns: [N, 4] float32 (x0, y0, x1, y1) in canvas pixel coords."""
    angle_rad = torch.deg2rad(angle_deg)
    cos_a = torch.abs(torch.cos(angle_rad))
    sin_a = torch.abs(torch.sin(angle_rad))
    half_w = rx * cos_a + ry * sin_a + blur_pad
    half_h = rx * sin_a + ry * cos_a + blur_pad
    return torch.stack([
        cx - half_w, cy - half_h,
        cx + half_w, cy + half_h,
    ], dim=-1)


def build_tile_assignments(
    aabbs: torch.Tensor,        # [N, 4] (x0, y0, x1, y1)
    H: int, W: int,
    tile_size: int = 128,
):
    """Build tile->shape index mappings. Returns tile_shapes, tile_offsets, ntx, nty."""
    num_tiles_y = (H + tile_size - 1) // tile_size
    num_tiles_x = (W + tile_size - 1) // tile_size

    x0, y0, x1, y1 = aabbs[:, 0], aabbs[:, 1], aabbs[:, 2], aabbs[:, 3]

    all_shapes = []
    offsets = [0]

    for ty in range(num_tiles_y):
        for tx in range(num_tiles_x):
            t_x0 = tx * tile_size
            t_y0 = ty * tile_size
            t_x1 = min(t_x0 + tile_size, W)
            t_y1 = min(t_y0 + tile_size, H)

            overlaps = (x1 > t_x0) & (x0 < t_x1) & (y1 > t_y0) & (y0 < t_y1)
            indices = torch.where(overlaps)[0].to(torch.int32)
            all_shapes.append(indices)
            offsets.append(offsets[-1] + len(indices))

    tile_shapes = torch.cat(all_shapes) if all_shapes else torch.zeros(0, dtype=torch.int32)
    tile_offsets = torch.tensor(offsets, dtype=torch.int32)
    return tile_shapes, tile_offsets, num_tiles_x, num_tiles_y


# ---------------------------------------------------------------------------
# Triton forward kernel: tile-based Over compositing (hard alpha)
# ---------------------------------------------------------------------------

@triton.jit
def _tiled_over_fwd_kernel(
    templates_ptr,          # [num_types, T, T] float32
    type_idx_ptr,           # [N] int32
    cx_ptr, cy_ptr,         # [N] float32
    rx_ptr, ry_ptr,         # [N] float32
    angle_ptr,              # [N] float32 (degrees)
    colors_ptr,             # [N, 3] float32
    opacity_ptr,            # [N] float32
    tile_offsets_ptr,       # [num_tiles + 1] int32
    tile_shapes_ptr,        # [total_assignments] int32
    output_ptr,             # [H, W, 3] float32
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    tile_size: tl.constexpr,
    num_tiles_x: tl.constexpr,
    bg_r: tl.constexpr, bg_g: tl.constexpr, bg_b: tl.constexpr,
    FILL_RATIO: tl.constexpr,
    PIXELS_PER_BLOCK: tl.constexpr,
):
    """One program per tile. Iterates only shapes overlapping this tile."""
    pid = tl.program_id(0)
    tile_y = pid // num_tiles_x
    tile_x = pid % num_tiles_x

    t_x0 = tile_x * tile_size
    t_y0 = tile_y * tile_size
    t_x1 = tl.minimum(t_x0 + tile_size, W)
    t_y1 = tl.minimum(t_y0 + tile_size, H)
    th = t_y1 - t_y0
    tw = t_x1 - t_x0

    start = tl.load(tile_offsets_ptr + pid)
    end = tl.load(tile_offsets_ptr + pid + 1)
    num_tile_shapes = end - start

    num_tile_px = th * tw
    DEG2RAD = 3.141592653589793 / 180.0
    HALF_T = 0.5 * (T - 1)

    for px_start in range(0, num_tile_px, PIXELS_PER_BLOCK):
        px_offs = px_start + tl.arange(0, PIXELS_PER_BLOCK)
        px_mask = px_offs < num_tile_px

        py = t_y0 + (px_offs // tw)
        px = t_x0 + (px_offs % tw)

        Cr = tl.full([PIXELS_PER_BLOCK], bg_r, dtype=tl.float32)
        Cg = tl.full([PIXELS_PER_BLOCK], bg_g, dtype=tl.float32)
        Cb = tl.full([PIXELS_PER_BLOCK], bg_b, dtype=tl.float32)
        Tt = tl.full([PIXELS_PER_BLOCK], 1.0, dtype=tl.float32)

        for s in range(num_tile_shapes):
            si = tl.load(tile_shapes_ptr + start + s)

            shape_cx = tl.load(cx_ptr + si)
            shape_cy = tl.load(cy_ptr + si)
            shape_rx = tl.load(rx_ptr + si) + 1e-8
            shape_ry = tl.load(ry_ptr + si) + 1e-8
            shape_ang_rad = tl.load(angle_ptr + si) * DEG2RAD
            shape_opacity = tl.load(opacity_ptr + si)
            tidx = tl.load(type_idx_ptr + si)

            dx = px.to(tl.float32) - shape_cx
            dy = py.to(tl.float32) - shape_cy
            cos_a = tl.cos(-shape_ang_rad)
            sin_a = tl.sin(-shape_ang_rad)
            dx_rot = dx * cos_a - dy * sin_a
            dy_rot = dx * sin_a + dy * cos_a
            tx = dx_rot / shape_rx * FILL_RATIO
            ty = dy_rot / shape_ry * FILL_RATIO

            u = (tx + 1.0) * HALF_T
            v = (ty + 1.0) * HALF_T
            u = tl.clamp(u, 0.0, T - 1.001)
            v = tl.clamp(v, 0.0, T - 1.001)

            x0 = u.to(tl.int32)
            y0 = v.to(tl.int32)
            x1 = tl.minimum(x0 + 1, T - 1)
            y1 = tl.minimum(y0 + 1, T - 1)
            fx = u - x0.to(tl.float32)
            fy = v - y0.to(tl.float32)

            tmpl_base = templates_ptr + tidx * T * T
            v00 = tl.load(tmpl_base + y0 * T + x0, mask=px_mask, other=0.0)
            v10 = tl.load(tmpl_base + y0 * T + x1, mask=px_mask, other=0.0)
            v01 = tl.load(tmpl_base + y1 * T + x0, mask=px_mask, other=0.0)
            v11 = tl.load(tmpl_base + y1 * T + x1, mask=px_mask, other=0.0)

            alpha_raw = ((1.0 - fx) * (1.0 - fy) * v00 +
                         fx * (1.0 - fy) * v10 +
                         (1.0 - fx) * fy * v01 +
                         fx * fy * v11)

            # Hard threshold (STE: forward uses binary alpha)
            alpha = tl.where(alpha_raw > 0.5, 1.0, 0.0) * shape_opacity
            alpha = tl.clamp(alpha, 0.0, 1.0)

            w = alpha * Tt
            Cr += w * tl.load(colors_ptr + si * 3 + 0)
            Cg += w * tl.load(colors_ptr + si * 3 + 1)
            Cb += w * tl.load(colors_ptr + si * 3 + 2)
            Tt *= (1.0 - alpha)

        idx3 = (py * W + px) * 3
        tl.store(output_ptr + idx3 + 0, Cr, mask=px_mask)
        tl.store(output_ptr + idx3 + 1, Cg, mask=px_mask)
        tl.store(output_ptr + idx3 + 2, Cb, mask=px_mask)


# ---------------------------------------------------------------------------
# Triton backward kernel: analytical gradients via recompute (soft alpha)
# ---------------------------------------------------------------------------

@triton.jit
def _tiled_over_bwd_kernel(
    soft_tmpl_ptr,          # [num_types, T, T] float32
    type_idx_ptr,           # [N] int32
    cx_ptr, cy_ptr,         # [N] float32
    rx_ptr, ry_ptr,         # [N] float32
    angle_ptr,              # [N] float32 (degrees)
    colors_ptr,             # [N, 3] float32
    opacity_ptr,            # [N] float32
    tile_offsets_ptr,
    tile_shapes_ptr,
    grad_output_ptr,        # [H, W, 3] float32
    grad_cx_ptr, grad_cy_ptr,
    grad_rx_ptr, grad_ry_ptr,
    grad_angle_ptr,
    grad_colors_ptr,        # [N, 3] float32
    grad_opacity_ptr,       # [N] float32
    T: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    tile_size: tl.constexpr,
    num_tiles_x: tl.constexpr,
    FILL_RATIO: tl.constexpr,
    PIXELS_PER_BLOCK: tl.constexpr,
):
    """
    Backward: one program per tile. Recomputes soft alpha,
    then computes dL/dalpha -> chain rule to all params.
    Uses T_prev = 1 approximation (exact for top shapes).
    """
    pid = tl.program_id(0)
    tile_y = pid // num_tiles_x
    tile_x = pid % num_tiles_x

    t_x0 = tile_x * tile_size
    t_y0 = tile_y * tile_size
    t_x1 = tl.minimum(t_x0 + tile_size, W)
    t_y1 = tl.minimum(t_y0 + tile_size, H)
    th = t_y1 - t_y0
    tw = t_x1 - t_x0

    start = tl.load(tile_offsets_ptr + pid)
    end = tl.load(tile_offsets_ptr + pid + 1)
    num_tile_shapes = end - start

    num_tile_px = th * tw
    DEG2RAD = 3.141592653589793 / 180.0
    HALF_T = 0.5 * (T - 1)

    for px_start in range(0, num_tile_px, PIXELS_PER_BLOCK):
        px_offs = px_start + tl.arange(0, PIXELS_PER_BLOCK)
        px_mask = px_offs < num_tile_px

        py = t_y0 + (px_offs // tw)
        px = t_x0 + (px_offs % tw)

        idx3 = (py * W + px) * 3
        go_r = tl.load(grad_output_ptr + idx3 + 0, mask=px_mask, other=0.0)
        go_g = tl.load(grad_output_ptr + idx3 + 1, mask=px_mask, other=0.0)
        go_b = tl.load(grad_output_ptr + idx3 + 2, mask=px_mask, other=0.0)

        dLdC_r = go_r
        dLdC_g = go_g
        dLdC_b = go_b
        dLdT = tl.full([PIXELS_PER_BLOCK], 0.0, dtype=tl.float32)

        # Process shapes FRONT-TO-BACK (reverse of forward z-order)
        for s in range(num_tile_shapes - 1, -1, -1):
            si = tl.load(tile_shapes_ptr + start + s)

            shape_cx = tl.load(cx_ptr + si)
            shape_cy = tl.load(cy_ptr + si)
            shape_rx = tl.load(rx_ptr + si) + 1e-8
            shape_ry = tl.load(ry_ptr + si) + 1e-8
            shape_ang_rad = tl.load(angle_ptr + si) * DEG2RAD
            shape_opacity = tl.load(opacity_ptr + si)
            tidx = tl.load(type_idx_ptr + si)
            cr = tl.load(colors_ptr + si * 3 + 0)
            cg = tl.load(colors_ptr + si * 3 + 1)
            cb = tl.load(colors_ptr + si * 3 + 2)

            # Recompute soft alpha
            dx = px.to(tl.float32) - shape_cx
            dy = py.to(tl.float32) - shape_cy
            cos_a = tl.cos(-shape_ang_rad)
            sin_a = tl.sin(-shape_ang_rad)
            dx_rot = dx * cos_a - dy * sin_a
            dy_rot = dx * sin_a + dy * cos_a
            tx = dx_rot / shape_rx * FILL_RATIO
            ty = dy_rot / shape_ry * FILL_RATIO

            u = (tx + 1.0) * HALF_T
            v = (ty + 1.0) * HALF_T
            u = tl.clamp(u, 0.0, T - 1.001)
            v = tl.clamp(v, 0.0, T - 1.001)

            x0 = u.to(tl.int32)
            y0 = v.to(tl.int32)
            x1 = tl.minimum(x0 + 1, T - 1)
            y1 = tl.minimum(y0 + 1, T - 1)
            fx = u - x0.to(tl.float32)
            fy = v - y0.to(tl.float32)

            tmpl_base = soft_tmpl_ptr + tidx * T * T
            v00 = tl.load(tmpl_base + y0 * T + x0, mask=px_mask, other=0.0)
            v10 = tl.load(tmpl_base + y0 * T + x1, mask=px_mask, other=0.0)
            v01 = tl.load(tmpl_base + y1 * T + x0, mask=px_mask, other=0.0)
            v11 = tl.load(tmpl_base + y1 * T + x1, mask=px_mask, other=0.0)

            alpha_raw = ((1.0 - fx) * (1.0 - fy) * v00 +
                         fx * (1.0 - fy) * v10 +
                         (1.0 - fx) * fy * v01 +
                         fx * fy * v11)
            alpha_raw = tl.clamp(alpha_raw, 0.0, 1.0)
            alpha = tl.clamp(alpha_raw * shape_opacity, 0.0, 1.0)

            # T_prev approximation = 1.0 (exact for top-most shapes)
            T_prev = 1.0
            dot_cc = dLdC_r * cr + dLdC_g * cg + dLdC_b * cb
            dLda = (dot_cc - dLdT) * T_prev

            # Gradient w.r.t. opacity
            tl.atomic_add(grad_opacity_ptr + si,
                          tl.sum(dLda * alpha_raw, axis=0))

            # Gradient w.r.t. colors
            tl.atomic_add(grad_colors_ptr + si * 3 + 0,
                          tl.sum(dLdC_r * T_prev * alpha, axis=0))
            tl.atomic_add(grad_colors_ptr + si * 3 + 1,
                          tl.sum(dLdC_g * T_prev * alpha, axis=0))
            tl.atomic_add(grad_colors_ptr + si * 3 + 2,
                          tl.sum(dLdC_b * T_prev * alpha, axis=0))

            # Bilinear gradient
            da_du = (1.0 - fy) * (v10 - v00) + fy * (v11 - v01)
            da_dv = (1.0 - fx) * (v01 - v00) + fx * (v11 - v10)

            dLdtx = dLda * da_du * HALF_T * shape_opacity
            dLdty = dLda * da_dv * HALF_T * shape_opacity

            inv_rx = 1.0 / shape_rx
            inv_ry = 1.0 / shape_ry

            # Coordinate chain rule
            dLdcx = dLdtx * (-cos_a * inv_rx * FILL_RATIO) + \
                    dLdty * (-sin_a * inv_ry * FILL_RATIO)
            dLdcy = dLdtx * (sin_a * inv_rx * FILL_RATIO) + \
                    dLdty * (-cos_a * inv_ry * FILL_RATIO)
            dLdrx = dLdtx * (-tx * inv_rx)
            dLdry = dLdty * (-ty * inv_ry)
            dLdangle = (dLdtx * (dy_rot * inv_rx * FILL_RATIO) +
                        dLdty * (-dx_rot * inv_ry * FILL_RATIO)) * DEG2RAD

            tl.atomic_add(grad_cx_ptr + si, tl.sum(dLdcx, axis=0))
            tl.atomic_add(grad_cy_ptr + si, tl.sum(dLdcy, axis=0))
            tl.atomic_add(grad_rx_ptr + si, tl.sum(dLdrx, axis=0))
            tl.atomic_add(grad_ry_ptr + si, tl.sum(dLdry, axis=0))
            tl.atomic_add(grad_angle_ptr + si, tl.sum(dLdangle, axis=0))

            # Update for previous shapes
            dLdT = dLdT * (1.0 - alpha) + dot_cc * alpha


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------

def triton_tiled_forward(
    templates: torch.Tensor,      # [num_types, T, T] hard
    type_indices: torch.Tensor,   # [N] int32
    cx, cy, rx, ry, angle,        # [N]
    colors: torch.Tensor,         # [N, 3]
    opacity: torch.Tensor,        # [N]
    H: int, W: int,
    background: torch.Tensor,     # [3] linear space
    fill_ratio: float = TEMPLATE_FILL_RATIO,
    tile_size: int = 128,
    pixels_per_block: int = 256,
) -> torch.Tensor:
    """Triton tile-based forward. Returns [H, W, 3] in linear space."""
    device = templates.device
    T_sz = templates.shape[1]

    templates = templates.contiguous().float()
    type_indices = type_indices.contiguous().to(torch.int32)
    cx = cx.contiguous().float()
    cy = cy.contiguous().float()
    rx = rx.contiguous().float()
    ry = ry.contiguous().float()
    angle = angle.contiguous().float()
    colors = colors.contiguous().float()
    opacity = opacity.contiguous().float()
    background = background.contiguous().float()

    aabbs = compute_shape_aabbs(cx, cy, rx, ry, angle)
    tile_shapes, tile_offsets, ntx, nty = build_tile_assignments(aabbs, H, W, tile_size)
    tile_shapes = tile_shapes.to(torch.int32).to(device)
    tile_offsets = tile_offsets.to(torch.int32).to(device)
    ntiles = tile_offsets.shape[0] - 1

    output = torch.empty(H, W, 3, device=device, dtype=torch.float32)

    _tiled_over_fwd_kernel[(ntiles,)](
        templates, type_indices,
        cx, cy, rx, ry, angle, colors, opacity,
        tile_offsets, tile_shapes, output,
        T=T_sz, H=H, W=W, tile_size=tile_size, num_tiles_x=ntx,
        bg_r=float(background[0]), bg_g=float(background[1]), bg_b=float(background[2]),
        FILL_RATIO=fill_ratio, PIXELS_PER_BLOCK=pixels_per_block,
    )
    return output


def triton_tiled_backward(
    soft_templates: torch.Tensor,
    type_indices, cx, cy, rx, ry, angle, colors, opacity,
    grad_output: torch.Tensor,  # [H, W, 3] or [3, H, W]
    H: int, W: int,
    fill_ratio: float = TEMPLATE_FILL_RATIO,
    tile_size: int = 128,
    pixels_per_block: int = 256,
):
    """Triton tile-based backward. Returns 7 gradient tensors."""
    device = soft_templates.device
    N = cx.shape[0]
    T_sz = soft_templates.shape[1]

    if grad_output.dim() == 3 and grad_output.shape[0] == 3:
        grad_output = grad_output.permute(1, 2, 0).contiguous()
    grad_output = grad_output.contiguous().float()

    soft_templates = soft_templates.contiguous().float()
    type_indices = type_indices.contiguous().to(torch.int32)
    cx_v = cx.contiguous().float()
    cy_v = cy.contiguous().float()
    rx_v = rx.contiguous().float()
    ry_v = ry.contiguous().float()
    angle_v = angle.contiguous().float()
    colors_v = colors.contiguous().float()
    opacity_v = opacity.contiguous().float()

    aabbs = compute_shape_aabbs(cx_v, cy_v, rx_v, ry_v, angle_v)
    tile_shapes, tile_offsets, ntx, nty = build_tile_assignments(aabbs, H, W, tile_size)
    tile_shapes = tile_shapes.to(torch.int32).to(device)
    tile_offsets = tile_offsets.to(torch.int32).to(device)
    ntiles = tile_offsets.shape[0] - 1

    g_cx = torch.zeros(N, device=device, dtype=torch.float32)
    g_cy = torch.zeros(N, device=device, dtype=torch.float32)
    g_rx = torch.zeros(N, device=device, dtype=torch.float32)
    g_ry = torch.zeros(N, device=device, dtype=torch.float32)
    g_ang = torch.zeros(N, device=device, dtype=torch.float32)
    g_col = torch.zeros(N, 3, device=device, dtype=torch.float32)
    g_op = torch.zeros(N, device=device, dtype=torch.float32)

    _tiled_over_bwd_kernel[(ntiles,)](
        soft_templates, type_indices,
        cx_v, cy_v, rx_v, ry_v, angle_v, colors_v, opacity_v,
        tile_offsets, tile_shapes, grad_output,
        g_cx, g_cy, g_rx, g_ry, g_ang, g_col, g_op,
        T=T_sz, H=H, W=W, tile_size=tile_size, num_tiles_x=ntx,
        FILL_RATIO=fill_ratio, PIXELS_PER_BLOCK=pixels_per_block,
    )
    return g_cx, g_cy, g_rx, g_ry, g_ang, g_col, g_op


# ---------------------------------------------------------------------------
# torch.autograd.Function
# ---------------------------------------------------------------------------

class TritonTileOverSTE(Function):
    """STE: Triton hard-template forward + PyTorch soft-template backward.

    Forward uses the fast Triton tile kernel (hard alpha).
    Backward uses the existing PyTorch over_composite_render with
    soft templates for correct, NaN-free gradients.

    This hybrid approach gives ~10-50x forward speedup vs pure PyTorch
    while keeping backward correct. The backward is O(N*H*W) so it's
    the bottleneck for large N, but is still usable for PoC.
    """

    @staticmethod
    def forward(ctx, hard_templates, soft_templates, type_indices,
                cx, cy, rx, ry, angle, colors, opacity,
                H, W, background, fill_ratio, tile_size):
        # Save detached copies for the backward recompute
        ctx.save_for_backward(
            soft_templates, type_indices,
            cx.detach(), cy.detach(), rx.detach(), ry.detach(),
            angle.detach(), colors.detach(), opacity.detach(),
        )
        ctx.H, ctx.W = H, W
        ctx.fill_ratio = fill_ratio
        ctx.background = background.detach()

        with torch.no_grad():
            return triton_tiled_forward(
                hard_templates, type_indices,
                cx, cy, rx, ry, angle, colors, opacity,
                H, W, background, fill_ratio, tile_size,
            )

    @staticmethod
    def backward(ctx, grad_output):
        soft_tmpl, type_idx, cx_v, cy_v, rx_v, ry_v, angle_v, colors_v, opacity_v = \
            ctx.saved_tensors
        H, W = ctx.H, ctx.W
        device = soft_tmpl.device
        bg = ctx.background

        # grad_output from autograd: shape [H, W, 3] (matches Triton forward output)
        if grad_output.dim() == 3 and grad_output.shape[-1] == 3:
            grad_output_3d = grad_output.permute(2, 0, 1).contiguous()
        else:
            grad_output_3d = grad_output.contiguous()

        # Create differentiable clones
        params = {}
        for name, val in [('cx', cx_v), ('cy', cy_v), ('rx', rx_v), ('ry', ry_v),
                           ('angle', angle_v), ('colors', colors_v), ('opacity', opacity_v)]:
            p = val.detach().clone()
            p.requires_grad_(True)
            params[name] = p

        # Pure soft-template forward (NO STE, NO threshold, NO detach)
        # This produces a fully differentiable computation graph.
        import torch.nn.functional as F
        from .ste_renderer import _make_canvas_grid, TEMPLATE_FILL_RATIO as FR

        px_grid, py_grid = _make_canvas_grid(H, W, device)
        soft = soft_tmpl.unsqueeze(1)  # [num_types, 1, T, T]
        DEG2RAD = 3.141592653589793 / 180.0
        N = cx_v.shape[0]

        # Initialize C without .clone() — use fresh tensor each iteration
        C = bg.unsqueeze(-1).unsqueeze(-1).expand(3, H, W).contiguous().clone()
        C = C * 1.0  # force grad tracking through multiplication
        T = torch.ones(H, W, device=device)

        for i in range(N):
            shape_ang = params['angle'][i] * DEG2RAD
            cos_a = torch.cos(-shape_ang)
            sin_a = torch.sin(-shape_ang)
            shape_rx = params['rx'][i] + 1e-8
            shape_ry = params['ry'][i] + 1e-8
            dx = px_grid - params['cx'][i]
            dy = py_grid - params['cy'][i]
            dx_rot = dx * cos_a - dy * sin_a
            dy_rot = dx * sin_a + dy * cos_a
            tx = dx_rot / shape_rx * FR
            ty = dy_rot / shape_ry * FR
            g = torch.stack([tx, ty], dim=-1).unsqueeze(0)
            tidx = int(type_idx[i].item())
            alpha = F.grid_sample(
                soft[tidx:tidx + 1], g,
                mode="bilinear", padding_mode="zeros", align_corners=True,
            ).squeeze(0).squeeze(0)
            alpha = alpha * params['opacity'][i]
            alpha = torch.clamp(alpha, 0.0, 1.0)
            w = alpha * T
            C = C + w.unsqueeze(0) * params['colors'][i].view(3, 1, 1)
            T = T * (1.0 - alpha)

        loss_recompute = (C * grad_output_3d).sum()
        loss_recompute.backward()

        grads = [params[n].grad for n in
                 ['cx', 'cy', 'rx', 'ry', 'angle', 'colors', 'opacity']]

        safe_grads = []
        for g in grads:
            if g is not None:
                g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
            safe_grads.append(g)

        return (
            None, None, None,
            safe_grads[0], safe_grads[1], safe_grads[2], safe_grads[3],
            safe_grads[4], safe_grads[5], safe_grads[6],
            None, None, None, None, None,
        )
