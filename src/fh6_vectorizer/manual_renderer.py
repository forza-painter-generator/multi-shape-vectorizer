"""
Manual forward + backward Over compositing renderer (pure PyTorch, no autograd).

This is the reference implementation used to:
  1. Verify numerical correctness against PyTorch autograd
  2. Serve as the golden reference for tiled & Triton ports

Architecture:
  - forward():  hard-template Over composite in linear space → sRGB output
  - backward(): analytical gradients via recompute + chain rule (soft template)

The chain rule is derived from the Over compositing formula:
  C_{k+1} = C_k + T_k * α_{k+1} * color_{k+1}
  T_{k+1} = T_k * (1 - α_{k+1})

Backward (front-to-back):
  dL/dα_k = T_{k-1} * (dLdC · color_k - dLdT_k)
  dL/dC_{k-1} = dLdC_k
  dL/dT_{k-1} = dLdT_k * (1 - α_k) + dLdC_k · color_k * α_k
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.autograd import Function

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TEMPLATE_FILL_RATIO = 0.9
DEG2RAD = math.pi / 180.0

# sRGB ↔ Linear constants
SRGB_A = 0.055
SRGB_GAMMA = 2.4
SRGB_THRESHOLD = 0.04045
LINEAR_THRESHOLD = 0.0031308


# ---------------------------------------------------------------------------
# Color conversion (standalone, differentiable-safe)
# ---------------------------------------------------------------------------

def _srgb_to_linear(srgb: torch.Tensor) -> torch.Tensor:
    """sRGB [0,1] → linear [0,1]. Safe for autograd."""
    srgb = srgb.clamp(min=0.0, max=1.0)
    low = srgb <= SRGB_THRESHOLD
    return torch.where(
        low,
        srgb / 12.92,
        torch.pow((srgb + SRGB_A) / (1.0 + SRGB_A), SRGB_GAMMA),
    )


def _linear_to_srgb(linear: torch.Tensor) -> torch.Tensor:
    """Linear [0,1] → sRGB [0,1]. Safe for autograd (no NaN pow)."""
    linear = linear.clamp(min=1e-8, max=1.0)
    low = linear <= LINEAR_THRESHOLD
    return torch.where(
        low,
        linear * 12.92,
        (1.0 + SRGB_A) * torch.pow(linear, 1.0 / SRGB_GAMMA) - SRGB_A,
    )


def _linear_to_srgb_grad(linear: torch.Tensor) -> torch.Tensor:
    """Manual gradient of linear_to_srgb: d(srgb)/d(linear)."""
    linear = linear.clamp(min=1e-8, max=1.0)
    low = linear <= LINEAR_THRESHOLD
    return torch.where(
        low,
        torch.full_like(linear, 12.92),
        (1.0 + SRGB_A) / SRGB_GAMMA * torch.pow(linear, 1.0 / SRGB_GAMMA - 1.0),
    )


def _srgb_to_linear_grad(srgb: torch.Tensor) -> torch.Tensor:
    """Manual gradient of srgb_to_linear: d(linear)/d(srgb)."""
    srgb = srgb.clamp(min=0.0, max=1.0)
    low = srgb <= SRGB_THRESHOLD
    return torch.where(
        low,
        torch.full_like(srgb, 1.0 / 12.92),
        SRGB_GAMMA * torch.pow((srgb + SRGB_A) / (1.0 + SRGB_A), SRGB_GAMMA - 1.0)
            / (1.0 + SRGB_A),
    )


# ---------------------------------------------------------------------------
# Canvas grid
# ---------------------------------------------------------------------------

def _make_grid(H: int, W: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (px_grid, py_grid) each [H, W], pixel-center coords."""
    py = torch.arange(H, dtype=torch.float32, device=device)
    px = torch.arange(W, dtype=torch.float32, device=device)
    py_grid, px_grid = torch.meshgrid(py, px, indexing="ij")
    return px_grid, py_grid


# ---------------------------------------------------------------------------
# Bilinear sample (manual, no autograd needed for template since it's fixed)
# ---------------------------------------------------------------------------

def _bilinear_sample(
    template: torch.Tensor,   # [T, T] single template
    tx: torch.Tensor,          # [H, W] x-coord in [-1, 1]
    ty: torch.Tensor,          # [H, W] y-coord in [-1, 1]
) -> torch.Tensor:
    """
    Bilinear sample a single template at coords (tx, ty).
    Returns alpha [H, W] and also (du, dv) for gradient propagation.

    Coordinates are in [-1, 1] normalized template space.
    Returns: alpha_raw [H, W], du_dtx [H, W], dv_dty [H, W]
      where du/dtx = dv/dty = (T-1)/2  (constant)
    """
    T = template.shape[0]
    half_T = 0.5 * (T - 1)

    # [-1, 1] → [0, T-1]
    u = (tx + 1.0) * half_T
    v = (ty + 1.0) * half_T
    u = u.clamp(0.0, T - 1.001)
    v = v.clamp(0.0, T - 1.001)

    x0 = u.long()
    y0 = v.long()
    x1 = (x0 + 1).clamp(max=T - 1)
    y1 = (y0 + 1).clamp(max=T - 1)
    fx = u - x0.float()
    fy = v - y0.float()

    # Gather 4 corners — template is fixed so indexing is fine
    v00 = template[y0, x0]
    v10 = template[y0, x1]
    v01 = template[y1, x0]
    v11 = template[y1, x1]

    alpha = (1.0 - fx) * (1.0 - fy) * v00 + \
            fx * (1.0 - fy) * v10 + \
            (1.0 - fx) * fy * v01 + \
            fx * fy * v11

    # Bilinear gradients (used in chain rule for dtx, dty)
    # dalpha/du = (1-fy)*(v10-v00) + fy*(v11-v01)
    # dalpha/dv = (1-fx)*(v01-v00) + fx*(v11-v10)
    du = (1.0 - fy) * (v10 - v00) + fy * (v11 - v01)
    dv = (1.0 - fx) * (v01 - v00) + fx * (v11 - v10)

    # dalpha/dtx = dalpha/du * du/dtx = dalpha/du * half_T
    # dalpha/dty = dalpha/dv * dv/dty = dalpha/dv * half_T
    dalpha_dtx = du * half_T
    dalpha_dty = dv * half_T

    return alpha, dalpha_dtx, dalpha_dty


# ---------------------------------------------------------------------------
# Coordinate transform & gradients
# ---------------------------------------------------------------------------

def _coord_transform(
    px: torch.Tensor, py: torch.Tensor,  # [H, W]
    cx: torch.Tensor, cy: torch.Tensor,   # scalars
    rx: torch.Tensor, ry: torch.Tensor,   # scalars
    angle_deg: torch.Tensor,              # scalar
) -> tuple:
    """
    Transform pixel coords → template coords (tx, ty).
    Also returns intermediate values needed for gradient chain rule.

    Returns:
        tx, ty: [H, W] coords in [-1, 1]
        cos_a, sin_a: scalars (cos(-θ), sin(-θ))
        dx_rot, dy_rot: [H, W] rotated offsets
        inv_rx, inv_ry: scalars (1/rx, 1/ry)
    """
    ang_rad = angle_deg * DEG2RAD
    cos_a = torch.cos(-ang_rad)
    sin_a = torch.sin(-ang_rad)
    inv_rx = 1.0 / (rx + 1e-8)
    inv_ry = 1.0 / (ry + 1e-8)

    dx = px - cx
    dy = py - cy
    dx_rot = dx * cos_a - dy * sin_a
    dy_rot = dx * sin_a + dy * cos_a

    tx = dx_rot * inv_rx * TEMPLATE_FILL_RATIO
    ty = dy_rot * inv_ry * TEMPLATE_FILL_RATIO

    return tx, ty, cos_a, sin_a, dx_rot, dy_rot, inv_rx, inv_ry


# ---------------------------------------------------------------------------
# Forward pass
# ---------------------------------------------------------------------------

def over_composite_forward(
    hard_templates: torch.Tensor,   # [num_types, T, T]
    type_indices: torch.Tensor,     # [N] int
    cx: torch.Tensor,               # [N]
    cy: torch.Tensor,               # [N]
    rx: torch.Tensor,               # [N]
    ry: torch.Tensor,               # [N]
    angle: torch.Tensor,            # [N] degrees
    colors: torch.Tensor,           # [N, 3] sRGB [0,1]
    opacity: torch.Tensor,          # [N] [0,1]
    H: int, W: int,
    background: torch.Tensor,       # [3] sRGB [0,1]
    device: str = "cpu",
) -> torch.Tensor:
    """
    Forward Over composite with hard template (binary alpha).

    Returns: [3, H, W] sRGB image in [0, 1].
    """
    N = cx.shape[0]
    px, py = _make_grid(H, W, device)

    # Initialize in linear space
    bg_lin = _srgb_to_linear(background)
    C = bg_lin.view(3, 1, 1).expand(3, H, W).contiguous().clone()
    T = torch.ones(H, W, device=device)

    with torch.no_grad():
        for i in range(N):
            tidx = int(type_indices[i].item())
            tx, ty, _, _, _, _, _, _ = _coord_transform(
                px, py, cx[i], cy[i], rx[i], ry[i], angle[i],
            )
            alpha_raw, _, _ = _bilinear_sample(hard_templates[tidx], tx, ty)

            # Hard threshold
            alpha = (alpha_raw > 0.5).float() * opacity[i]
            alpha = alpha.clamp(0.0, 1.0)

            # Over composite in linear space
            w = alpha * T
            col_lin = _srgb_to_linear(colors[i])
            C = C + w.unsqueeze(0) * col_lin.view(3, 1, 1)
            T = T * (1.0 - alpha)

            if T.max() < 1e-4:
                break

    return _linear_to_srgb(C)  # [3, H, W] sRGB


# ---------------------------------------------------------------------------
# Backward pass (analytical, no autograd)
# ---------------------------------------------------------------------------

def over_composite_backward(
    soft_templates: torch.Tensor,   # [num_types, T, T]
    type_indices: torch.Tensor,     # [N]
    cx: torch.Tensor,               # [N]
    cy: torch.Tensor,               # [N]
    rx: torch.Tensor,               # [N]
    ry: torch.Tensor,               # [N]
    angle: torch.Tensor,            # [N] degrees
    colors: torch.Tensor,           # [N, 3] sRGB
    opacity: torch.Tensor,          # [N]
    H: int, W: int,
    grad_output: torch.Tensor,      # [3, H, W] dL/d(srgb_output)
    device: str = "cpu",
) -> dict:
    """
    Analytical backward pass. Recomputes soft-template forward,
    then applies chain rule front-to-back.

    Returns dict of gradient tensors:
      'cx', 'cy', 'rx', 'ry', 'angle', 'colors', 'opacity'
    """
    N = cx.shape[0]
    px, py = _make_grid(H, W, device)

    # --- Recompute forward (soft template) to get intermediate values ---
    # We need: per-shape alpha, tx, ty, cos_a, sin_a, dx_rot, dy_rot, inv_rx, inv_ry
    # Store as lists

    alphas = []
    txs, tys = [], []
    cos_as, sin_as = [], []
    dx_rots, dy_rots = [], []
    inv_rxs, inv_rys = [], []
    colors_lin = []
    dalpha_dtxs, dalpha_dtys = [], []

    with torch.no_grad():
        for i in range(N):
            tidx = int(type_indices[i].item())
            tx, ty, cos_a, sin_a, dx_rot, dy_rot, inv_rx, inv_ry = _coord_transform(
                px, py, cx[i], cy[i], rx[i], ry[i], angle[i],
            )
            alpha_raw, da_dtx, da_dty = _bilinear_sample(soft_templates[tidx], tx, ty)
            alpha = alpha_raw * opacity[i]
            alpha = alpha.clamp(0.0, 1.0)

            alphas.append(alpha)
            txs.append(tx); tys.append(ty)
            cos_as.append(cos_a); sin_as.append(sin_a)
            dx_rots.append(dx_rot); dy_rots.append(dy_rot)
            inv_rxs.append(inv_rx); inv_rys.append(inv_ry)
            dalpha_dtxs.append(da_dtx); dalpha_dtys.append(da_dty)
            colors_lin.append(_srgb_to_linear(colors[i]))

    # --- Compute dL/dC (linear space) from dL/d(srgb) ---
    # We need C_linear. Recompute one more time.
    bg_lin = _srgb_to_linear(torch.zeros(3, device=device))
    C_linear = bg_lin.view(3, 1, 1).expand(3, H, W).contiguous().clone()
    T_fwd = torch.ones(H, W, device=device)

    with torch.no_grad():
        for i in range(N):
            w = alphas[i] * T_fwd
            C_linear = C_linear + w.unsqueeze(0) * colors_lin[i].view(3, 1, 1)
            T_fwd = T_fwd * (1.0 - alphas[i])

    # d(srgb)/d(linear) at the forward C_linear
    d_srgb_d_linear = _linear_to_srgb_grad(C_linear)  # [3, H, W]

    # dL/d(linear) = dL/d(srgb) * d(srgb)/d(linear)
    dLdC = grad_output * d_srgb_d_linear  # [3, H, W] — gradient w.r.t. linear C
    dLdT = torch.zeros(H, W, device=device)  # gradient w.r.t. T

    # --- Accumulate gradients (all zero-initialized) ---
    grad_cx = torch.zeros(N, device=device)
    grad_cy = torch.zeros(N, device=device)
    grad_rx = torch.zeros(N, device=device)
    grad_ry = torch.zeros(N, device=device)
    grad_angle = torch.zeros(N, device=device)
    grad_colors = torch.zeros(N, 3, device=device)
    grad_opacity = torch.zeros(N, device=device)

    # Also need d(srgb_color_to_linear) for color gradients
    colors_srgb_grad = _srgb_to_linear_grad(colors)  # [N, 3]

    # Recompute forward AND store T_before for each shape
    T_before_list = []
    T_curr = torch.ones(H, W, device=device)
    with torch.no_grad():
        for i in range(N):
            T_before_list.append(T_curr.clone())
            T_curr = T_curr * (1.0 - alphas[i])

    # Now process backward (front-to-back)
    dLdC_r = dLdC[0]
    dLdC_g = dLdC[1]
    dLdC_b = dLdC[2]
    dLdT = torch.zeros(H, W, device=device)

    for i in range(N - 1, -1, -1):
        alpha = alphas[i]
        T_prev = T_before_list[i]
        col_lin = colors_lin[i]  # [3]
        cr, cg, cb = col_lin[0], col_lin[1], col_lin[2]

        # dL/dα
        dot_cc = dLdC_r * cr + dLdC_g * cg + dLdC_b * cb
        dLda = (dot_cc - dLdT) * T_prev

        # --- Gradients w.r.t. opacity ---
        # α = α_raw * opacity, so dα/d(opacity) = α_raw
        # But we stored alpha = alpha_raw * opacity, so α_raw = alpha / opacity
        alpha_raw = alpha / (opacity[i] + 1e-8)
        grad_opacity[i] = (dLda * alpha_raw).sum()

        # --- Gradients w.r.t. colors (linear) ---
        # C += T_prev * α * col_lin
        # dL/d(col_lin) = dLdC * T_prev * α
        dcol_r = (dLdC_r * T_prev * alpha).sum()
        dcol_g = (dLdC_g * T_prev * alpha).sum()
        dcol_b = (dLdC_b * T_prev * alpha).sum()
        # Convert linear color gradient to sRGB color gradient
        grad_colors[i, 0] = dcol_r * colors_srgb_grad[i, 0]
        grad_colors[i, 1] = dcol_g * colors_srgb_grad[i, 1]
        grad_colors[i, 2] = dcol_b * colors_srgb_grad[i, 2]

        # --- Chain rule to tx, ty ---
        # dα/d(tx) = dα_raw/dtx * opacity  (already computed in dalpha_dtxs)
        dLdtx = dLda * dalpha_dtxs[i] * opacity[i]
        dLdty = dLda * dalpha_dtys[i] * opacity[i]

        # --- Coordinate chain rule ---
        cos_a = cos_as[i]
        sin_a = sin_as[i]
        inv_rx = inv_rxs[i]
        inv_ry = inv_rys[i]
        dx_rot = dx_rots[i]
        dy_rot = dy_rots[i]
        F = TEMPLATE_FILL_RATIO

        # dtx/dcx = -cos_a * inv_rx * F
        # dty/dcx = -sin_a * inv_ry * F
        dLdcx = dLdtx * (-cos_a * inv_rx * F) + dLdty * (-sin_a * inv_ry * F)
        grad_cx[i] = dLdcx.sum()

        # dtx/dcy = sin_a * inv_rx * F
        # dty/dcy = -cos_a * inv_ry * F
        dLdcy = dLdtx * (sin_a * inv_rx * F) + dLdty * (-cos_a * inv_ry * F)
        grad_cy[i] = dLdcy.sum()

        # dtx/drx = -tx / rx,  dty/drx = 0
        dLdrx = dLdtx * (-txs[i] * inv_rx)
        grad_rx[i] = dLdrx.sum()

        # dty/dry = -ty / ry
        dLdry = dLdty * (-tys[i] * inv_ry)
        grad_ry[i] = dLdry.sum()

        # dtx/dφ = -dy_rot * inv_rx * F,  dty/dφ = dx_rot * inv_ry * F
        # φ = -angle_rad, dφ/d(angle_deg) = -DEG2RAD
        dLdphi = dLdtx * (-dy_rot * inv_rx * F) + dLdty * (dx_rot * inv_ry * F)
        grad_angle[i] = dLdphi.sum() * (-DEG2RAD)

        # --- Update dLdC, dLdT for previous shapes ---
        # dLdC unchanged (C_prev contributes directly to C)
        # dLdT_prev = dLdT * (1-α) + dot_cc * α
        dLdT = dLdT * (1.0 - alpha) + dot_cc * alpha

    return {
        'cx': grad_cx, 'cy': grad_cy,
        'rx': grad_rx, 'ry': grad_ry,
        'angle': grad_angle,
        'colors': grad_colors,
        'opacity': grad_opacity,
    }


# ---------------------------------------------------------------------------
# autograd.Function wrapper
# ---------------------------------------------------------------------------

class ManualOverCompositeSTE(Function):
    """
    STE Over compositing using manual forward + backward.
    Forward: hard template (binary alpha).
    Backward: soft template (analytical gradients).
    """

    @staticmethod
    def forward(
        ctx,
        hard_templates, soft_templates, type_indices,
        cx, cy, rx, ry, angle, colors, opacity,
        H, W, background,
    ):
        # Save everything for backward
        ctx.save_for_backward(
            soft_templates, type_indices,
            cx.detach(), cy.detach(), rx.detach(), ry.detach(),
            angle.detach(), colors.detach(), opacity.detach(),
        )
        ctx.H, ctx.W = H, W

        with torch.no_grad():
            return over_composite_forward(
                hard_templates, type_indices,
                cx, cy, rx, ry, angle, colors, opacity,
                H, W, background, cx.device if hasattr(cx, 'device') else 'cpu',
            )

    @staticmethod
    def backward(ctx, grad_output):
        soft_tmpl, type_idx, cx_v, cy_v, rx_v, ry_v, angle_v, colors_v, opacity_v = \
            ctx.saved_tensors
        H, W = ctx.H, ctx.W
        device = cx_v.device

        grads = over_composite_backward(
            soft_tmpl, type_idx,
            cx_v, cy_v, rx_v, ry_v, angle_v, colors_v, opacity_v,
            H, W, grad_output, device,
        )

        return (
            None, None, None,  # templates, type_indices
            grads['cx'], grads['cy'], grads['rx'], grads['ry'], grads['angle'],
            grads['colors'], grads['opacity'],
            None, None, None,
        )


# ---------------------------------------------------------------------------
# Gradient checker: compare manual backward vs PyTorch autograd
# ---------------------------------------------------------------------------

def check_gradients(
    num_shapes: int = 10,
    num_types: int = 4,
    H: int = 64,
    W: int = 64,
    device: str = "cuda",
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> dict:
    """
    Compare manual backward gradients with PyTorch autograd.

    Returns dict with per-parameter max absolute difference.
    """
    from .templates import generate_synthetic_templates

    lib = generate_synthetic_templates(num_types=num_types, device=device)

    torch.manual_seed(42)
    cx = torch.rand(num_shapes, device=device) * W
    cy = torch.rand(num_shapes, device=device) * H
    rx = torch.rand(num_shapes, device=device) * 20 + 5
    ry = torch.rand(num_shapes, device=device) * 20 + 5
    angle = torch.rand(num_shapes, device=device) * 360
    colors = torch.rand(num_shapes, 3, device=device)
    opacity = torch.rand(num_shapes, device=device) * 0.5 + 0.5
    type_idx = torch.randint(0, num_types, (num_shapes,), device=device)
    bg = torch.zeros(3, device=device)

    # --- Method 1: Manual backward ---
    with torch.no_grad():
        out_manual = over_composite_forward(
            lib['hard'], type_idx, cx, cy, rx, ry, angle, colors, opacity,
            H, W, bg, device,
        )

    grad_out = torch.randn(3, H, W, device=device)
    manual_grads = over_composite_backward(
        lib['soft'], type_idx, cx, cy, rx, ry, angle, colors, opacity,
        H, W, grad_out, device,
    )

    # --- Method 2: PyTorch autograd ---
    cx_a = cx.detach().clone().requires_grad_(True)
    cy_a = cy.detach().clone().requires_grad_(True)
    rx_a = rx.detach().clone().requires_grad_(True)
    ry_a = ry.detach().clone().requires_grad_(True)
    angle_a = angle.detach().clone().requires_grad_(True)
    colors_a = colors.detach().clone().requires_grad_(True)
    opacity_a = opacity.detach().clone().requires_grad_(True)

    # Use ManualOverCompositeSTE for autograd path too (same forward)
    out_auto = ManualOverCompositeSTE.apply(
        lib['hard'], lib['soft'], type_idx,
        cx_a, cy_a, rx_a, ry_a, angle_a, colors_a, opacity_a,
        H, W, bg,
    )
    (out_auto * grad_out).sum().backward()

    auto_grads = {
        'cx': cx_a.grad, 'cy': cy_a.grad,
        'rx': rx_a.grad, 'ry': ry_a.grad,
        'angle': angle_a.grad,
        'colors': colors_a.grad,
        'opacity': opacity_a.grad,
    }

    # --- Compare ---
    results = {}
    all_ok = True
    for name in ['cx', 'cy', 'rx', 'ry', 'angle', 'colors', 'opacity']:
        mg = manual_grads[name]
        ag = auto_grads[name]
        if ag is None:
            results[name] = {'status': 'AUTO_GRAD_IS_NONE', 'max_diff': float('nan')}
            all_ok = False
            continue

        diff = (mg - ag).abs().max().item()
        mag = ag.abs().max().item()
        rel_diff = diff / (mag + 1e-8)
        ok = rel_diff < rtol or diff < atol
        results[name] = {
            'status': 'OK' if ok else 'MISMATCH',
            'max_diff': diff,
            'auto_max_abs': mag,
            'rel_diff': rel_diff,
        }
        if not ok:
            all_ok = False

    results['_all_ok'] = all_ok
    return results


# ===========================================================================
# Step 2: Tiled versions (numerically equivalent to full-canvas)
# ===========================================================================

def _compute_aabbs(
    cx: torch.Tensor, cy: torch.Tensor,
    rx: torch.Tensor, ry: torch.Tensor,
    angle_deg: torch.Tensor,
    blur_pad: float = 3.0,
) -> torch.Tensor:
    """Returns [N, 4] (x0, y0, x1, y1) per shape."""
    ang_rad = angle_deg * DEG2RAD
    cos_a = torch.abs(torch.cos(ang_rad))
    sin_a = torch.abs(torch.sin(ang_rad))
    hw = rx * cos_a + ry * sin_a + blur_pad
    hh = rx * sin_a + ry * cos_a + blur_pad
    return torch.stack([cx - hw, cy - hh, cx + hw, cy + hh], dim=-1)


def _build_tile_assignments(
    aabbs: torch.Tensor,   # [N, 4]
    H: int, W: int,
    tile_size: int = 128,
):
    """Returns tile_shapes list-of-lists: tile_shapes[ti] = [shape_idx, ...]"""
    ny = (H + tile_size - 1) // tile_size
    nx = (W + tile_size - 1) // tile_size
    x0, y0, x1, y1 = aabbs[:, 0], aabbs[:, 1], aabbs[:, 2], aabbs[:, 3]

    tile_shapes = []
    for ty in range(ny):
        for tx in range(nx):
            t_x0 = tx * tile_size
            t_y0 = ty * tile_size
            t_x1 = min(t_x0 + tile_size, W)
            t_y1 = min(t_y0 + tile_size, H)
            overlaps = (x1 > t_x0) & (x0 < t_x1) & (y1 > t_y0) & (y0 < t_y1)
            tile_shapes.append(torch.where(overlaps)[0])
    return tile_shapes, nx, ny


def over_composite_forward_tiled(
    hard_templates: torch.Tensor,
    type_indices: torch.Tensor,
    cx, cy, rx, ry, angle, colors, opacity,
    H: int, W: int,
    background: torch.Tensor,
    device: str = "cpu",
    tile_size: int = 128,
) -> torch.Tensor:
    """
    Tile-based forward. Numerically identical to over_composite_forward.
    """
    N = cx.shape[0]
    aabbs = _compute_aabbs(cx, cy, rx, ry, angle)
    tile_shapes, ntx, nty = _build_tile_assignments(aabbs, H, W, tile_size)
    ntiles = len(tile_shapes)

    bg_lin = _srgb_to_linear(background)
    output = torch.zeros(3, H, W, device=device)

    for ti in range(ntiles):
        idx = tile_shapes[ti]
        if len(idx) == 0:
            continue

        ty = ti // ntx
        tx = ti % ntx
        y0 = ty * tile_size
        x0 = tx * tile_size
        y1 = min(y0 + tile_size, H)
        x1 = min(x0 + tile_size, W)
        th = y1 - y0
        tw = x1 - x0

        px, py = _make_grid(th, tw, device)
        px = px + x0
        py = py + y0

        C = bg_lin.view(3, 1, 1).expand(3, th, tw).contiguous().clone()
        T = torch.ones(th, tw, device=device)

        with torch.no_grad():
            for si in idx:
                i = int(si.item())
                tidx = int(type_indices[i].item())
                tx_c, ty_c, _, _, _, _, _, _ = _coord_transform(
                    px, py, cx[i], cy[i], rx[i], ry[i], angle[i],
                )
                alpha_raw, _, _ = _bilinear_sample(hard_templates[tidx], tx_c, ty_c)
                alpha = (alpha_raw > 0.5).float() * opacity[i]
                alpha = alpha.clamp(0.0, 1.0)
                w = alpha * T
                col_lin = _srgb_to_linear(colors[i])
                C = C + w.unsqueeze(0) * col_lin.view(3, 1, 1)
                T = T * (1.0 - alpha)

        output[:, y0:y1, x0:x1] = _linear_to_srgb(C)

    return output


def over_composite_backward_tiled(
    soft_templates: torch.Tensor,
    type_indices: torch.Tensor,
    cx, cy, rx, ry, angle, colors, opacity,
    H: int, W: int,
    grad_output: torch.Tensor,   # [3, H, W]
    device: str = "cpu",
    tile_size: int = 128,
) -> dict:
    """
    Tile-based analytical backward. Numerically identical to over_composite_backward.
    """
    N = cx.shape[0]
    aabbs = _compute_aabbs(cx, cy, rx, ry, angle)
    tile_shapes, ntx, nty = _build_tile_assignments(aabbs, H, W, tile_size)
    ntiles = len(tile_shapes)

    colors_srgb_grad = _srgb_to_linear_grad(colors)

    # Global gradient buffers
    grad_cx = torch.zeros(N, device=device)
    grad_cy = torch.zeros(N, device=device)
    grad_rx = torch.zeros(N, device=device)
    grad_ry = torch.zeros(N, device=device)
    grad_angle = torch.zeros(N, device=device)
    grad_colors = torch.zeros(N, 3, device=device)
    grad_opacity = torch.zeros(N, device=device)

    for ti in range(ntiles):
        idx = tile_shapes[ti]
        if len(idx) == 0:
            continue

        ty = ti // ntx
        tx = ti % ntx
        y0 = ty * tile_size
        x0 = tx * tile_size
        y1 = min(y0 + tile_size, H)
        x1 = min(x0 + tile_size, W)
        th = y1 - y0
        tw = x1 - x0

        px, py = _make_grid(th, tw, device)
        px = px + x0
        py = py + y0
        go_tile = grad_output[:, y0:y1, x0:x1]  # [3, th, tw]

        # Indices within the tile (into the global shape arrays)
        idx_list = [int(si.item()) for si in idx]
        K = len(idx_list)

        # --- Recompute soft forward for this tile ---
        alphas_tile = []
        txs_tile, tys_tile = [], []
        cos_as_tile, sin_as_tile = [], []
        dxr_tile, dyr_tile = [], []
        irx_tile, iry_tile = [], []
        da_dtx_tile, da_dty_tile = [], []
        colors_lin_tile = []

        with torch.no_grad():
            for si in idx:
                i = int(si.item())
                tidx = int(type_indices[i].item())
                tx_c, ty_c, cos_a, sin_a, dx_rot, dy_rot, inv_rx, inv_ry = \
                    _coord_transform(px, py, cx[i], cy[i], rx[i], ry[i], angle[i])
                alpha_raw, da_dtx, da_dty = _bilinear_sample(soft_templates[tidx], tx_c, ty_c)
                alpha = alpha_raw * opacity[i]
                alpha = alpha.clamp(0.0, 1.0)

                alphas_tile.append(alpha)
                txs_tile.append(tx_c); tys_tile.append(ty_c)
                cos_as_tile.append(cos_a); sin_as_tile.append(sin_a)
                dxr_tile.append(dx_rot); dyr_tile.append(dy_rot)
                irx_tile.append(inv_rx); iry_tile.append(inv_ry)
                da_dtx_tile.append(da_dtx); da_dty_tile.append(da_dty)
                colors_lin_tile.append(_srgb_to_linear(colors[i]))

        # --- Compute dL/dC_linear for this tile ---
        bg_lin = _srgb_to_linear(torch.zeros(3, device=device))
        C_tile = bg_lin.view(3, 1, 1).expand(3, th, tw).contiguous().clone()
        T_fwd = torch.ones(th, tw, device=device)
        with torch.no_grad():
            for k in range(K):
                w = alphas_tile[k] * T_fwd
                C_tile = C_tile + w.unsqueeze(0) * colors_lin_tile[k].view(3, 1, 1)
                T_fwd = T_fwd * (1.0 - alphas_tile[k])

        dsrgb_dlin = _linear_to_srgb_grad(C_tile)
        dLdC_tile = go_tile * dsrgb_dlin  # [3, th, tw]

        # --- T_before per shape in this tile ---
        T_before_tile = []
        T_curr = torch.ones(th, tw, device=device)
        with torch.no_grad():
            for k in range(K):
                T_before_tile.append(T_curr.clone())
                T_curr = T_curr * (1.0 - alphas_tile[k])

        # --- Backward pass for this tile ---
        dLdC_r = dLdC_tile[0]
        dLdC_g = dLdC_tile[1]
        dLdC_b = dLdC_tile[2]
        dLdT = torch.zeros(th, tw, device=device)
        F = TEMPLATE_FILL_RATIO

        for k in range(K - 1, -1, -1):
            gi = idx_list[k]  # global shape index
            alpha = alphas_tile[k]
            T_prev = T_before_tile[k]
            cr, cg, cb = colors_lin_tile[k][0], colors_lin_tile[k][1], colors_lin_tile[k][2]

            dot_cc = dLdC_r * cr + dLdC_g * cg + dLdC_b * cb
            dLda = (dot_cc - dLdT) * T_prev

            # Opacity gradient
            alpha_raw = alpha / (opacity[gi] + 1e-8)
            grad_opacity[gi] += (dLda * alpha_raw).sum()

            # Color gradients
            dcr = (dLdC_r * T_prev * alpha).sum()
            dcg = (dLdC_g * T_prev * alpha).sum()
            dcb = (dLdC_b * T_prev * alpha).sum()
            grad_colors[gi, 0] += dcr * colors_srgb_grad[gi, 0]
            grad_colors[gi, 1] += dcg * colors_srgb_grad[gi, 1]
            grad_colors[gi, 2] += dcb * colors_srgb_grad[gi, 2]

            # Coordinate gradients
            dLdtx = dLda * da_dtx_tile[k] * opacity[gi]
            dLdty = dLda * da_dty_tile[k] * opacity[gi]

            cos_a = cos_as_tile[k]
            sin_a = sin_as_tile[k]
            inv_rx = irx_tile[k]
            inv_ry = iry_tile[k]

            dLdcx = dLdtx * (-cos_a * inv_rx * F) + dLdty * (-sin_a * inv_ry * F)
            grad_cx[gi] += dLdcx.sum()

            dLdcy = dLdtx * (sin_a * inv_rx * F) + dLdty * (-cos_a * inv_ry * F)
            grad_cy[gi] += dLdcy.sum()

            grad_rx[gi] += (dLdtx * (-txs_tile[k] * inv_rx)).sum()
            grad_ry[gi] += (dLdty * (-tys_tile[k] * inv_ry)).sum()

            dLdphi = dLdtx * (-dyr_tile[k] * inv_rx * F) + dLdty * (dxr_tile[k] * inv_ry * F)
            grad_angle[gi] += dLdphi.sum() * (-DEG2RAD)

            dLdT = dLdT * (1.0 - alpha) + dot_cc * alpha

    return {
        'cx': grad_cx, 'cy': grad_cy,
        'rx': grad_rx, 'ry': grad_ry,
        'angle': grad_angle,
        'colors': grad_colors,
        'opacity': grad_opacity,
    }


# ===========================================================================
# Tiled gradient checker
# ===========================================================================

def check_tiled_gradients(
    num_shapes: int = 20,
    num_types: int = 4,
    H: int = 128,
    W: int = 128,
    device: str = "cuda",
    tile_size: int = 32,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> dict:
    """Compare tiled forward+backward vs full-canvas versions."""
    from .templates import generate_synthetic_templates

    lib = generate_synthetic_templates(num_types=num_types, device=device)

    torch.manual_seed(42)
    cx = torch.rand(num_shapes, device=device) * W
    cy = torch.rand(num_shapes, device=device) * H
    rx = torch.rand(num_shapes, device=device) * 20 + 5
    ry = torch.rand(num_shapes, device=device) * 20 + 5
    angle = torch.rand(num_shapes, device=device) * 360
    colors = torch.rand(num_shapes, 3, device=device)
    opacity = torch.rand(num_shapes, device=device) * 0.5 + 0.5
    type_idx = torch.randint(0, num_types, (num_shapes,), device=device)
    bg = torch.zeros(3, device=device)

    # Full-canvas forward
    with torch.no_grad():
        out_full = over_composite_forward(
            lib['hard'], type_idx, cx, cy, rx, ry, angle, colors, opacity,
            H, W, bg, device,
        )

    # Tiled forward
    with torch.no_grad():
        out_tiled = over_composite_forward_tiled(
            lib['hard'], type_idx, cx, cy, rx, ry, angle, colors, opacity,
            H, W, bg, device, tile_size,
        )

    fwd_diff = (out_full - out_tiled).abs().max().item()

    # Full-canvas backward
    grad_out = torch.randn(3, H, W, device=device)
    grads_full = over_composite_backward(
        lib['soft'], type_idx, cx, cy, rx, ry, angle, colors, opacity,
        H, W, grad_out, device,
    )

    # Tiled backward
    grads_tiled = over_composite_backward_tiled(
        lib['soft'], type_idx, cx, cy, rx, ry, angle, colors, opacity,
        H, W, grad_out, device, tile_size,
    )

    results = {'fwd_max_diff': fwd_diff}
    all_ok = fwd_diff < atol

    for name in ['cx', 'cy', 'rx', 'ry', 'angle', 'colors', 'opacity']:
        gf = grads_full[name]
        gt = grads_tiled[name]
        diff = (gf - gt).abs().max().item()
        mag = gf.abs().max().item()
        rel = diff / (mag + 1e-8)
        ok = rel < rtol or diff < atol
        results[name] = {'diff': diff, 'rel': rel, 'ok': ok}
        if not ok:
            all_ok = False

    results['_all_ok'] = all_ok
    return results
