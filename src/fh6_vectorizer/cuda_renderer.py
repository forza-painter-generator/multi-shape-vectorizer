"""
CUDA-accelerated Over compositing renderer using torch.utils.cpp_extension.

Provides fused CUDA kernels for the forward (Over compositing) pass
as an alternative to Triton on Windows. Backward pass relies on
PyTorch autograd through F.grid_sample.

Usage:
    from .cuda_renderer import CUDARenderer
    renderer = CUDARenderer(hard_templates, soft_templates)
    result = renderer(cx, cy, rx, ry, angle, colors, opacity, type_indices, H, W, bg)

Reference:
  - vinylizer/src/cuda/render_kernel.cu — CUDA Over compositing reference
  - IMPLEMENTATION_PLAN.md §5 — IGS Triton kernel patterns
"""

import os
from pathlib import Path
from typing import Optional

import torch
from torch.utils.cpp_extension import load_inline

# --- CUDA kernel source ---
# Fused Over compositing: for each pixel, iterate shapes back-to-front,
# compute alpha from template via bilinear sampling, accumulate.

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

// Bilinear sample from a 2D template texture
// template: [T, T] float (stored row-major)
// T: template size
// tx, ty: normalized coordinates in [-1, 1] (grid_sample convention)
__device__ float bilinear_sample_2d(
    const float* __restrict__ template_data,
    int T,
    float tx, float ty
) {
    // Convert [-1, 1] to [0, T-1]
    float u = (tx + 1.0f) * 0.5f * (T - 1);
    float v = (ty + 1.0f) * 0.5f * (T - 1);

    // Clamp to valid range
    u = fminf(fmaxf(u, 0.0f), T - 1.001f);
    v = fminf(fmaxf(v, 0.0f), T - 1.001f);

    int x0 = (int)u;
    int y0 = (int)v;
    int x1 = min(x0 + 1, T - 1);
    int y1 = min(y0 + 1, T - 1);

    float fx = u - x0;
    float fy = v - y0;

    float v00 = template_data[y0 * T + x0];
    float v10 = template_data[y0 * T + x1];
    float v01 = template_data[y1 * T + x0];
    float v11 = template_data[y1 * T + x1];

    return (1.0f - fx) * (1.0f - fy) * v00
         + fx * (1.0f - fy) * v10
         + (1.0f - fx) * fy * v01
         + fx * fy * v11;
}

// Over compositing kernel: one thread per pixel
__global__ void over_composite_forward_kernel(
    const float* __restrict__ templates,   // [num_types, T, T]
    const int64_t* __restrict__ type_idx,  // [N]
    const float* __restrict__ cx,          // [N]
    const float* __restrict__ cy,          // [N]
    const float* __restrict__ rx,          // [N]
    const float* __restrict__ ry,          // [N]
    const float* __restrict__ angle,       // [N] degrees
    const float* __restrict__ colors,      // [N, 3]
    const float* __restrict__ opacity,     // [N]
    float* __restrict__ output,            // [H, W, 3]
    int N, int T, int H, int W,
    float bg_r, float bg_g, float bg_b,
    float fill_ratio
) {
    int px = blockIdx.x * blockDim.x + threadIdx.x;
    int py = blockIdx.y * blockDim.y + threadIdx.y;

    if (px >= W || py >= H) return;

    // Initialize with background
    float Cr = bg_r, Cg = bg_g, Cb = bg_b;
    float transmittance = 1.0f;

    float DEG2RAD = 3.14159265358979323846f / 180.0f;

    for (int i = 0; i < N; i++) {
        float shape_cx = cx[i];
        float shape_cy = cy[i];
        float shape_rx = rx[i] + 1e-8f;
        float shape_ry = ry[i] + 1e-8f;
        float shape_angle = angle[i] * DEG2RAD;

        // Coordinate transform: canvas pixel → template space
        float dx = (float)px - shape_cx;
        float dy = (float)py - shape_cy;

        float cos_a = cosf(-shape_angle);
        float sin_a = sinf(-shape_angle);
        float dx_rot = dx * cos_a - dy * sin_a;
        float dy_rot = dx * sin_a + dy * cos_a;

        float tx = dx_rot / shape_rx * fill_ratio;
        float ty = dy_rot / shape_ry * fill_ratio;

        // Bilinear sample from template
        int tidx = (int)type_idx[i];
        const float* tmpl = templates + tidx * T * T;
        float alpha = bilinear_sample_2d(tmpl, T, tx, ty);

        // Threshold: hard binary alpha
        alpha = (alpha > 0.5f) ? 1.0f : 0.0f;
        alpha = alpha * opacity[i];
        alpha = fminf(fmaxf(alpha, 0.0f), 1.0f);

        float w = alpha * transmittance;
        Cr += w * colors[i * 3 + 0];
        Cg += w * colors[i * 3 + 1];
        Cb += w * colors[i * 3 + 2];
        transmittance *= (1.0f - alpha);

        if (transmittance < 1e-4f) break;
    }

    int out_idx = (py * W + px) * 3;
    output[out_idx + 0] = Cr;
    output[out_idx + 1] = Cg;
    output[out_idx + 2] = Cb;
}


torch::Tensor cuda_over_composite_forward(
    torch::Tensor templates,       // [num_types, T, T] float
    torch::Tensor type_indices,    // [N] int64
    torch::Tensor cx, torch::Tensor cy,  // [N] float
    torch::Tensor rx, torch::Tensor ry,  // [N] float
    torch::Tensor angle,           // [N] float (degrees)
    torch::Tensor colors,          // [N, 3] float
    torch::Tensor opacity,         // [N] float
    int H, int W,
    torch::Tensor background,      // [3] float
    float fill_ratio
) {
    int N = cx.size(0);
    int T = templates.size(1);
    auto output = torch::zeros({H, W, 3}, templates.options());

    dim3 threads(16, 16);
    dim3 blocks((W + 15) / 16, (H + 15) / 16);

    over_composite_forward_kernel<<<blocks, threads>>>(
        templates.data_ptr<float>(),
        type_indices.data_ptr<int64_t>(),
        cx.data_ptr<float>(), cy.data_ptr<float>(),
        rx.data_ptr<float>(), ry.data_ptr<float>(),
        angle.data_ptr<float>(),
        colors.data_ptr<float>(),
        opacity.data_ptr<float>(),
        output.data_ptr<float>(),
        N, T, H, W,
        background[0].item<float>(),
        background[1].item<float>(),
        background[2].item<float>(),
        fill_ratio
    );

    return output;
}
"""

_CPP_SOURCE = """
torch::Tensor cuda_over_composite_forward(
    torch::Tensor templates,
    torch::Tensor type_indices,
    torch::Tensor cx, torch::Tensor cy,
    torch::Tensor rx, torch::Tensor ry,
    torch::Tensor angle,
    torch::Tensor colors,
    torch::Tensor opacity,
    int H, int W,
    torch::Tensor background,
    float fill_ratio);
"""

# --- Build cache ---
_cuda_module: Optional[object] = None


def _get_cuda_module() -> object:
    """Lazy-load the CUDA extension (compiled once, cached)."""
    global _cuda_module
    if _cuda_module is not None:
        return _cuda_module

    # Cache compiled module in user's temp dir
    cache_dir = Path.home() / ".cache" / "fh6_vectorizer_cuda"
    cache_dir.mkdir(parents=True, exist_ok=True)

    try:
        _cuda_module = load_inline(
            name="fh6_cuda_over_composite",
            cpp_sources=_CPP_SOURCE,
            cuda_sources=_CUDA_SOURCE,
            functions=["cuda_over_composite_forward"],
            extra_cuda_cflags=["-O3", "--use_fast_math"],
            build_directory=str(cache_dir),
            verbose=False,
        )
        print("CUDA extension compiled successfully.")
    except Exception as e:
        print(f"CUDA extension compilation failed: {e}")
        print("Falling back to PyTorch renderer.")
        _cuda_module = None

    return _cuda_module


def cuda_over_composite_fused(
    hard_templates: torch.Tensor,  # [num_types, T, T]
    type_indices: torch.Tensor,     # [N]
    cx: torch.Tensor,               # [N]
    cy: torch.Tensor,               # [N]
    rx: torch.Tensor,               # [N]
    ry: torch.Tensor,               # [N]
    angle: torch.Tensor,            # [N] degrees
    colors: torch.Tensor,           # [N, 3] in [0, 1]
    opacity: torch.Tensor,          # [N]
    canvas_height: int,
    canvas_width: int,
    background: torch.Tensor,       # [3]
    fill_ratio: float = 0.9,
) -> Optional[torch.Tensor]:
    """
    CUDA-accelerated forward Over compositing.

    Returns None if CUDA extension is not available (caller should fall back).
    """
    mod = _get_cuda_module()
    if mod is None:
        return None

    # Ensure contiguous float tensors on CUDA
    def _to_cuda(t):
        return t.contiguous().cuda().float() if t.dtype != torch.float32 else t.contiguous().cuda()

    result = mod.cuda_over_composite_forward(
        _to_cuda(hard_templates),
        type_indices.contiguous().cuda(),
        _to_cuda(cx), _to_cuda(cy),
        _to_cuda(rx), _to_cuda(ry),
        _to_cuda(angle),
        _to_cuda(colors),
        _to_cuda(opacity),
        canvas_height, canvas_width,
        _to_cuda(background),
        fill_ratio,
    )
    # [H, W, 3] → [3, H, W]
    return result.permute(2, 0, 1).contiguous()
