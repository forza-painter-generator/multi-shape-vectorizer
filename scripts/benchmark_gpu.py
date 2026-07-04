#!/usr/bin/env python
"""Benchmark PyTorch vs Triton vs torch.compile on GPU."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from fh6_vectorizer.templates import generate_synthetic_templates
from fh6_vectorizer.ste_renderer import over_composite_render, TEMPLATE_FILL_RATIO
from fh6_vectorizer.triton_kernels import TritonOverCompositeSTE

lib = generate_synthetic_templates(num_types=4, device="cuda")

def make_r(n=100, h=256, w=256):
    from fh6_vectorizer.ste_renderer import STEVectorRenderer
    return STEVectorRenderer(num_shapes=n, num_types=4, hard_templates=lib["hard"], soft_templates=lib["soft"], canvas_height=h, canvas_width=w, device="cuda")

def bench(name, fn, iters=10):
    for _ in range(3):
        out = fn()
        out.mean().backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
        out.mean().backward()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000

for h, n in [(256, 100), (256, 300), (512, 200)]:
    print(f"\n--- {h}x{h}, {n} shapes ---")
    r = make_r(n, h, h)

    def pytorch_fn():
        return over_composite_render(
            r.hard_templates, r.soft_templates, r.type_indices,
            r.cx, r.cy, r.rx, r.ry, r.angle, r.colors, r.opacity,
            h, h, r.background, "cuda",
        )

    def triton_fn():
        return TritonOverCompositeSTE.apply(
            r.hard_templates, r.soft_templates, r.type_indices,
            r.cx, r.cy, r.rx, r.ry, r.angle, r.colors, r.opacity,
            h, h, r.background, TEMPLATE_FILL_RATIO,
        )

    pt = bench("PyTorch", pytorch_fn)
    tr = bench("Triton", triton_fn)
    print(f"  PyTorch GPU:  {pt:8.1f}ms")
    print(f"  Triton STE:   {tr:8.1f}ms  ({pt/tr:.1f}x speedup)")
