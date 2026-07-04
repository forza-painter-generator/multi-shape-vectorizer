#!/usr/bin/env python
"""Benchmark PyTorch eager vs torch.compile on GPU (Triton backend)."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from fh6_vectorizer.templates import generate_synthetic_templates
from fh6_vectorizer.ste_renderer import STEVectorRenderer

lib = generate_synthetic_templates(num_types=4, device="cuda")

def make_r(n=100, h=256, w=256):
    return STEVectorRenderer(num_shapes=n, num_types=4, hard_templates=lib["hard"], soft_templates=lib["soft"], canvas_height=h, canvas_width=w, device="cuda")

for h, n in [(256, 100), (256, 300), (512, 200)]:
    print(f"\n--- {h}x{h}, {n} shapes ---")

    # PyTorch eager (force non-compiled path)
    r1 = make_r(n, h, h)
    for _ in range(3):
        out = r1._pytorch_forward()
        out.mean().backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        out = r1._pytorch_forward()
        out.mean().backward()
    torch.cuda.synchronize()
    pt = (time.perf_counter() - t0) / 10 * 1000

    # torch.compile (uses Triton)
    r2 = make_r(n, h, h)
    # clear compile cache
    r2._compiled_forward = torch.compile(r2._pytorch_forward, mode="reduce-overhead")
    for _ in range(5):
        out = r2.forward()
        out.mean().backward()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(10):
        out = r2.forward()
        out.mean().backward()
    torch.cuda.synchronize()
    tc = (time.perf_counter() - t0) / 10 * 1000

    print(f"  PyTorch eager:  {pt:8.1f}ms")
    print(f"  torch.compile:  {tc:8.1f}ms  ({pt/tc:.1f}x speedup)")
