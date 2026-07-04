#!/usr/bin/env python
"""
Performance benchmark across canvas sizes and shape counts.

Usage:
    python scripts/benchmark.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from fh6_vectorizer.templates import generate_synthetic_templates
from fh6_vectorizer.ste_renderer import STEVectorRenderer


def benchmark_single(
    canvas_size: int,
    num_shapes: int,
    num_types: int = 8,
    device: str = "cpu",
    warmup: int = 3,
    iters: int = 10,
) -> dict:
    """Benchmark a single configuration."""
    lib = generate_synthetic_templates(num_types=num_types, device=device)
    renderer = STEVectorRenderer(
        num_shapes=num_shapes, num_types=num_types,
        hard_templates=lib["hard"], soft_templates=lib["soft"],
        canvas_height=canvas_size, canvas_width=canvas_size,
        device=device,
    )

    # Warmup
    for _ in range(warmup):
        rendered = renderer()
        loss = rendered.mean()
        loss.backward()

    # Benchmark forward
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        rendered = renderer()
        if device == "cuda":
            torch.cuda.synchronize()
    forward_time = (time.perf_counter() - t0) / iters

    # Benchmark forward + backward
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        rendered = renderer()
        loss = rendered.mean()
        loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()
    fwd_bwd_time = (time.perf_counter() - t0) / iters

    return {
        "canvas": canvas_size,
        "shapes": num_shapes,
        "forward_ms": forward_time * 1000,
        "fwd_bwd_ms": fwd_bwd_time * 1000,
    }


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"{'Canvas':>8} {'Shapes':>8} {'Forward(ms)':>12} {'Fwd+Bwd(ms)':>12}")
    print("-" * 44)

    sizes = [128, 256, 512]
    shape_counts = [50, 200, 500]

    for size in sizes:
        for n in shape_counts:
            if size >= 512 and n >= 500 and device == "cpu":
                print(f"{size:>8} {n:>8} {'(skipped)':>12} {'(too large for CPU)':>12}")
                continue
            try:
                result = benchmark_single(size, n, device=device)
                print(
                    f"{result['canvas']:>8} {result['shapes']:>8} "
                    f"{result['forward_ms']:>10.1f}  {result['fwd_bwd_ms']:>10.1f}"
                )
            except Exception as e:
                print(f"{size:>8} {n:>8} {'ERROR':>12} {str(e)[:20]:>12}")


if __name__ == "__main__":
    main()
