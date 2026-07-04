#!/usr/bin/env python
"""
Performance profiling script for the FH6 vectorizer.

Uses torch.profiler to identify bottlenecks in the rendering and optimization pipeline.

Usage:
    python scripts/profile.py [--size 256] [--num-shapes 100] [--device cpu]
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from fh6_vectorizer.templates import generate_synthetic_templates
from fh6_vectorizer.ste_renderer import STEVectorRenderer


def profile_rendering(
    canvas_size: int = 256,
    num_shapes: int = 100,
    num_types: int = 8,
    device: str = "cpu",
    steps: int = 3,
):
    """Profile the rendering + backward pass."""
    lib = generate_synthetic_templates(num_types=num_types, device=device)
    renderer = STEVectorRenderer(
        num_shapes=num_shapes, num_types=num_types,
        hard_templates=lib["hard"], soft_templates=lib["soft"],
        canvas_height=canvas_size, canvas_width=canvas_size,
        device=device,
    )

    print(f"Profiling: {canvas_size}×{canvas_size}, {num_shapes} shapes, {num_types} types, device={device}")

    # Warmup
    for _ in range(2):
        rendered = renderer()
        loss = rendered.mean()
        loss.backward()

    # Profile
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
        ] + ([torch.profiler.ProfilerActivity.CUDA] if device == "cuda" else []),
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        for i in range(steps):
            rendered = renderer()
            loss = rendered.mean()
            loss.backward()
            prof.step()

    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    # Also output as Chrome trace
    trace_path = f"profile_trace_{canvas_size}_{num_shapes}.json"
    prof.export_chrome_trace(trace_path)
    print(f"\nChrome trace saved to {trace_path}")
    print("Open chrome://tracing in Chrome and load this file.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Profile FH6 vectorizer")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--num-shapes", type=int, default=100)
    parser.add_argument("--num-types", type=int, default=8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()

    profile_rendering(args.size, args.num_shapes, args.num_types, args.device, args.steps)
