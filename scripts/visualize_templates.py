#!/usr/bin/env python
"""
Visualize template library: save hard + soft templates as a PNG grid.

Usage:
    python scripts/visualize_templates.py [--output templates.png] [--num-types 16]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fh6_vectorizer.templates import generate_synthetic_templates, TEMPLATE_SIZE


def visualize_templates(
    output_path: Path,
    num_types: int = 16,
    cols: int = 4,
):
    """Generate template library and save as a PNG grid."""
    try:
        from PIL import Image
    except ImportError:
        print("Error: Pillow is required.")
        sys.exit(1)

    lib = generate_synthetic_templates(num_types=num_types)
    hard = lib["hard"].numpy()  # [N, T, T]
    soft = lib["soft"].numpy()
    names = lib["names"]
    N = hard.shape[0]

    rows = (N * 2 + cols - 1) // cols  # 2 rows per shape (hard + soft)
    cell = TEMPLATE_SIZE + 4  # cell size with padding
    canvas = np.ones((rows * cell, cols * cell, 3), dtype=np.float32)

    for i in range(N):
        r_hard = (i // cols) * 2
        r_soft = r_hard + 1
        c = i % cols

        y0, x0 = r_hard * cell + 2, c * cell + 2
        y1, x1 = y0 + TEMPLATE_SIZE, x0 + TEMPLATE_SIZE
        canvas[y0:y1, x0:x1, 0] = hard[i]
        canvas[y0:y1, x0:x1, 1] = hard[i]
        canvas[y0:y1, x0:x1, 2] = hard[i]

        # Label
        # (skip for brevity — could use PIL draw)

        y0_s, x0_s = r_soft * cell + 2, c * cell + 2
        canvas[y0_s:y1, x0_s:x1, 0] = soft[i]
        canvas[y0_s:y1, x0_s:x1, 1] = soft[i]
        canvas[y0_s:y1, x0_s:x1, 2] = soft[i]

    canvas = (canvas.clip(0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(canvas, mode="RGB")
    img.save(output_path)

    print(f"Saved template visualization to {output_path}")
    print(f"Templates: {names}")
    print(f"Grid: {rows} rows × {cols} cols (hard top, soft bottom per pair)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize template library")
    parser.add_argument("-o", "--output", type=Path, default=Path("templates.png"),
                        help="Output PNG path")
    parser.add_argument("-n", "--num-types", type=int, default=16,
                        help="Number of template types")
    parser.add_argument("--cols", type=int, default=4, help="Grid columns")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    visualize_templates(args.output, args.num_types, args.cols)
