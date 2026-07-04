#!/usr/bin/env python
"""
CLI entry point for the FH6 Multi-Shape Differentiable Vectorizer.

Usage:
    python scripts/run_optimize.py --input image.png --output result.png

Examples:
    # Basic synthetic shapes, 200 shapes, 3 cycles
    python scripts/run_optimize.py -i photo.jpg -o result.png -n 200

    # Larger canvas, more shapes, perceptual loss
    python scripts/run_optimize.py -i photo.jpg -o result.png \\
        -n 500 --size 512 512 --perceptual --cycles 5

    # Use FH6 data
    python scripts/run_optimize.py -i photo.jpg -o result.png \\
        --fh6-data "E:/workspace/forza-painter-fh6/src/data/fh6_vinyl_resources/Vinyls"
"""

import argparse
import math
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from fh6_vectorizer.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="FH6 Multi-Shape Differentiable Vectorizer — PoC",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Input/Output
    parser.add_argument("-i", "--input", required=True, help="Input image path")
    parser.add_argument("-o", "--output", default="output.png", help="Output image path")

    # Rendering
    parser.add_argument("-n", "--num-shapes", type=int, default=200,
                        help="Number of shapes to optimize (default: 200)")
    parser.add_argument("--size", type=int, nargs=2, default=[256, 256],
                        help="Canvas size H W (default: 256 256)")
    parser.add_argument("--num-types", type=int, default=8,
                        help="Number of synthetic template types (default: 8)")

    # Optimization
    parser.add_argument("--cycles", type=int, default=3,
                        help="Number of optimization cycles (default: 3)")
    parser.add_argument("--global-steps", type=int, default=150,
                        help="Phase A steps per cycle (default: 150)")
    parser.add_argument("--local-steps", type=int, default=50,
                        help="Phase C steps per cycle (default: 50)")
    parser.add_argument("--lr", type=float, default=0.05,
                        help="Learning rate (default: 0.05)")
    parser.add_argument("--perceptual", action="store_true",
                        help="Enable VGG perceptual loss")
    parser.add_argument("--l1-weight", type=float, default=0.0,
                        help="L1 loss weight (default: 0)")
    parser.add_argument("--huber-weight", type=float, default=0.0,
                        help="Huber loss weight (default: 0)")
    parser.add_argument("--grayscale-weight", type=float, default=0.0,
                        help="Grayscale MSE loss weight (default: 0)")
    parser.add_argument("--alpha-reg", type=float, default=0.0,
                        help="Alpha regularization weight (default: 0)")
    parser.add_argument("--smart-types", action="store_true",
                        help="Enable smart type selection during relocation")
    parser.add_argument("--no-importance", action="store_true",
                        help="Disable importance-map-based initialization")

    # Templates
    parser.add_argument("--fh6-data", type=str, default=None,
                        help="Path to FH6 Vinyls/ directory for real shape data")
    parser.add_argument("--template-cache", type=str, default=None,
                        help="Path to cache/load pre-built template library")
    parser.add_argument("--config-file", type=str, default=None,
                        help="Path to JSON config file (e.g., configs/default.json)")
    parser.add_argument("--fh6-json", action="store_true",
                        help="Generate FH6-compatible JSON output")
    parser.add_argument("--snapshot-dir", type=str, default=None,
                        help="Save intermediate renders to this directory every N steps")

    # Device
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use (default: cuda if available, else cpu)")

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build config — load from JSON file first, then override with CLI args
    config = {}
    if args.config_file:
        from fh6_vectorizer.pipeline import load_config
        config = load_config(Path(args.config_file))
        print(f"Loaded config from {args.config_file}")

    # CLI overrides
    config.update({
        "lr": args.lr,
        "global_steps": args.global_steps,
        "local_steps": args.local_steps,
        "num_cycles": args.cycles,
        "use_perceptual_loss": args.perceptual,
    })
    if args.l1_weight > 0:
        config["l1_weight"] = args.l1_weight
    if args.huber_weight > 0:
        config["huber_weight"] = args.huber_weight
    if args.grayscale_weight > 0:
        config["grayscale_weight"] = args.grayscale_weight
    if args.alpha_reg > 0:
        config["alpha_reg_weight"] = args.alpha_reg
    if args.smart_types:
        config["smart_type_selection"] = True
    if args.no_importance:
        config["use_importance_sampling"] = False
    if args.fh6_json:
        config["include_fh6_json"] = True

    print(f"Device: {args.device}")
    print(f"Canvas: {args.size[0]}×{args.size[1]}, Shapes: {args.num_shapes}, Cycles: {args.cycles}")
    print(f"Config: {config}")

    final, history = run_pipeline(
        target_image_path=input_path,
        output_path=output_path,
        num_shapes=args.num_shapes,
        num_types=args.num_types,
        canvas_size=(args.size[0], args.size[1]),
        use_fh6_data=args.fh6_data is not None,
        fh6_vinyls_root=Path(args.fh6_data) if args.fh6_data else None,
        template_cache_path=Path(args.template_cache) if args.template_cache else None,
        snapshot_dir=Path(args.snapshot_dir) if args.snapshot_dir else None,
        config=config,
        device=args.device,
    )

    print("\nDone!")


if __name__ == "__main__":
    main()
