#!/usr/bin/env python
"""
Plot loss curves from a history JSON file.

Usage:
    python scripts/plot_loss.py output.history.json [--output plot.png]
"""

import argparse
import json
import sys
from pathlib import Path


def plot_loss(history_path: Path, output_path: Path = None):
    """Plot loss curves from a JSON history file."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("Error: matplotlib is required. Install with: pip install matplotlib")
        sys.exit(1)

    with open(history_path) as f:
        history = json.load(f)

    if not history:
        print("Error: empty history file")
        sys.exit(1)

    # Extract loss components
    keys = [k for k in history[0].keys()]
    series = {k: [step.get(k, 0) for step in history] for k in keys}
    x = list(range(len(history)))

    fig, axes = plt.subplots(len(keys), 1, figsize=(12, 3 * len(keys)), sharex=True)
    if len(keys) == 1:
        axes = [axes]

    colors = plt.cm.tab10.colors
    for ax, (key, color) in zip(axes, zip(keys, colors)):
        ax.plot(x, series[key], color=color, linewidth=0.8, alpha=0.9)
        ax.set_ylabel(key)
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{key} over optimization steps")

    fig.suptitle(f"Loss curves — {history_path.name}", fontsize=14)
    fig.tight_layout()

    out = output_path or history_path.with_suffix(".png")
    fig.savefig(out, dpi=120)
    print(f"Saved plot to {out}")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot loss curves from history JSON")
    parser.add_argument("history", type=Path, help="Path to .history.json file")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output PNG path")
    args = parser.parse_args()

    if not args.history.exists():
        print(f"Error: file not found: {args.history}")
        sys.exit(1)

    plot_loss(args.history, args.output)
