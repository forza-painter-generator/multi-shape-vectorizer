# FH6 Multi-Shape Differentiable Vectorizer

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

Automatically reconstruct any image using FH6 (Forza Horizon 6) vinyl shapes via
**differentiable rendering** — combining ideas from
[vinylizer](https://github.com/), [diffbmp](https://github.com/smhongok/diffbmp)
(CVPR 2026), and [IGS](https://github.com/KohakuBlueleaf/IGS).

## Architecture

```
Input Image → Preprocessing → Importance Map
                                    ↓
Templates (hard+soft) ──→ STE Over-Compositing Renderer ←── Adam Optimizer
                                    ↓                              ↑
                              Rendered Image ──→ MSE + Perceptual Loss
                                    ↓
                              Cyclic Relocation
                                    ↓
                              FH6 JSON Output
```

### Key Features

- **STE (Straight-Through Estimator)**: Forward renders hard/binary shapes (matching
  in-game appearance), backward uses soft/blurred shapes for gradient flow.
- **Multi-shape support**: 16 synthetic geometric templates (circle, square, star,
  heart, crescent, etc.) or real FH6 polygon data.
- **Over compositing in linear color space**: Physically correct Porter-Duff
  alpha blending with sRGB ↔ Linear conversion.
- **Cyclic relocation**: Automatically detects useless shapes and moves them to
  high-error regions (from vinylizer).
- **Rich loss functions**: MSE, L1, Huber, Grayscale, VGG Perceptual, alpha regularization.
- **Importance-map initialization**: Canny edges + color variance guide initial shape placement.
- **FH6 JSON export**: Generate files directly importable into Forza Horizon 6.
- **Tile-based rendering**: Efficient for large canvases (>256px).

## Installation

```bash
# Clone
git clone <repo-url>
cd multi-shape-vectorizer

# Install
pip install -e .

# Optional: for perceptual loss
pip install torchvision

# Optional: for plot_loss.py
pip install matplotlib
```

## Quick Start

```bash
# Basic: 200 shapes, 256×256, 3 cycles, MSE loss
python scripts/run_optimize.py -i photo.jpg -o result.png -n 200 --size 256 256

# Higher quality: 500 shapes, perceptual loss, importance sampling, FH6 JSON
python scripts/run_optimize.py -i photo.jpg -o result.png \
    -n 500 --size 512 512 --cycles 5 --perceptual \
    --l1-weight 0.1 --alpha-reg 0.01 --fh6-json

# Use JSON config file
python scripts/run_optimize.py -i photo.jpg -o result.png \
    --config-file configs/default.json

# Use real FH6 shape data
python scripts/run_optimize.py -i photo.jpg -o result.png \
    --fh6-data "path/to/Vinyls/" --template-cache templates.pt
```

## Tools

```bash
# Visualize template library
python scripts/visualize_templates.py -n 16 -o templates.png

# Plot loss curves
python scripts/plot_loss.py result.history.json -o loss_plot.png
```

## Project Structure

```
multi-shape-vectorizer/
├── configs/
│   └── default.json              # Default hyperparameters
├── src/fh6_vectorizer/
│   ├── templates.py              # Template generation (synthetic + FH6)
│   ├── ste_renderer.py           # STE Over-compositing renderer (+ tiled)
│   ├── optimizer.py              # Adam + cyclic relocation + rollback
│   ├── loss.py                   # MSE, L1, Huber, Perceptual, Grayscale, Alpha
│   ├── preprocess.py             # K-means, Canny, importance map
│   ├── json_writer.py            # FH6 JSON export
│   └── pipeline.py               # End-to-end pipeline
├── scripts/
│   ├── run_optimize.py           # CLI entry point
│   ├── plot_loss.py              # Loss curve visualization
│   └── visualize_templates.py    # Template PNG grid
├── tests/
│   ├── test_core.py              # Core unit tests
│   └── test_new_features.py      # New feature integration tests
├── TASKS.md                      # Detailed task checklist
├── IMPLEMENTATION_PLAN.md        # Full technical design document
└── pyproject.toml
```

## Key References

| Reference | Use |
|-----------|-----|
| [diffbmp](https://arxiv.org/abs/2602.22625) (CVPR 2026) | Soft rasterization via Gaussian blur, multi-shape framework |
| vinylizer | Over compositing, cyclic relocation, STE strategy |
| [IGS](https://github.com/KohakuBlueleaf/IGS) | Triton fused kernel patterns (for future GPU optimization) |
| forza-painter-fh6 | FH6 shape geometry data + import/export logic |

## License

MIT
