# Changelog

## [0.2.0] — 2026-07-04

### Added
- **16 synthetic templates**: pentagon, hexagon, crescent, heart, arrow, droplet, chevron, star4
- **sRGB ↔ Linear color space conversion** in Over compositing for physically correct blending
- **L1 / Huber / Grayscale MSE / Alpha regularization** loss functions
- **Importance-map-based shape initialization** (Canny edges + color variance)
- **Color sampling from target image** for smart initial colors
- **Smart type selection** during relocation (try all template types, pick best)
- **FH6 JSON output** (`json_writer.py`) with Y-flip, angle reversal, uint8 colors
- **Image preprocessing module** (`preprocess.py`): K-means, Canny, importance maps
- **Tile-based rendering** with AABB culling — auto-enabled for canvases ≥256px
- **Relocation MSE rollback** — reverts bad relocations
- **Loss history logging** — auto-saved as `.history.json`
- **Intermediate snapshots** — save renders every N steps during optimization
- **`scripts/plot_loss.py`** — matplotlib visualizations from history JSON
- **`scripts/visualize_templates.py`** — template PNG grid export
- **`scripts/profile.py`** — performance profiling with torch.profiler
- **`scripts/benchmark.py`** — benchmark across canvas sizes and shape counts
- **`configs/default.json`** — JSON-based configuration with 6 config groups
- **23 unit tests** across `test_core.py`, `test_new_features.py`, `test_extra.py`
- **`README.md`** — full project documentation

### Changed
- Over compositing now operates in linear color space
- `GradientOptimizer` supports all new loss weights and config options
- `STEVectorRenderer.forward()` auto-selects tiled vs non-tiled rendering

## [0.1.0] — 2026-07-03

### Added
- Core STE differentiable rendering pipeline
- Porter-Duff Over compositing with cyclic relocation
- 8 synthetic geometric templates (circle, square, triangle, ellipse, diamond, star, cross, ring)
- MSE + VGG Perceptual loss functions
- Adam optimizer with cosine annealing
- FH6 template library builder (from forza-painter-fh6 data)
- CLI entry point (`scripts/run_optimize.py`)
- Basic unit test suite
