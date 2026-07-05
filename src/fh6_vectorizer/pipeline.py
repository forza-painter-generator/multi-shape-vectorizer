"""
Main pipeline: ties together template generation, rendering, and optimization.

Provides a simple API for the full vectorization workflow.
"""

import json
import math
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

from .templates import (
    build_template_library,
    generate_synthetic_templates,
    load_template_library,
    save_template_library,
)
from .ste_renderer import STEVectorRenderer
from .optimizer import GradientOptimizer
from .preprocess import compute_importance_map
from .json_writer import generate_fh6_json


def load_target_image(
    path: Path,
    target_size: tuple[int, int] = (256, 256),
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Load and preprocess a target image, preserving alpha channel.

    Args:
        path: path to image file
        target_size: (H, W) to resize to
        device: torch device

    Returns:
        target: [3, H, W] RGB in [0, 1]
        alpha_mask: [1, H, W] in [0, 1], 0=transparent, 1=opaque
    """
    img = Image.open(path).convert("RGBA")
    img = img.resize((target_size[1], target_size[0]), Image.LANCZOS)
    arr = torch.from_numpy(np.array(img)).float() / 255.0
    # HWC: [R, G, B, A]
    rgb = arr[:, :, :3].permute(2, 0, 1).to(device)  # [3, H, W]
    alpha = arr[:, :, 3:4].permute(2, 0, 1).to(device)  # [1, H, W]
    return rgb, alpha


def load_config(config_path: Path) -> dict:
    """
    Load configuration from a JSON file.

    Args:
        config_path: path to a JSON config file (e.g., configs/default.json)
    Returns:
        config dict
    """
    with open(config_path, "r") as f:
        return json.load(f)


def save_output_image(tensor: torch.Tensor, path: Path, alpha_mask: torch.Tensor = None) -> None:
    """
    Save a rendered image tensor to file, with optional alpha.

    Args:
        tensor: [3, H, W] in [0, 1]
        path: output path (.png)
        alpha_mask: [1, H, W] or [H, W] in [0, 1], applied as alpha channel
    """
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = (arr.clip(0, 1) * 255).astype("uint8")
    if alpha_mask is not None:
        a = alpha_mask.detach().cpu()
        if a.dim() == 3:
            a = a.squeeze(0)
        a = (a.clip(0, 1) * 255).to(torch.uint8).numpy()
        arr = np.dstack([arr, a])
    img = Image.fromarray(arr, "RGBA" if alpha_mask is not None else "RGB")
    img.save(path)
    print(f"Saved output to {path}")


def run_pipeline(
    target_image_path: Path,
    output_path: Optional[Path] = None,
    num_shapes: int = 200,
    num_types: int = 8,
    canvas_size: tuple[int, int] = (256, 256),
    use_fh6_data: bool = False,
    fh6_vinyls_root: Optional[Path] = None,
    template_cache_path: Optional[Path] = None,
    snapshot_dir: Optional[Path] = None,
    config: Optional[dict] = None,
    device: str = "cpu",
) -> tuple[torch.Tensor, list[dict]]:
    """
    Run the full vectorization pipeline.

    Args:
        target_image_path: input image to vectorize
        output_path: where to save the final rendered image
        num_shapes: N, number of shapes to optimize
        num_types: number of template types (for synthetic templates)
        canvas_size: (H, W) rendering resolution
        use_fh6_data: whether to use real FH6 shape data
        fh6_vinyls_root: path to FH6 Vinyls/ directory
        template_cache_path: path to cache/load pre-built templates
        snapshot_dir: if set, save intermediate renders here every N steps
        config: optimizer hyperparameters
        device: "cpu" or "cuda"

    Returns:
        (final_rendered_image, loss_history)
    """
    H, W = canvas_size

    # --- Step 1: Build or load template library ---
    print("=" * 50)
    print("Step 1: Building template library...")

    if template_cache_path and Path(template_cache_path).exists():
        print(f"  Loading cached templates from {template_cache_path}")
        library = load_template_library(Path(template_cache_path), device=device)
    elif use_fh6_data and fh6_vinyls_root:
        print(f"  Building FH6 templates from {fh6_vinyls_root}")
        library = build_template_library(
            vinyls_root=Path(fh6_vinyls_root),
            device=device,
        )
    else:
        print(f"  Generating {num_types} synthetic templates")
        library = generate_synthetic_templates(
            num_types=num_types,
            device=device,
        )

    num_types_actual = library["hard"].shape[0]
    print(f"  Loaded {num_types_actual} templates: {library['names']}")

    if template_cache_path and not Path(template_cache_path).exists():
        save_template_library(library, Path(template_cache_path))

    # --- Step 2: Load target image ---
    print("\nStep 2: Loading target image...")
    target, alpha_mask = load_target_image(target_image_path, target_size=canvas_size, device=device)
    print(f"  Target size: {target.shape}, alpha range: [{alpha_mask.min():.2f}, {alpha_mask.max():.2f}]")
    has_transparency = (alpha_mask < 0.99).any().item()
    if has_transparency:
        print("  Detected transparency - transparent regions will be ignored")

    # --- Step 2.5: Compute importance map (for smart initialization) ---
    importance_map = None
    use_imp = config.get("use_importance_sampling", True) if config else True
    if use_imp:
        print("  Computing importance map...")
        target_np = target.cpu().permute(1, 2, 0).numpy()
        target_np = (target_np * 255).astype(np.uint8)
        importance_map = compute_importance_map(
            target_np,
            edge_weight=config.get("edge_weight", 0.5) if config else 0.5,
            variance_weight=config.get("variance_weight", 0.3) if config else 0.3,
            uniform_weight=config.get("uniform_weight", 0.2) if config else 0.2,
            kmeans_k=config.get("kmeans_k") if config else None,
        ).to(device)
        # Zero out importance in transparent regions
        if has_transparency:
            importance_map = importance_map * alpha_mask.squeeze(0)
        print(f"  Importance map: {importance_map.shape}")

    # --- Step 3: Initialize renderer ---
    print(f"\nStep 3: Initializing renderer with {num_shapes} shapes...")
    renderer = STEVectorRenderer(
        num_shapes=num_shapes,
        num_types=num_types_actual,
        hard_templates=library["hard"],
        soft_templates=library["soft"],
        canvas_height=H,
        canvas_width=W,
        device=device,
    )

    # --- Step 4: Run optimization ---
    print("\nStep 4: Running optimization...")
    optimizer = GradientOptimizer(
        renderer=renderer,
        target=target,
        config=config,
        importance_map=importance_map,
        alpha_mask=alpha_mask if has_transparency else None,
        device=device,
        snapshot_dir=str(snapshot_dir) if snapshot_dir else None,
        snapshot_interval=config.get("snapshot_interval", 25) if config else 25,
    )
    history = optimizer.optimize()

    # --- Step 5: Render final result ---
    print("\nStep 5: Rendering final result...")
    final = optimizer.render_final()

    # --- Step 6: Save outputs ---
    if output_path:
        # Save rendered image with alpha channel
        final_rgba = optimizer.renderer(return_alpha=True)
        # Split RGB + A
        final_rgb = final_rgba[:3]
        final_alpha = final_rgba[3:4]  # [1, H, W]
        save_output_image(final_rgb, output_path, alpha_mask=final_alpha)

        # Save FH6 JSON if requested
        if config and config.get("include_fh6_json", False):
            json_path = output_path.with_suffix(".json")
            fh6_type_map = library.get("type_map", {i: i for i in range(num_types_actual)})
            generate_fh6_json(renderer, fh6_type_map, json_path)

        # Save loss history as JSON
        history_path = output_path.with_suffix(".history.json")
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)
        print(f"Saved loss history to {history_path}")

    # Compute final metrics
    mse = torch.nn.functional.mse_loss(final, target).item()
    psnr = 10 * math.log10(1.0 / mse) if mse > 0 else float("inf")
    print(f"\nFinal metrics: MSE={mse:.6f}, PSNR={psnr:.2f} dB")

    return final, history
