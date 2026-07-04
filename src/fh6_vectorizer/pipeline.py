"""
Main pipeline: ties together template generation, rendering, and optimization.

Provides a simple API for the full vectorization workflow.
"""

from pathlib import Path
from typing import Optional

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


def load_target_image(
    path: Path,
    target_size: tuple[int, int] = (256, 256),
    device: str = "cpu",
) -> torch.Tensor:
    """
    Load and preprocess a target image.

    Args:
        path: path to image file
        target_size: (H, W) to resize to
        device: torch device

    Returns:
        target: [3, H, W] in [0, 1]
    """
    img = Image.open(path).convert("RGB")
    img = img.resize((target_size[1], target_size[0]), Image.LANCZOS)
    arr = torch.from_numpy(__import__("numpy").array(img)).float() / 255.0
    # HWC → CHW
    arr = arr.permute(2, 0, 1)
    return arr.to(device)


def save_output_image(tensor: torch.Tensor, path: Path) -> None:
    """
    Save a rendered image tensor to file.

    Args:
        tensor: [3, H, W] in [0, 1]
        path: output path (.png)
    """
    arr = tensor.detach().cpu().permute(1, 2, 0).numpy()
    arr = (arr.clip(0, 1) * 255).astype("uint8")
    img = Image.fromarray(arr)
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
    target = load_target_image(target_image_path, target_size=canvas_size, device=device)
    print(f"  Target size: {target.shape}")

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
        device=device,
    )
    history = optimizer.optimize()

    # --- Step 5: Render final result ---
    print("\nStep 5: Rendering final result...")
    final = optimizer.render_final()

    # --- Step 6: Save output ---
    if output_path:
        save_output_image(final, output_path)

    # Compute final metrics
    mse = torch.nn.functional.mse_loss(final, target).item()
    psnr = 10 * __import__("math").log10(1.0 / mse) if mse > 0 else float("inf")
    print(f"\nFinal metrics: MSE={mse:.6f}, PSNR={psnr:.2f} dB")

    return final, history
