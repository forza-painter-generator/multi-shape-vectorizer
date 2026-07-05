"""Quick validation of all new PoC features."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from fh6_vectorizer.templates import generate_synthetic_templates
from fh6_vectorizer.loss import (
    srgb_to_linear, linear_to_srgb,
    l1_loss, huber_loss, grayscale_mse_loss, alpha_regularization,
)
from fh6_vectorizer.preprocess import (
    compute_importance_map, importance_weighted_sample, color_from_target,
)
from fh6_vectorizer.json_writer import shape_params_to_fh6
from fh6_vectorizer.ste_renderer import STEVectorRenderer
from fh6_vectorizer.optimizer import _flatten_config


def test_all():
    # 1: 5 gradient templates
    print("Test 1: 5 gradient templates...")
    lib = generate_synthetic_templates(num_types=5)
    assert lib["hard"].shape[0] == 5
    assert lib["soft"].shape[0] == 5
    assert "gradient_ellipse1" in lib["names"]
    assert "gradient_rect1" in lib["names"]
    print(f"  PASS — {lib['names']}")

    # 2: sRGB roundtrip
    print("Test 2: sRGB ↔ Linear roundtrip...")
    x = torch.rand(100)
    x_round = linear_to_srgb(srgb_to_linear(x))
    assert torch.allclose(x, x_round, atol=0.02)
    print("  PASS")

    # 3: New loss functions
    print("Test 3: New loss functions...")
    a = torch.rand(3, 32, 32)
    assert l1_loss(a, a).item() < 1e-5
    assert huber_loss(a, a).item() < 1e-5
    assert grayscale_mse_loss(a, a).item() < 1e-5
    assert alpha_regularization(torch.ones(10) * 0.5).item() < 1e-5
    print("  PASS")

    # 4: Importance map
    print("Test 4: Importance map...")
    img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    imp = compute_importance_map(img)
    assert imp.shape == (64, 64)
    assert imp.min() >= 0 and imp.max() <= 1
    cx, cy = importance_weighted_sample(imp, 20)
    assert cx.shape == (20,) and cy.shape == (20,)
    print("  PASS")

    # 5: Color from target
    print("Test 5: Color from target...")
    target = torch.rand(3, 64, 64)
    cx = torch.tensor([16.0, 32.0, 48.0])
    cy = torch.tensor([16.0, 32.0, 48.0])
    colors = color_from_target(target, cx, cy, noise_std=0.0)
    assert colors.shape == (3, 3)
    assert colors.min() >= 0 and colors.max() <= 1
    print("  PASS")

    # 6: FH6 JSON
    print("Test 6: FH6 JSON writer...")
    shapes = shape_params_to_fh6(
        cx=torch.tensor([100.0, 200.0]),
        cy=torch.tensor([150.0, 250.0]),
        rx=torch.tensor([30.0, 40.0]),
        ry=torch.tensor([30.0, 40.0]),
        angle=torch.tensor([45.0, 90.0]),
        colors=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        opacity=torch.tensor([1.0, 0.5]),
        type_indices=torch.tensor([0, 1]),
        type_map={0: 1048677, 1: 1048678},
    )
    assert len(shapes) == 2
    assert shapes[0]["type"] == 1048677
    assert shapes[0]["color"] == [255, 0, 0, 255]
    assert shapes[1]["color"][0:3] == [0, 255, 0]
    assert shapes[1]["color"][3] in (127, 128)  # 0.5 * 255 = 127.5 → round
    print("  PASS")

    # 7: Flatten config
    print("Test 7: Flatten config...")
    nested = {"optimization": {"lr": 0.05}, "loss": {"mse_weight": 1.0}}
    flat = {}
    _flatten_config(nested, flat)
    assert flat.get("lr") == 0.05
    assert flat.get("mse_weight") == 1.0
    print("  PASS")

    # 8: Render with sRGB-Linear
    print("Test 8: Render with sRGB-Linear...")
    renderer = STEVectorRenderer(
        num_shapes=5, num_types=4,
        hard_templates=lib["hard"][:4], soft_templates=lib["soft"][:4],
        canvas_height=32, canvas_width=32,
    )
    rendered = renderer()
    assert rendered.shape == (3, 32, 32)
    assert rendered.min() >= 0 and rendered.max() <= 1
    print("  PASS")

    # 9: End-to-end with new features
    print("Test 9: End-to-end with importance map...")
    from fh6_vectorizer.pipeline import run_pipeline
    import tempfile
    from PIL import Image

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        img = Image.new("RGB", (64, 64), color=(128, 0, 0))
        img.save(f.name)
        input_path = Path(f.name)

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f_out:
        output_path = Path(f_out.name)

    try:
        final, history = run_pipeline(
            target_image_path=input_path, output_path=output_path,
            num_shapes=20, num_types=8, canvas_size=(64, 64),
            config={
                "lr": 0.1, "global_steps": 10, "local_steps": 5,
                "num_cycles": 2, "use_perceptual_loss": False,
                "use_importance_sampling": True,
                "smart_color_init": True,
                "l1_weight": 0.1,
            },
            device="cpu",
        )
        assert final.shape == (3, 64, 64)
        assert len(history) > 0
        # Check history file was written
        hist_path = output_path.with_suffix(".history.json")
        assert hist_path.exists(), "History file not written"
        print("  PASS")
    finally:
        input_path.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)

    print()
    print("All 9 tests passed!")


if __name__ == "__main__":
    test_all()
