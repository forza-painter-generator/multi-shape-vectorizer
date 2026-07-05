"""
Basic smoke tests for the FH6 vectorizer PoC.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fh6_vectorizer.templates import (
    generate_synthetic_templates,
    TEMPLATE_SIZE,
)
from fh6_vectorizer.ste_renderer import (
    STEVectorRenderer,
    over_composite_render,
    _make_canvas_grid,
    compute_template_coords,
)
from fh6_vectorizer.loss import mse_loss, VGGFeatureExtractor


class TestTemplates:
    """Template generation tests."""

    def test_synthetic_generation(self):
        """Synthetic templates should have correct shape and value ranges."""
        lib = generate_synthetic_templates(num_types=5, device="cpu")

        assert lib["hard"].shape == (5, TEMPLATE_SIZE, TEMPLATE_SIZE)
        assert lib["soft"].shape == (5, TEMPLATE_SIZE, TEMPLATE_SIZE)
        assert len(lib["names"]) == 5

        # Hard templates are binary
        hard_vals = lib["hard"].unique()
        assert set(hard_vals.tolist()).issubset({0.0, 1.0})

        # Soft templates are continuous [0, 1]
        assert lib["soft"].min() >= 0
        assert lib["soft"].max() <= 1

        # Soft templates should have values between 0 and 1 (blurred edges)
        soft_middle = lib["soft"][0]
        assert (soft_middle > 0).sum() > 0
        assert (soft_middle < 1).sum() > 0  # At least some blurred edge pixels


class TestRenderer:
    """STE renderer tests."""

    def test_grid_creation(self):
        """Canvas grid should have correct shape and values."""
        px, py = _make_canvas_grid(64, 64)
        assert px.shape == (64, 64)
        assert py.shape == (64, 64)
        assert px[0, 0] == 0
        assert px[0, -1] == 63
        assert py[0, 0] == 0
        assert py[-1, 0] == 63

    def test_template_coords(self):
        """Template coordinate computation should map correctly."""
        px, py = _make_canvas_grid(32, 32)
        cx = torch.tensor(16.0)
        cy = torch.tensor(16.0)
        rx = torch.tensor(32.0)
        ry = torch.tensor(32.0)
        angle = torch.tensor(0.0)

        grid = compute_template_coords(px, py, cx, cy, rx, ry, angle)
        assert grid.shape == (1, 32, 32, 2)

        # Center pixel should map to near (0, 0)
        center_coords = grid[0, 16, 16]
        assert abs(center_coords[0]) < 0.1  # tx near 0
        assert abs(center_coords[1]) < 0.1  # ty near 0

    def test_basic_render(self):
        """Basic rendering should produce valid output."""
        lib = generate_synthetic_templates(num_types=4, device="cpu")
        N = 10
        renderer = STEVectorRenderer(
            num_shapes=N,
            num_types=4,
            hard_templates=lib["hard"],
            soft_templates=lib["soft"],
            canvas_height=64,
            canvas_width=64,
        )

        rendered = renderer()
        assert rendered.shape == (3, 64, 64)
        assert rendered.min() >= 0
        assert rendered.max() <= 1

    def test_render_deterministic(self):
        """Same parameters should produce same output."""
        lib = generate_synthetic_templates(num_types=4, device="cpu")
        renderer = STEVectorRenderer(
            num_shapes=5,
            num_types=4,
            hard_templates=lib["hard"],
            soft_templates=lib["soft"],
            canvas_height=32,
            canvas_width=32,
        )

        r1 = renderer().clone()
        r2 = renderer().clone()
        assert torch.allclose(r1, r2)

    def test_ste_gradient_flow(self):
        """STE should allow gradients to flow through."""
        lib = generate_synthetic_templates(num_types=4, device="cpu")
        renderer = STEVectorRenderer(
            num_shapes=5,
            num_types=4,
            hard_templates=lib["hard"],
            soft_templates=lib["soft"],
            canvas_height=32,
            canvas_width=32,
        )

        rendered = renderer()
        loss = rendered.mean()
        loss.backward()

        # All continuous params should have gradients
        assert renderer.cx.grad is not None
        assert renderer.cy.grad is not None
        assert renderer.rx.grad is not None
        assert renderer.ry.grad is not None
        assert renderer.angle.grad is not None
        assert renderer.colors.grad is not None
        assert renderer.opacity.grad is not None

        # Type indices (discrete) should NOT have gradients
        assert renderer.type_indices.grad is None


class TestLoss:
    """Loss function tests."""

    def test_mse_zero(self):
        """MSE should be zero for identical images."""
        x = torch.rand(3, 32, 32)
        loss = mse_loss(x, x)
        assert loss.item() == pytest.approx(0.0, abs=1e-6)

    def test_mse_positive(self):
        """MSE should be positive for different images."""
        x = torch.ones(3, 32, 32)
        y = torch.zeros(3, 32, 32)
        loss = mse_loss(x, y)
        assert loss.item() == pytest.approx(1.0, abs=1e-3)


class TestIntegration:
    """End-to-end integration test."""

    def test_small_optimization(self):
        """Small optimization run should complete without errors."""
        from fh6_vectorizer.pipeline import run_pipeline

        # Create a simple test image
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
                target_image_path=input_path,
                output_path=output_path,
                num_shapes=20,
                num_types=4,
                canvas_size=(64, 64),
                config={
                    "lr": 0.1,
                    "global_steps": 10,
                    "local_steps": 5,
                    "num_cycles": 2,
                    "use_perceptual_loss": False,
                },
                device="cpu",
            )

            assert final.shape == (3, 64, 64)
            assert len(history) > 0

        finally:
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
