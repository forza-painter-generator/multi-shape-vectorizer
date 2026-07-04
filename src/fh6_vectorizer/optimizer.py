"""
Gradient-based optimizer with cyclic relocation.

Implements the optimization loop from vinylizer:
  - Phase A: Global Adam optimization
  - Phase B: Scan for "useless" shapes (low gradient + low opacity)
  - Phase C: Relocate useless shapes to high-error regions, freeze old shapes

Reference: vinylizer/src/core/optimizer.cpp gradient_optimize()
"""

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from .loss import combined_loss, mse_loss, VGGFeatureExtractor
from .ste_renderer import STEVectorRenderer


def compute_error_map(
    rendered: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Compute per-pixel error map for relocation targeting.

    Uses luminance-weighted MSE for better visual saliency.

    Args:
        rendered: [3, H, W]
        target: [3, H, W]

    Returns:
        error_map: [H, W] per-pixel error
    """
    diff = rendered - target
    # Luminance-weighted: (0.299R + 0.587G + 0.114B)
    lum_weights = torch.tensor([0.299, 0.587, 0.114], device=rendered.device).view(3, 1, 1)
    error = (diff * diff * lum_weights).sum(dim=0)
    return error


def find_relocation_candidates(
    renderer: STEVectorRenderer,
    grad_history: list[dict],
    relocation_fraction: float = 0.15,
    min_opacity: float = 0.05,
    grad_threshold_quantile: float = 0.25,
) -> torch.Tensor:
    """
    Identify shapes that are "useless" and should be relocated.

    Criteria:
      1. Low gradient norm (converged, not contributing much)
      2. Low opacity (barely visible or fully occluded)

    Args:
        renderer: the STE renderer
        grad_history: list of gradient norm dicts from recent steps
        relocation_fraction: fraction of total shapes to relocate
        min_opacity: opacity below which shape is considered useless
        grad_threshold_quantile: quantile for gradient norm threshold

    Returns:
        Boolean mask [N] — True for shapes to relocate
    """
    N = renderer.cx.shape[0]
    num_to_relocate = max(1, int(N * relocation_fraction))

    # Compute average gradient norm from history
    if len(grad_history) > 0:
        avg_grad = torch.zeros(N, device=renderer.device)
        for h in grad_history:
            for k, v in h.items():
                avg_grad += v.detach()
        avg_grad /= len(grad_history)
    else:
        avg_grad = torch.ones(N, device=renderer.device)

    # Score: lower is more "useless"
    # Combine gradient norm and opacity
    opacity = renderer.opacity.data
    grad_score = avg_grad / (avg_grad.max() + 1e-8)
    opacity_score = opacity / (opacity.max() + 1e-8)
    usefulness = grad_score * 0.5 + opacity_score * 0.5

    # Select lowest-scoring shapes
    _, indices = torch.sort(usefulness)
    relocate_mask = torch.zeros(N, dtype=torch.bool, device=renderer.device)
    relocate_mask[indices[:num_to_relocate]] = True

    # Also relocate shapes with very low opacity
    relocate_mask = relocate_mask | (opacity < min_opacity)

    return relocate_mask


def relocate_shapes(
    renderer: STEVectorRenderer,
    error_map: torch.Tensor,
    relocate_mask: torch.Tensor,
    num_types: int,
):
    """
    Relocate useless shapes to high-error positions.

    For each relocated shape:
      1. Sample new position from error_map (weighted by error)
      2. Randomize scale and rotation
      3. Try each template type at the new position

    Args:
        renderer: the STE renderer
        error_map: [H, W] per-pixel error
        relocate_mask: [N] boolean, which shapes to relocate
        num_types: number of available template types
    """
    if not relocate_mask.any():
        return

    H, W = error_map.shape
    device = renderer.device

    # Flatten error map for weighted sampling
    error_flat = error_map.flatten()
    error_flat = error_flat / (error_flat.sum() + 1e-8)

    relocate_indices = torch.where(relocate_mask)[0]
    num_reloc = len(relocate_indices)

    # Sample new positions
    flat_indices = torch.multinomial(error_flat, num_reloc, replacement=True)
    new_cy = (flat_indices // W).float()
    new_cx = (flat_indices % W).float()

    with torch.no_grad():
        renderer.cx.data[relocate_mask] = new_cx.to(device)
        renderer.cy.data[relocate_mask] = new_cy.to(device)
        renderer.rx.data[relocate_mask] = torch.rand(num_reloc, device=device) * 40 + 10
        renderer.ry.data[relocate_mask] = torch.rand(num_reloc, device=device) * 40 + 10
        renderer.angle.data[relocate_mask] = torch.rand(num_reloc, device=device) * 360
        renderer.colors.data[relocate_mask] = torch.rand(num_reloc, 3, device=device)
        renderer.opacity.data[relocate_mask] = torch.rand(num_reloc, device=device) * 0.3 + 0.3

        # Randomize template types
        renderer.type_indices.data[relocate_mask] = torch.randint(
            0, num_types, (num_reloc,), device=device
        )


class GradientOptimizer:
    """
    Main optimizer orchestrating the optimization pipeline.

    Pipeline:
      for each cycle:
        Phase A: Global Adam optimization (all shapes)
        Phase B: Scan for useless shapes
        Phase C: Relocation + local optimization (frozen old shapes)
    """

    def __init__(
        self,
        renderer: STEVectorRenderer,
        target: torch.Tensor,
        config: Optional[dict] = None,
        device: str = "cpu",
    ):
        """
        Args:
            renderer: STEVectorRenderer with optimizable parameters
            target: [3, H, W] target image in [0, 1]
            config: optimization hyperparameters
            device: torch device
        """
        self.renderer = renderer
        self.target = target.to(device)
        self.device = device

        # Default config
        cfg = {
            "lr": 0.05,
            "global_steps": 150,      # Phase A steps per cycle
            "local_steps": 50,         # Phase C steps per cycle
            "num_cycles": 3,           # Total A+B+C cycles
            "relocation_fraction": 0.15,
            "use_perceptual_loss": False,
            "perceptual_weight": 0.2,
            "mse_weight": 1.0,
        }
        if config:
            cfg.update(config)
        self.cfg = cfg

        # Optimizer: separate param groups for geometry vs color
        self.optimizer = Adam([
            {"params": [renderer.cx, renderer.cy, renderer.rx, renderer.ry, renderer.angle],
             "lr": cfg["lr"]},
            {"params": [renderer.colors, renderer.opacity],
             "lr": cfg["lr"] * 0.5},
        ])

        # LR scheduler
        total_steps = cfg["num_cycles"] * (cfg["global_steps"] + cfg["local_steps"])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        # Perceptual loss (optional)
        self.vgg = None
        if cfg["use_perceptual_loss"]:
            self.vgg = VGGFeatureExtractor(device=device)

        self.grad_history = []

    def _compute_loss(
        self, rendered: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Compute combined loss."""
        if self.vgg is not None:
            return combined_loss(
                rendered.unsqueeze(0),
                self.target.unsqueeze(0),
                self.vgg,
                mse_weight=self.cfg["mse_weight"],
                perceptual_weight=self.cfg["perceptual_weight"],
            )
        else:
            mse = mse_loss(rendered, self.target)
            return mse, {"mse": mse.item(), "perceptual": 0.0}

    def _optimize_step(self, frozen_mask: Optional[torch.Tensor] = None) -> dict:
        """
        Single optimization step.

        Args:
            frozen_mask: [N] bool, True = shape parameters are frozen

        Returns:
            loss dict
        """
        self.optimizer.zero_grad()

        rendered = self.renderer()
        loss, loss_dict = self._compute_loss(rendered)
        loss.backward()

        # Record gradient norms for relocation detection
        grad_norms = {}
        for name in ["cx", "cy", "rx", "ry", "angle", "colors", "opacity"]:
            param = getattr(self.renderer, name)
            if param.grad is not None:
                grad_norms[name] = param.grad.norm(dim=-1) if param.grad.ndim > 1 else param.grad.abs()

        # Zero out gradients for frozen shapes
        if frozen_mask is not None:
            for name in ["cx", "cy", "rx", "ry", "angle", "colors", "opacity"]:
                param = getattr(self.renderer, name)
                if param.grad is not None:
                    if param.grad.ndim > 1:
                        param.grad[frozen_mask] = 0
                    else:
                        param.grad[frozen_mask] = 0

        self.optimizer.step()
        self.scheduler.step()
        self.renderer.clamp_params()

        self.grad_history.append(grad_norms)
        if len(self.grad_history) > 50:
            self.grad_history.pop(0)

        return loss_dict

    def run_cycle(
        self,
        cycle_idx: int,
        frozen_mask: Optional[torch.Tensor] = None,
    ) -> list[dict]:
        """
        Run one optimization cycle (Phase A or Phase C).

        Args:
            cycle_idx: cycle number (for logging)
            frozen_mask: optional frozen shape mask for Phase C

        Returns:
            list of per-step loss dicts
        """
        steps = self.cfg["local_steps"] if frozen_mask is not None else self.cfg["global_steps"]
        phase = "C (local)" if frozen_mask is not None else "A (global)"
        history = []

        for step in range(steps):
            loss_dict = self._optimize_step(frozen_mask)
            history.append(loss_dict)

            if step % 25 == 0 or step == steps - 1:
                print(
                    f"  Cycle {cycle_idx} {phase} [{step:3d}/{steps}] "
                    f"MSE={loss_dict['mse']:.6f}"
                    + (f" Perc={loss_dict['perceptual']:.6f}" if loss_dict['perceptual'] else "")
                )

        return history

    def optimize(self) -> list[dict]:
        """
        Run full optimization pipeline.

        Returns:
            Full loss history across all cycles
        """
        full_history = []
        num_types = self.renderer.hard_templates.shape[0]

        for cycle in range(self.cfg["num_cycles"]):
            print(f"\n{'='*50}")
            print(f"Cycle {cycle + 1}/{self.cfg['num_cycles']}")
            print(f"{'='*50}")

            # Phase A: Global optimization
            print(f"  Phase A: Global Adam ({self.cfg['global_steps']} steps)")
            hist_a = self.run_cycle(cycle, frozen_mask=None)
            full_history.extend(hist_a)

            # Phase B: Scan and relocate
            print(f"  Phase B: Scanning for useless shapes...")
            with torch.no_grad():
                rendered = self.renderer()
            error_map = compute_error_map(rendered, self.target)
            relocate_mask = find_relocation_candidates(
                self.renderer,
                self.grad_history[-20:],  # Last 20 steps
                relocation_fraction=self.cfg["relocation_fraction"],
            )
            num_relocated = relocate_mask.sum().item()
            print(f"    Relocating {num_relocated} shapes")

            relocate_shapes(
                self.renderer, error_map, relocate_mask, num_types
            )

            # Phase C: Local optimization (frozen old shapes)
            if num_relocated > 0:
                print(f"  Phase C: Local optimization ({self.cfg['local_steps']} steps)")
                hist_c = self.run_cycle(cycle, frozen_mask=~relocate_mask)
                full_history.extend(hist_c)

        return full_history

    def render_final(self) -> torch.Tensor:
        """Render the final result (no gradients)."""
        with torch.no_grad():
            return self.renderer()
