"""
Gradient-based optimizer with cyclic relocation.

Implements the optimization loop from vinylizer:
  - Phase A: Global Adam optimization
  - Phase B: Scan for "useless" shapes (low gradient + low opacity)
  - Phase C: Relocate useless shapes to high-error regions, freeze old shapes

Features:
  - Importance-map-weighted shape initialization
  - Smart color initialization from target image
  - Smart type selection (try all types at error positions)
  - Alpha/opacity regularization
  - Configurable loss composition

Reference: vinylizer/src/core/optimizer.cpp gradient_optimize()
"""

import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

from .loss import (
    combined_loss,
    mse_loss,
    l1_loss,
    huber_loss,
    grayscale_mse_loss,
    alpha_regularization,
    VGGFeatureExtractor,
)
from .ste_renderer import STEVectorRenderer, over_composite_render
from .preprocess import (
    compute_importance_map,
    importance_weighted_sample,
    color_from_target,
)


def _flatten_config(config: dict, target: dict, prefix: str = ""):
    """Flatten nested config dict into flat target dict (e.g., from JSON)."""
    for key, value in config.items():
        full_key = f"{prefix}{key}"
        if isinstance(value, dict) and not key.startswith("_"):
            _flatten_config(value, target, f"{full_key}_")
        else:
            target[key] = value


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
    # Handle NaN (can happen with extreme parameter values)
    diff = torch.nan_to_num(diff, nan=0.0, posinf=1.0, neginf=-1.0)
    # Luminance-weighted: (0.299R + 0.587G + 0.114B)
    lum_weights = torch.tensor([0.299, 0.587, 0.114], device=rendered.device).view(3, 1, 1)
    error = (diff * diff * lum_weights).sum(dim=0)
    # Clamp to non-negative
    error = error.clamp(min=0.0)
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
    target: Optional[torch.Tensor] = None,
    smart_type_selection: bool = False,
):
    """
    Relocate useless shapes to high-error positions.

    For each relocated shape:
      1. Sample new position from error_map (weighted by error)
      2. Randomize scale and rotation
      3. Optionally: try each template type at the new position (smart selection)
      4. Optionally: sample colors from target image

    Args:
        renderer: the STE renderer
        error_map: [H, W] per-pixel error
        relocate_mask: [N] boolean, which shapes to relocate
        num_types: number of available template types
        target: [3, H, W] target image (for smart type selection + color init)
        smart_type_selection: if True, try all types and pick the best
    """
    if not relocate_mask.any():
        return

    H, W = error_map.shape
    device = renderer.device

    # Flatten error map for weighted sampling
    error_flat = error_map.flatten()
    error_flat = error_flat.clamp(min=0)  # Ensure non-negative
    error_sum = error_flat.sum()
    if error_sum < 1e-8:
        # All errors are zero — uniform sampling
        error_flat = torch.ones_like(error_flat)
        error_sum = error_flat.sum()
    error_flat = error_flat / error_sum

    relocate_indices = torch.where(relocate_mask)[0]
    num_reloc = len(relocate_indices)

    # Sample new positions
    flat_indices = torch.multinomial(error_flat, num_reloc, replacement=True)
    new_cy = (flat_indices // W).float().to(device)
    new_cx = (flat_indices % W).float().to(device)

    with torch.no_grad():
        renderer.cx.data[relocate_mask] = new_cx
        renderer.cy.data[relocate_mask] = new_cy
        renderer.rx.data[relocate_mask] = torch.rand(num_reloc, device=device) * 30 + 8
        renderer.ry.data[relocate_mask] = torch.rand(num_reloc, device=device) * 30 + 8
        renderer.angle.data[relocate_mask] = torch.rand(num_reloc, device=device) * 360
        renderer.opacity.data[relocate_mask] = torch.rand(num_reloc, device=device) * 0.3 + 0.3

        # Smart color init: sample colors from target at new positions
        if target is not None:
            new_colors = color_from_target(target, new_cx, new_cy, noise_std=0.05)
            renderer.colors.data[relocate_mask] = new_colors.to(device)
        else:
            renderer.colors.data[relocate_mask] = torch.rand(num_reloc, 3, device=device)

        # Smart type selection: try all types, pick the one with lowest local MSE
        if smart_type_selection and target is not None:
            _select_best_types(
                renderer, relocate_mask, target, num_types, new_cx, new_cy, device
            )
        else:
            renderer.type_indices.data[relocate_mask] = torch.randint(
                0, num_types, (num_reloc,), device=device
            )


def _select_best_types(
    renderer: STEVectorRenderer,
    relocate_mask: torch.Tensor,
    target: torch.Tensor,
    num_types: int,
    positions_cx: torch.Tensor,
    positions_cy: torch.Tensor,
    device: str,
    patch_size: int = 32,
):
    """
    For each relocated shape, try all template types and pick the best one
    based on local MSE against the target image.

    This is an expensive but effective way to choose template types.

    Args:
        renderer: STEVectorRenderer
        relocate_mask: [N] bool mask
        target: [3, H, W] target image
        num_types: number of template types
        positions_cx, positions_cy: [num_reloc] new positions
        device: torch device
        patch_size: size of local patch to compare
    """
    relocate_indices = torch.where(relocate_mask)[0]
    num_reloc = len(relocate_indices)
    H, W = target.shape[1], target.shape[2]
    half = patch_size // 2

    for j, global_idx in enumerate(relocate_indices):
        cx = int(positions_cx[j].item())
        cy = int(positions_cy[j].item())

        # Extract local target patch
        y0 = max(0, cy - half)
        y1 = min(H, cy + half)
        x0 = max(0, cx - half)
        x1 = min(W, cx + half)
        target_patch = target[:, y0:y1, x0:x1]  # [3, ph, pw]

        best_type = 0
        best_mse = float("inf")

        # Temporarily set position
        orig_cx = renderer.cx.data[global_idx].clone()
        orig_cy = renderer.cy.data[global_idx].clone()
        renderer.cx.data[global_idx] = float(cx)
        renderer.cy.data[global_idx] = float(cy)

        for t in range(num_types):
            renderer.type_indices.data[global_idx] = t
            with torch.no_grad():
                rendered = renderer()
                rendered_patch = rendered[:, y0:y1, x0:x1]
                mse = F.mse_loss(rendered_patch, target_patch).item()
                if mse < best_mse:
                    best_mse = mse
                    best_type = t

        # Restore position (unchanged) and set best type
        renderer.cx.data[global_idx] = orig_cx
        renderer.cy.data[global_idx] = orig_cy
        renderer.type_indices.data[global_idx] = best_type


class GradientOptimizer:
    """
    Main optimizer orchestrating the optimization pipeline.

    Pipeline:
      for each cycle:
        Phase A: Global Adam optimization (all shapes)
        Phase B: Scan for useless shapes
        Phase C: Relocation + local optimization (frozen old shapes)

    Initialization:
      - Importance-map-weighted position sampling (if enabled)
      - Color sampling from target image (if enabled)
    """

    def __init__(
        self,
        renderer: STEVectorRenderer,
        target: torch.Tensor,
        config: Optional[dict] = None,
        importance_map: Optional[torch.Tensor] = None,
        device: str = "cpu",
        snapshot_dir: Optional[str] = None,
        snapshot_interval: int = 50,
    ):
        """
        Args:
            renderer: STEVectorRenderer with optimizable parameters
            target: [3, H, W] target image in [0, 1]
            config: optimization hyperparameters (see configs/default.json)
            importance_map: [H, W] optional pre-computed importance map
            device: torch device
            snapshot_dir: if set, save intermediate renders to this dir every N steps
            snapshot_interval: save snapshot every N global steps
        """
        self.renderer = renderer
        self.target = target.to(device)
        self.device = device
        self.snapshot_dir = Path(snapshot_dir) if snapshot_dir else None
        self.snapshot_interval = snapshot_interval
        self._global_step_counter = 0

        # Default config
        cfg = {
            # Optimization
            "lr": 0.05,
            "color_lr_ratio": 0.5,
            "global_steps": 150,
            "local_steps": 50,
            "num_cycles": 3,
            "relocation_fraction": 0.15,
            # Loss
            "use_perceptual_loss": False,
            "mse_weight": 1.0,
            "perceptual_weight": 0.2,
            "l1_weight": 0.0,
            "huber_weight": 0.0,
            "grayscale_weight": 0.0,
            "alpha_reg_weight": 0.0,
            "alpha_target_mean": 0.5,
            # Init
            "use_importance_sampling": True,
            "smart_color_init": True,
            "smart_type_selection": False,
        }
        if config:
            # Flatten nested config
            _flatten_config(config, cfg)
        self.cfg = cfg

        # --- Initialize shapes if importance map is provided ---
        if importance_map is not None and cfg.get("use_importance_sampling", False):
            self._init_from_importance(importance_map)
        if cfg.get("smart_color_init", False):
            self._init_colors_from_target()

        # Optimizer: separate param groups for geometry vs color
        self.optimizer = Adam([
            {"params": [renderer.cx, renderer.cy, renderer.rx, renderer.ry, renderer.angle],
             "lr": cfg["lr"]},
            {"params": [renderer.colors, renderer.opacity],
             "lr": cfg["lr"] * cfg["color_lr_ratio"]},
        ])

        # LR scheduler
        total_steps = cfg["num_cycles"] * (cfg["global_steps"] + cfg["local_steps"])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)

        # Perceptual loss (optional)
        self.vgg = None
        if cfg["use_perceptual_loss"]:
            self.vgg = VGGFeatureExtractor(device=device)

        self.grad_history = []

    def _init_from_importance(self, importance_map: torch.Tensor):
        """Initialize shape positions by weighted sampling from importance map."""
        N = self.renderer.cx.shape[0]
        cx_new, cy_new = importance_weighted_sample(importance_map, N)
        with torch.no_grad():
            self.renderer.cx.data = cx_new.to(self.device)
            self.renderer.cy.data = cy_new.to(self.device)

    def _init_colors_from_target(self):
        """Initialize shape colors by sampling from target image."""
        with torch.no_grad():
            new_colors = color_from_target(
                self.target, self.renderer.cx.data, self.renderer.cy.data,
                noise_std=0.05,
            )
            self.renderer.colors.data = new_colors.to(self.device)

    def _compute_loss(
        self, rendered: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """Compute combined loss with all configured components."""
        loss_dict = {}

        if self.vgg is not None:
            total, loss_dict = combined_loss(
                rendered.unsqueeze(0),
                self.target.unsqueeze(0),
                self.vgg,
                mse_weight=self.cfg.get("mse_weight", 1.0),
                perceptual_weight=self.cfg.get("perceptual_weight", 0.2),
                l1_weight=self.cfg.get("l1_weight", 0.0),
                huber_weight=self.cfg.get("huber_weight", 0.0),
                grayscale_weight=self.cfg.get("grayscale_weight", 0.0),
            )
            return total, loss_dict

        # Build loss from components, ensuring total always has grad_fn
        mse_w = self.cfg.get("mse_weight", 1.0)
        l1_w = self.cfg.get("l1_weight", 0.0)
        huber_w = self.cfg.get("huber_weight", 0.0)
        gray_w = self.cfg.get("grayscale_weight", 0.0)

        # Start with at least MSE to guarantee a differentiable path
        total = mse_w * mse_loss(rendered, self.target)
        loss_dict["mse"] = total.item() / max(mse_w, 1e-8)

        if l1_w > 0:
            l1_val = l1_w * l1_loss(rendered, self.target)
            total = total + l1_val
            loss_dict["l1"] = l1_val.item()

        if huber_w > 0:
            hub_val = huber_w * huber_loss(rendered, self.target)
            total = total + hub_val
            loss_dict["huber"] = hub_val.item()

        if gray_w > 0:
            gray_val = gray_w * grayscale_mse_loss(rendered, self.target)
            total = total + gray_val
            loss_dict["grayscale"] = gray_val.item()

        loss_dict.setdefault("perceptual", 0.0)

        # Alpha regularization
        alpha_w = self.cfg.get("alpha_reg_weight", 0.0)
        if alpha_w > 0:
            alpha_val = alpha_regularization(
                self.renderer.opacity,
                target_mean=self.cfg.get("alpha_target_mean", 0.5),
            )
            total = total + alpha_w * alpha_val
            loss_dict["alpha_reg"] = alpha_val.item()

        return total, loss_dict

    def _optimize_step(self, frozen_mask: Optional[torch.Tensor] = None) -> dict:
        """
        Single optimization step.

        Args:
            frozen_mask: [N] bool, True = shape parameters are frozen

        Returns:
            loss dict
        """
        self.optimizer.zero_grad()

        # Ensure params require grad after in-place ops (clamp, relocate)
        for name in ["cx", "cy", "rx", "ry", "angle", "colors", "opacity"]:
            p = getattr(self.renderer, name)
            if not p.requires_grad:
                p.requires_grad_(True)

        rendered = self.renderer()
        loss, loss_dict = self._compute_loss(rendered)

        # Safety check
        if loss.grad_fn is None:
            raise RuntimeError(
                f"Loss has no grad_fn. rendered.grad_fn={rendered.grad_fn}"
            )

        loss.backward()

        # NaN protection: clip gradients and fix NaN param values
        for name in ["cx", "cy", "rx", "ry", "angle", "colors", "opacity"]:
            param = getattr(self.renderer, name)
            if param.grad is not None:
                # Replace NaN/Inf gradients with zeros
                if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
                    param.grad = torch.nan_to_num(
                        param.grad, nan=0.0, posinf=1.0, neginf=-1.0
                    )
                    param.grad.clamp_(-10.0, 10.0)

        # Fix NaN parameter values (shouldn't happen but just in case)
        for name in ["cx", "cy", "rx", "ry", "angle", "colors", "opacity"]:
            param = getattr(self.renderer, name)
            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                param.data = torch.nan_to_num(
                    param.data, nan=0.0, posinf=1.0, neginf=0.0
                )

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

        # Save intermediate snapshot if configured
        if (
            frozen_mask is None
            and self.snapshot_dir is not None
            and self._global_step_counter % self.snapshot_interval == 0
        ):
            self._save_snapshot(self._global_step_counter, loss_dict)

        self._global_step_counter += 1
        return loss_dict

    def _save_snapshot(self, step: int, loss_dict: dict):
        """Save an intermediate render snapshot to disk."""
        from PIL import Image
        import numpy as np

        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            rendered = self.renderer()
        arr = rendered.detach().cpu().permute(1, 2, 0).numpy()
        arr = (arr.clip(0, 1) * 255).astype(np.uint8)
        path = self.snapshot_dir / f"step_{step:05d}_mse_{loss_dict.get('mse', 0):.6f}.png"
        Image.fromarray(arr).save(path)

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
                parts = [f"MSE={loss_dict.get('mse', 0):.6f}"]
                if loss_dict.get('perceptual', 0):
                    parts.append(f"Perc={loss_dict['perceptual']:.6f}")
                if loss_dict.get('l1', 0):
                    parts.append(f"L1={loss_dict['l1']:.6f}")
                if loss_dict.get('alpha_reg', 0):
                    parts.append(f"α={loss_dict['alpha_reg']:.6f}")
                print(
                    f"  Cycle {cycle_idx} {phase} [{step:3d}/{steps}] "
                    + " ".join(parts)
                )

        return history

    def _snapshot_params(self) -> dict:
        """Save a snapshot of all optimizable parameters."""
        names = ["cx", "cy", "rx", "ry", "angle", "colors", "opacity", "type_indices"]
        return {n: getattr(self.renderer, n).data.clone() for n in names}

    def _restore_params(self, snapshot: dict):
        """Restore parameters from a snapshot."""
        with torch.no_grad():
            for n, v in snapshot.items():
                getattr(self.renderer, n).data.copy_(v)

    def optimize(self) -> list[dict]:
        """
        Run full optimization pipeline.

        Returns:
            Full loss history across all cycles
        """
        full_history = []
        num_types = self.renderer.hard_templates.shape[0]
        rollback_threshold = self.cfg.get("relocation_rollback_factor", 1.5)

        for cycle in range(self.cfg["num_cycles"]):
            print(f"\n{'='*50}")
            print(f"Cycle {cycle + 1}/{self.cfg['num_cycles']}")
            print(f"{'='*50}")

            # Phase A: Global optimization
            print(f"  Phase A: Global Adam ({self.cfg['global_steps']} steps)")
            hist_a = self.run_cycle(cycle, frozen_mask=None)
            full_history.extend(hist_a)

            # Record pre-relocation state
            with torch.no_grad():
                pre_reloc_render = self.renderer()
            pre_reloc_mse = F.mse_loss(pre_reloc_render, self.target).item()
            pre_snapshot = self._snapshot_params()

            # Phase B: Scan and relocate
            print(f"  Phase B: Scanning for useless shapes...")
            error_map = compute_error_map(pre_reloc_render, self.target)
            relocate_mask = find_relocation_candidates(
                self.renderer,
                self.grad_history[-20:],
                relocation_fraction=self.cfg["relocation_fraction"],
            )
            num_relocated = relocate_mask.sum().item()
            print(f"    Relocating {num_relocated} shapes")

            relocate_shapes(
                self.renderer, error_map, relocate_mask, num_types,
                target=self.target,
                smart_type_selection=self.cfg.get("smart_type_selection", False),
            )

            # Phase C: Local optimization (frozen old shapes)
            if num_relocated > 0:
                print(f"  Phase C: Local optimization ({self.cfg['local_steps']} steps)")
                hist_c = self.run_cycle(cycle, frozen_mask=~relocate_mask)
                full_history.extend(hist_c)

                # Rollback check: if Phase C made things significantly worse, revert
                with torch.no_grad():
                    post_c_render = self.renderer()
                post_c_mse = F.mse_loss(post_c_render, self.target).item()

                if post_c_mse > pre_reloc_mse * rollback_threshold:
                    print(
                        f"    ⚠ MSE degraded ({pre_reloc_mse:.6f} → {post_c_mse:.6f}), "
                        f"rolling back relocation"
                    )
                    self._restore_params(pre_snapshot)
                    # Remove Phase C history entries
                    full_history = full_history[:-len(hist_c)]
                else:
                    print(
                        f"    MSE: {pre_reloc_mse:.6f} → {post_c_mse:.6f}"
                    )

        return full_history

    def render_final(self) -> torch.Tensor:
        """Render the final result (no gradients)."""
        with torch.no_grad():
            return self.renderer()
