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
    rendered: torch.Tensor, target: torch.Tensor,
    alpha_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Compute per-pixel error map for relocation targeting.

    Uses luminance-weighted MSE for better visual saliency.
    If alpha_mask is provided, transparent regions get zero error.

    Args:
        rendered: [3, H, W]
        target: [3, H, W]
        alpha_mask: [1, H, W] or [H, W], 0=transparent, 1=opaque

    Returns:
        error_map: [H, W] per-pixel error
    """
    diff = rendered - target
    diff = torch.nan_to_num(diff, nan=0.0, posinf=1.0, neginf=-1.0)
    lum_weights = torch.tensor([0.299, 0.587, 0.114], device=rendered.device).view(3, 1, 1)
    error = (diff * diff * lum_weights).sum(dim=0)
    error = error.clamp(min=0.0)
    if alpha_mask is not None:
        if alpha_mask.dim() == 3:
            alpha_mask = alpha_mask.squeeze(0)
        error = error * alpha_mask
    return error


def find_relocation_candidates(
    renderer: STEVectorRenderer,
    grad_history: list[dict],
    stability_threshold: float = 1.0,
    min_visibility: float = 2.0,
) -> tuple[torch.Tensor, int, int, int]:
    """
    Vinylizer-style: classify shapes by gradient stability + visibility.

    Returns (relocate_mask, n_useful, n_useless, n_unstable).
    Only relocates shapes that are BOTH stable AND useless (low visibility).
    """
    N = renderer.cx.shape[0]
    device = renderer.device

    # Compute mean gradient norm from history
    if len(grad_history) < 3:
        return (torch.zeros(N, dtype=torch.bool, device=device), N, 0, 0)

    avg_grad = torch.zeros(N, device=device)
    for h in grad_history:
        for v in h.values():
            avg_grad += v.detach()
    avg_grad /= len(grad_history)

    # Stability: low mean gradient = converged
    is_stable = avg_grad < stability_threshold

    # Visibility: use opacity × area as proxy for actual blend weight
    # (full per-pixel visibility requires renderer changes)
    with torch.no_grad():
        area_proxy = (renderer.rx.data * renderer.ry.data).sqrt()
        visibility = renderer.opacity.data * area_proxy
        # Normalize to reasonable range
        vis_max = visibility.max() + 1e-8
        visibility = visibility / vis_max * 10.0  # scale to ~[0, 10]

    n_unstable = (~is_stable).sum().item()
    n_useful = (is_stable & (visibility >= min_visibility)).sum().item()
    n_useless = (is_stable & (visibility < min_visibility)).sum().item()

    relocate_mask = is_stable & (visibility < min_visibility)

    return relocate_mask, n_useful, n_useless, n_unstable


def relocate_shapes(
    renderer: STEVectorRenderer,
    error_map: torch.Tensor,
    relocate_mask: torch.Tensor,
    num_types: int,
    target: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Vinylizer-style: z-order compact + Top-K error sampling.

    1. Remove relocated shapes, shift remaining forward
    2. Sample new positions from top-5% highest error pixels
    3. Append new shapes at end (top z-order layer)
    4. Returns new_indices mask for Phase C local optimization
    """
    if not relocate_mask.any():
        return torch.zeros_like(relocate_mask)

    H, W = error_map.shape
    N = renderer.cx.shape[0]
    device = renderer.device
    relocate_indices = torch.where(relocate_mask)[0]
    num_reloc = len(relocate_indices)

    with torch.no_grad():
        # --- Top-K error sampling (vinylizer: sample from top 5%) ---
        error_flat = error_map.flatten()
        K = max(num_reloc * 5, min(50, int(error_flat.numel() * 0.05)))
        _, top_indices = torch.topk(error_flat, K)
        rand_k = torch.randint(0, K, (num_reloc,), device=device)
        flat_idx = top_indices[rand_k]
        new_cy = (flat_idx // W).float()
        new_cx = (flat_idx % W).float()

        # New params
        new_rx = torch.rand(num_reloc, device=device) * 30 + 5
        new_ry = torch.rand(num_reloc, device=device) * 30 + 5
        new_angle = torch.rand(num_reloc, device=device) * 360
        new_opacity = torch.ones(num_reloc, device=device)  # full opacity initially
        new_type = torch.randint(0, num_types, (num_reloc,), device=device)

        if target is not None:
            new_colors = color_from_target(target, new_cx, new_cy, noise_std=0.05)
        else:
            new_colors = torch.rand(num_reloc, 3, device=device)

        # --- z-order compact: remove relocated, shift forward ---
        keep_mask = ~relocate_mask
        num_keep = keep_mask.sum().item()

        for name in ["cx", "cy", "rx", "ry", "angle", "colors", "opacity", "type_indices"]:
            param = getattr(renderer, name)
            data = param.data
            # Compact: keep shapes shift to front
            if data.ndim == 1:
                data[:num_keep] = data[keep_mask]
            else:  # colors: [N, 3]
                data[:num_keep] = data[keep_mask]

        # --- Append new shapes at end (top z-order) ---
        base = num_keep
        renderer.cx.data[base:base + num_reloc] = new_cx
        renderer.cy.data[base:base + num_reloc] = new_cy
        renderer.rx.data[base:base + num_reloc] = new_rx
        renderer.ry.data[base:base + num_reloc] = new_ry
        renderer.angle.data[base:base + num_reloc] = new_angle
        renderer.opacity.data[base:base + num_reloc] = new_opacity
        renderer.colors.data[base:base + num_reloc] = new_colors
        renderer.type_indices.data[base:base + num_reloc] = new_type

    # Return mask of new shapes (indices base..N-1)
    new_indices = torch.zeros(N, dtype=torch.bool, device=device)
    new_indices[base:base + num_reloc] = True
    return new_indices


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
        alpha_mask: Optional[torch.Tensor] = None,
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
            alpha_mask: [1, H, W] optional alpha mask (0=transparent, 1=opaque)
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

        # Alpha mask: broadcast to [1, H, W] for easy multiplication
        if alpha_mask is not None:
            if alpha_mask.dim() == 2:
                alpha_mask = alpha_mask.unsqueeze(0)
            self.alpha_mask = alpha_mask.to(device)
        else:
            self.alpha_mask = None

        # Default config
        cfg = {
            # Optimization
            "lr": 0.05,
            "color_lr_ratio": 0.5,
            "global_steps": 150,
            "local_steps": 50,
            "num_cycles": 3,
            # Vinylizer-style relocation
            "reloc_stability_threshold": 1.0,
            "reloc_min_visibility": 0.5,
            # Legacy (deprecated)
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

    def _eval_target(self) -> torch.Tensor:
        """Return the evaluation target, matching step-loss MSE formula.

        Transparent pixels use black background (same as _optimize_step).
        This ensures snapshot MSE matches the logged step MSE.
        """
        if self.alpha_mask is not None:
            bg = torch.zeros(3, 1, 1, device=self.device)
            return self.target * self.alpha_mask + bg * (1.0 - self.alpha_mask)
        return self.target

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

        # vinylizer-style alpha-aware loss:
        # Transparent pixels → target = background color (so shapes get pushed away)
        # Opaque pixels → target = real image
        if self.alpha_mask is not None:
            mask = self.alpha_mask  # [1, H, W]: 1=opaque, 0=transparent
            bg = torch.zeros(3, 1, 1, device=self.device)  # black background
            target_with_bg = self.target * mask + bg * (1.0 - mask)
            # Full MSE on all pixels — transparent regions naturally push shapes away
            loss = torch.nn.functional.mse_loss(rendered, target_with_bg)
            loss_dict["mse"] = loss.item()

            # Add L1 if configured
            l1_w = self.cfg.get("l1_weight", 0.0)
            if l1_w > 0:
                l1_val = torch.nn.functional.l1_loss(rendered, target_with_bg) * l1_w
                loss = loss + l1_val
                loss_dict["l1"] = l1_val.item()

            # Transparent-region loss: L1 for strong gradient on dark shapes
            inv_mask = 1.0 - mask
            t_l1 = ((rendered - bg) * inv_mask).abs().mean()
            t_weight = self.cfg.get("transparent_weight", 3.0)
            loss = loss + t_weight * t_l1
            loss_dict["transparent_l1"] = t_l1.item()

            # Boundary constraint (diffbmp-style): penalize shapes whose
            # center is near transparent regions — their edges likely leak
            H_m, W_m = mask.shape[1], mask.shape[2]
            center_alpha = F.grid_sample(
                mask.unsqueeze(0),
                torch.stack([
                    (self.renderer.cx / W_m) * 2 - 1,
                    (self.renderer.cy / H_m) * 2 - 1,
                ], dim=-1).unsqueeze(0).unsqueeze(0),
                mode="bilinear", padding_mode="zeros", align_corners=True,
            ).squeeze()
            # Shapes near boundary: alpha in (0.1, 0.9) → penalize large rx/ry
            boundary_mask = (center_alpha > 0.1) & (center_alpha < 0.9)
            if boundary_mask.any():
                boundary_penalty = (
                    self.renderer.rx[boundary_mask].mean() +
                    self.renderer.ry[boundary_mask].mean()
                ) * 0.001
                loss = loss + boundary_penalty
                loss_dict["boundary_penalty"] = boundary_penalty.item()

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

        # Clamp shape centers to opaque regions (if alpha mask available)
        if self.alpha_mask is not None:
            self._clamp_to_opaque()

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

    def _clamp_to_opaque(self):
        """Remove shapes fully in transparent regions. Gentle boundary handling."""
        with torch.no_grad():
            mask = self.alpha_mask  # [1, H, W]
            H, W = mask.shape[1], mask.shape[2]

            px_n = (self.renderer.cx.data / W) * 2.0 - 1.0
            py_n = (self.renderer.cy.data / H) * 2.0 - 1.0
            grid = torch.stack([px_n, py_n], dim=-1).unsqueeze(0).unsqueeze(0)
            alpha_at = F.grid_sample(
                mask.unsqueeze(0), grid,
                mode="bilinear", padding_mode="zeros", align_corners=True,
            ).squeeze()

            # Only kill shapes whose center is deep in transparent (alpha < 0.05)
            kill = alpha_at < 0.05
            if kill.any():
                self.renderer.opacity.data[kill] = 0.01

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

    def _reset_optimizer(self):
        """Reset Adam state + scheduler after relocation (z-order changed)."""
        r = self.renderer
        self.optimizer = Adam([
            {"params": [r.cx, r.cy, r.rx, r.ry, r.angle],
             "lr": self.cfg["lr"]},
            {"params": [r.colors, r.opacity],
             "lr": self.cfg["lr"] * self.cfg["color_lr_ratio"]},
        ])
        total_steps = self.cfg["num_cycles"] * (self.cfg["global_steps"] + self.cfg["local_steps"])
        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
        self.grad_history = []

    def optimize(self) -> list[dict]:
        """
        Vinylizer-style optimization with best-cycle snapshot.

        No rollback — runs all cycles, saves best MSE snapshot per cycle.
        """
        full_history = []
        num_types = self.renderer.hard_templates.shape[0]
        best_mse = float("inf")
        best_snapshot = None

        for cycle in range(self.cfg["num_cycles"]):
            print(f"\n{'='*50}")
            print(f"Cycle {cycle + 1}/{self.cfg['num_cycles']}")
            print(f"{'='*50}")

            # Phase A: Global optimization
            print(f"  Phase A: Global Adam ({self.cfg['global_steps']} steps)")
            hist_a = self.run_cycle(cycle, frozen_mask=None)
            full_history.extend(hist_a)

            # Record Phase A MSE for best-snapshot tracking
            # Use same MSE formula as step loss (transparent→black bg)
            with torch.no_grad():
                rendered = self.renderer()
                target_ref = self._eval_target()
            cycle_mse = F.mse_loss(rendered, target_ref).item()
            if cycle_mse < best_mse:
                best_mse = cycle_mse
                best_snapshot = self._snapshot_params()
                print(f"    New best MSE: {best_mse:.6f}")

            # Phase B: Scan for useless shapes
            print(f"  Phase B: Scanning for useless shapes...")
            error_map = compute_error_map(rendered, self.target, self.alpha_mask)
            relocate_mask, n_use, n_less, n_unst = find_relocation_candidates(
                self.renderer,
                self.grad_history[-20:],
                stability_threshold=self.cfg.get("reloc_stability_threshold", 1.0),
                min_visibility=self.cfg.get("reloc_min_visibility", 2.0),
            )
            num_relocated = relocate_mask.sum().item()
            print(f"    useful={n_use} useless={n_less} unstable={n_unst}")

            if num_relocated == 0:
                continue

            # Phase B execute: z-order compact + relocate
            print(f"    Relocating {num_relocated} shapes")
            new_indices = relocate_shapes(
                self.renderer, error_map, relocate_mask, num_types,
                target=self.target,
            )

            # Reset Adam state after relocation (vinylizer: z-order changed)
            self._reset_optimizer()

            # Phase C: Local optimization (frozen old shapes)
            print(f"  Phase C: Local optimization ({self.cfg['local_steps']} steps)")
            frozen_mask = ~new_indices
            hist_c = self.run_cycle(cycle, frozen_mask=frozen_mask)
            full_history.extend(hist_c)

            # Also check after Phase C — local opt often beats Phase A
            with torch.no_grad():
                rendered = self.renderer()
                target_ref = self._eval_target()
            cycle_mse = F.mse_loss(rendered, target_ref).item()
            if cycle_mse < best_mse:
                best_mse = cycle_mse
                best_snapshot = self._snapshot_params()
                print(f"    New best MSE (after Phase C): {best_mse:.6f}")

        # Restore best snapshot at end (vinylizer: final best-cycle restore)
        if best_snapshot is not None:
            print(f"\nRestoring best cycle params (MSE={best_mse:.6f})")
            self._restore_params(best_snapshot)

        return full_history

    def render_final(self) -> torch.Tensor:
        """Render the final result (no gradients)."""
        with torch.no_grad():
            return self.renderer()
