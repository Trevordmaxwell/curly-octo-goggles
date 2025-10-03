"""Training objectives for unified model."""
from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# Optional dependency: wandb is referenced in other modules, but not here.
# Silence lint warnings about unused imports if module unavailable.


class UnifiedTrainingObjective:
    """Multi-component loss function for training the unified model."""

    def __init__(
        self,
        task_weight: float = 1.0,
        energy_weight: float = 0.1,
        convergence_weight: float = 0.05,
        stability_weight: float = 0.01,
        contraction_target: float = 0.9,
    ) -> None:
        self.w_task = task_weight
        self.w_energy = energy_weight
        self.w_conv = convergence_weight
        self.w_stab = stability_weight
        self.contraction_target = contraction_target

    def compute_loss(
        self,
        model: Any,
        batch: Tuple[torch.Tensor, torch.Tensor],
        diagnostics: Dict[str, Any],
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """Compute full training loss."""

        input_ids, target_ids = batch
        logits = diagnostics.get("logits")
        if logits is None:
            logits = model(input_ids)

        loss_task = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=-100,
        )

        solver_info = diagnostics.get("solver_info", {})
        final_energy = solver_info.get("final_energy", torch.tensor(0.0, device=loss_task.device))
        energy_components = solver_info.get("energy_components", {})

        loss_energy = self.w_energy * final_energy

        num_iterations = solver_info.get("iterations", 0)
        max_iter = getattr(model.solver, "max_iter", 1) or 1
        loss_convergence = self.w_conv * (num_iterations / max_iter)

        z_eq = diagnostics.get("z_equilibrium")
        lipschitz_const = 0.0
        if z_eq is not None:
            lipschitz_const = self._estimate_lipschitz(model, z_eq, diagnostics)

        loss_stability = self.w_stab * F.relu(
            torch.as_tensor(lipschitz_const, device=loss_task.device) - self.contraction_target
        )

        total_loss = loss_task + loss_energy + loss_convergence + loss_stability

        loss_components = {
            "total": float(total_loss.detach().item()),
            "task": float(loss_task.detach().item()),
            "energy": float(loss_energy.detach().item()) if isinstance(loss_energy, torch.Tensor) else loss_energy,
            "convergence": float(loss_convergence.detach().item())
            if isinstance(loss_convergence, torch.Tensor)
            else loss_convergence,
            "stability": float(loss_stability.detach().item())
            if isinstance(loss_stability, torch.Tensor)
            else loss_stability,
            "num_iterations": num_iterations,
            "lipschitz_constant": lipschitz_const,
            "energy_components": energy_components,
        }

        return total_loss, loss_components

    def _estimate_lipschitz(
        self,
        model: Any,
        z_equilibrium: torch.Tensor,
        diagnostics: Dict[str, Any],
        num_samples: int = 5,
    ) -> float:
        """Estimate Lipschitz constant of dynamics via sampling."""

        context = diagnostics.get("context")
        if context is None:
            return 0.0

        with torch.no_grad():
            lipschitz_estimates = []

            for _ in range(num_samples):
                epsilon = 0.01
                delta = epsilon * torch.randn_like(z_equilibrium)
                z_perturbed = z_equilibrium + delta

                f_z = model.dynamics(z_equilibrium, context, model.memory_patterns)
                f_z_pert = model.dynamics(z_perturbed, context, model.memory_patterns)

                numerator = torch.norm(f_z_pert - f_z, dim=-1).mean()
                denominator = torch.norm(delta, dim=-1).mean().clamp_min(1e-8)

                lipschitz_estimates.append((numerator / denominator).item())

        if not lipschitz_estimates:
            return 0.0

        return float(np.mean(lipschitz_estimates))
