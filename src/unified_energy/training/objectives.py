"""Training objectives for the unified Mamba-Hopfield-DEQ model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(slots=True)
class ObjectiveConfig:
    """Weights for the composite training objective."""

    task_weight: float = 1.0
    energy_weight: float = 0.1
    convergence_weight: float = 0.05
    stability_weight: float = 0.01
    contraction_target: float = 0.9


class UnifiedTrainingObjective:
    """Combine task loss with energy- and convergence-aware regularisers."""

    def __init__(self, config: ObjectiveConfig | None = None) -> None:
        self.config = config or ObjectiveConfig()

    def compute_loss(
        self,
        model,
        batch: Tuple[Tensor, Tensor],
        diagnostics: Dict[str, object],
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Return the scalar loss and a dictionary of human-readable metrics."""

        input_ids, target_ids = batch
        logits = diagnostics.get("logits")
        if logits is None:
            logits = model(input_ids)

        task_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids.view(-1),
            ignore_index=-100,
        )
        solver_info: Dict[str, object] = diagnostics.get("solver_info", {})
        context = diagnostics.get("context")
        z_equilibrium = diagnostics.get("z_equilibrium")
        energy_components_metrics: Dict[str, float] = {}
        if isinstance(context, Tensor) and isinstance(z_equilibrium, Tensor):
            f_z = model.dynamics(z_equilibrium, context, model.memory_patterns)
            energy_total, energy_components = model.energy_fn(
                z_equilibrium,
                f_z,
                model.memory_patterns,
            )
            energy_loss = self.config.energy_weight * energy_total
            energy_components_metrics = {
                key: float(value.detach()) for key, value in energy_components.items()
            }
        else:
            final_energy = self._extract_final_energy(solver_info)
            energy_term = torch.as_tensor(
                final_energy,
                dtype=logits.dtype,
                device=logits.device,
            )
            energy_loss = self.config.energy_weight * energy_term

        iterations = float(solver_info.get("iterations", 0) or 0)
        max_iter = float(getattr(model.solver.config, "max_iter", 1))
        convergence_term = torch.as_tensor(
            iterations / max(max_iter, 1.0),
            dtype=logits.dtype,
            device=logits.device,
        )
        convergence_loss = self.config.convergence_weight * convergence_term

        lipschitz_tensor = self._estimate_lipschitz(model, diagnostics, dtype=logits.dtype)
        stability_loss = self.config.stability_weight * F.relu(
            lipschitz_tensor - self.config.contraction_target
        )

        total_loss = (
            self.config.task_weight * task_loss
            + energy_loss
            + convergence_loss
            + stability_loss
        )

        components = {
            "total": float(total_loss.detach()),
            "task": float(task_loss.detach()),
            "energy": float(energy_loss.detach()),
            "convergence": float(convergence_loss.detach()),
            "stability": float(stability_loss.detach()),
            "num_iterations": iterations,
            "lipschitz_constant": float(lipschitz_tensor.detach()),
        }

        if energy_components_metrics:
            components["energy_components"] = energy_components_metrics
        else:
            energy_components = solver_info.get("energy_components")
            if isinstance(energy_components, dict):
                components["energy_components"] = {
                    key: float(value) if not isinstance(value, float) else value
                    for key, value in energy_components.items()
                }
        return total_loss, components

    def _extract_final_energy(self, solver_info: Dict[str, object]) -> float:
        if "final_energy" in solver_info:
            value = solver_info["final_energy"]
            if isinstance(value, Tensor):
                return float(value.detach())
            return float(value)
        history = solver_info.get("energy_history")
        if isinstance(history, (list, tuple)) and history:
            return float(history[-1])
        return 0.0

    def _estimate_lipschitz(
        self,
        model,
        diagnostics: Dict[str, object],
        *,
        num_samples: int = 3,
        epsilon: float = 1e-2,
        dtype: torch.dtype | None = None,
    ) -> Tensor:
        context = diagnostics.get("context")
        z_equilibrium = diagnostics.get("z_equilibrium")
        if not isinstance(context, Tensor) or not isinstance(z_equilibrium, Tensor):
            device = getattr(model, "memory_patterns", torch.empty(0)).device if hasattr(model, "memory_patterns") else "cpu"
            base_dtype = dtype or torch.float32
            return torch.zeros((), device=device, dtype=base_dtype)
        estimates: list[Tensor] = []
        for _ in range(num_samples):
            delta = epsilon * torch.randn_like(z_equilibrium)
            z_perturbed = z_equilibrium + delta
            f_z = model.dynamics(z_equilibrium, context, model.memory_patterns)
            f_z_pert = model.dynamics(z_perturbed, context, model.memory_patterns)
            numerator = torch.norm(f_z_pert - f_z, dim=-1)
            denominator = torch.norm(delta, dim=-1).clamp_min(1e-8)
            estimate = torch.mean(numerator / denominator)
            estimates.append(estimate)
        if not estimates:
            return torch.zeros_like(z_equilibrium.mean())
        return torch.stack(estimates).mean()


__all__ = ["ObjectiveConfig", "UnifiedTrainingObjective"]
