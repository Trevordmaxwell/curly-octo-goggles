"""Hybrid solver that enforces fixed-point and energy optimality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .beta_schedule import BetaAnnealingSchedule

@dataclass(slots=True)
class SolverConfig:
    """Configuration parameters for the unified equilibrium solver."""

    max_iter: int = 50
    tol_fixedpoint: float = 1e-3
    tol_energy: float = 1e-3
    solver_type: str = "alternating"
    anderson_memory: int = 5
    learning_rate: float = 1e-2

    def __post_init__(self) -> None:  # pragma: no cover - straightforward checks
        if self.max_iter <= 0:
            raise ValueError("max_iter must be positive")
        if self.tol_fixedpoint <= 0:
            raise ValueError("tol_fixedpoint must be positive")
        if self.tol_energy <= 0:
            raise ValueError("tol_energy must be positive")
        if self.anderson_memory <= 0:
            raise ValueError("anderson_memory must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.solver_type not in {"alternating", "simultaneous", "cascade"}:
            raise ValueError("solver_type must be one of 'alternating', 'simultaneous', 'cascade'")


class UnifiedEquilibriumSolver:
    """Solve for states satisfying both fixed-point and energy optimality."""

    def __init__(
        self,
        dynamics_fn,
        energy_fn,
        config: SolverConfig | None = None,
        *,
        beta_schedule: BetaAnnealingSchedule | None = None,
    ) -> None:
        self.dynamics = dynamics_fn
        self.energy = energy_fn
        self.config = config or SolverConfig()
        self.beta_schedule = beta_schedule

    def fixed_point_step(
        self,
        z: Tensor,
        x_context: Tensor,
        memory_patterns: Tensor,
        history: Sequence[Tuple[Tensor, Tensor]],
        *,
        beta: float | None = None,
    ) -> Tensor:
        """Perform an Anderson-accelerated fixed-point iteration."""

        f_z = self.dynamics(z, x_context, memory_patterns, beta=beta)
        if len(history) < 2:
            return f_z
        residuals = [f_prev - z_prev for z_prev, f_prev in history[-self.config.anderson_memory :]]
        coeffs = self._anderson_coefficients(residuals)
        if not coeffs:
            return f_z
        accelerated = torch.zeros_like(f_z)
        recent_history = history[-len(coeffs) :]
        for weight, (_, f_prev) in zip(coeffs, recent_history):
            accelerated = accelerated + weight * f_prev
        return accelerated

    def energy_descent_step(
        self,
        z: Tensor,
        x_context: Tensor,
        memory_patterns: Tensor,
        *,
        beta: float | None = None,
    ) -> Tensor:
        """Gradient descent step on the unified energy."""

        z_next = self.dynamics(z, x_context, memory_patterns, beta=beta)
        grad = self.energy.energy_gradient(
            z, memory_patterns, z_next=z_next, beta=beta
        )
        return z - self.config.learning_rate * grad

    def alternating_solve(
        self,
        z_init: Tensor,
        x_context: Tensor,
        memory_patterns: Tensor,
    ) -> Tuple[Tensor, Dict[str, object]]:
        """Alternate between fixed-point iterations and energy descent."""

        z = z_init
        history: List[Tuple[Tensor, Tensor]] = []
        energy_history: List[float] = []
        info: Dict[str, object] = {}
        last_energy_value: Tensor | None = None
        last_components: Dict[str, float] | None = None
        last_fp_residual = torch.tensor(0.0, device=z.device)
        last_energy_grad = torch.tensor(0.0, device=z.device)
        for iteration in range(self.config.max_iter):
            beta_value = self._beta_for_iteration(iteration + 1)
            if iteration % 2 == 0:
                z_next = self.fixed_point_step(
                    z,
                    x_context,
                    memory_patterns,
                    history,
                    beta=beta_value,
                )
            else:
                z_next = self.energy_descent_step(
                    z, x_context, memory_patterns, beta=beta_value
                )
            f_z = self.dynamics(z_next, x_context, memory_patterns, beta=beta_value)
            energy_value, components = self.energy(
                z_next, f_z, memory_patterns, beta=beta_value
            )
            energy_history.append(float(energy_value.detach()))
            history.append((z_next.detach(), f_z.detach()))
            fp_residual = torch.norm(f_z - z_next)
            energy_grad = self.energy.energy_gradient(
                z_next, memory_patterns, z_next=f_z, beta=beta_value
            )
            energy_grad_norm = torch.norm(energy_grad)
            last_energy_value = energy_value
            last_components = {k: float(v.detach()) for k, v in components.items()}
            last_fp_residual = fp_residual
            last_energy_grad = energy_grad_norm
            if self._has_converged(fp_residual.item(), energy_grad_norm.item(), energy_history):
                info = {
                    "converged": True,
                    "iterations": iteration + 1,
                    "final_fp_residual": fp_residual.item(),
                    "final_energy_grad": energy_grad_norm.item(),
                    "final_energy": float(energy_value.detach()),
                    "energy_components": {k: float(v.detach()) for k, v in components.items()},
                    "energy_history": energy_history,
                }
                return z_next, info
            z = z_next
        info = {
            "converged": False,
            "iterations": self.config.max_iter,
            "energy_history": energy_history,
        }
        if last_energy_value is not None:
            info.update(
                {
                    "final_energy": float(last_energy_value.detach()),
                    "final_fp_residual": float(last_fp_residual.detach()),
                    "final_energy_grad": float(last_energy_grad.detach()),
                    "energy_components": last_components or {},
                }
            )
        return z, info

    def simultaneous_solve(
        self,
        z_init: Tensor,
        x_context: Tensor,
        memory_patterns: Tensor,
    ) -> Tuple[Tensor, Dict[str, object]]:
        """Jointly optimise fixed-point residual and energy."""

        z = z_init.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([z], lr=self.config.learning_rate)
        energy_history: List[float] = []
        last_energy_value: Tensor | None = None
        last_components: Dict[str, float] | None = None
        last_fp_residual = torch.tensor(0.0, device=z.device)
        last_energy_grad = torch.tensor(0.0, device=z.device)
        for iteration in range(self.config.max_iter):
            beta_value = self._beta_for_iteration(iteration + 1)
            optimizer.zero_grad()
            f_z = self.dynamics(z, x_context, memory_patterns, beta=beta_value)
            fp_loss = torch.sum((z - f_z) ** 2)
            energy_value, components = self.energy(
                z, f_z, memory_patterns, beta=beta_value
            )
            total_loss = fp_loss + energy_value
            total_loss.backward()
            optimizer.step()
            with torch.no_grad():
                fp_residual = torch.norm(z - f_z)
                energy_grad = self.energy.energy_gradient(
                    z, memory_patterns, z_next=f_z, beta=beta_value
                )
                energy_grad_norm = torch.norm(energy_grad)
                energy_history.append(float(energy_value.detach()))
                last_energy_value = energy_value.detach()
                last_components = {k: float(v.detach()) for k, v in components.items()}
                last_fp_residual = fp_residual.detach()
                last_energy_grad = energy_grad_norm.detach()
                if self._has_converged(fp_residual.item(), energy_grad_norm.item(), energy_history):
                    return z.detach(), {
                        "converged": True,
                        "iterations": iteration + 1,
                        "final_fp_residual": fp_residual.item(),
                        "final_energy_grad": energy_grad_norm.item(),
                        "final_energy": float(energy_value.detach()),
                        "energy_components": {k: float(v.detach()) for k, v in components.items()},
                        "energy_history": energy_history,
                    }
        info = {
            "converged": False,
            "iterations": self.config.max_iter,
            "energy_history": energy_history,
        }
        if last_energy_value is not None:
            info.update(
                {
                    "final_energy": float(last_energy_value),
                    "final_fp_residual": float(last_fp_residual),
                    "final_energy_grad": float(last_energy_grad),
                    "energy_components": last_components or {},
                }
            )
        return z.detach(), info

    def cascade_solve(
        self,
        z_init: Tensor,
        x_context: Tensor,
        memory_patterns: Tensor,
    ) -> Tuple[Tensor, Dict[str, object]]:
        """First reach a fixed point quickly, then refine with energy minimisation."""

        z = z_init
        for iteration in range(max(1, self.config.max_iter // 2)):
            beta_value = self._beta_for_iteration(iteration + 1)
            z_next = self.dynamics(z, x_context, memory_patterns, beta=beta_value)
            if torch.norm(z_next - z) < self.config.tol_fixedpoint:
                z = z_next
                break
            z = z_next
        z = z.detach().requires_grad_(True)
        optimizer = torch.optim.LBFGS([z], max_iter=25, line_search_fn="strong_wolfe")

        def closure() -> Tensor:
            optimizer.zero_grad()
            beta_value = self._beta_for_iteration(1)
            f_z = self.dynamics(z, x_context, memory_patterns, beta=beta_value)
            energy_value, _ = self.energy(z, f_z, memory_patterns, beta=beta_value)
            energy_value.backward()
            return energy_value

        optimizer.step(closure)
        with torch.no_grad():
            beta_value = self._beta_for_iteration(1)
            f_z = self.dynamics(z, x_context, memory_patterns, beta=beta_value)
            energy_value, components = self.energy(
                z, f_z, memory_patterns, beta=beta_value
            )
            fp_residual = torch.norm(f_z - z)
            energy_grad = self.energy.energy_gradient(
                z, memory_patterns, z_next=f_z, beta=beta_value
            )
            energy_grad_norm = torch.norm(energy_grad)
        return z.detach(), {
            "converged": fp_residual.item() < self.config.tol_fixedpoint
            and energy_grad_norm.item() < self.config.tol_energy,
            "iterations": self.config.max_iter,
            "final_fp_residual": fp_residual.item(),
            "final_energy_grad": energy_grad_norm.item(),
            "final_energy": float(energy_value.detach()),
            "energy_components": {k: float(v.detach()) for k, v in components.items()},
            "energy_history": [],
        }

    def solve(
        self,
        z_init: Tensor,
        x_context: Tensor,
        memory_patterns: Tensor,
    ) -> Tuple[Tensor, Dict[str, object]]:
        """Dispatch to the configured solver mode."""

        mode = self.config.solver_type
        if mode == "alternating":
            return self.alternating_solve(z_init, x_context, memory_patterns)
        if mode == "simultaneous":
            return self.simultaneous_solve(z_init, x_context, memory_patterns)
        if mode == "cascade":
            return self.cascade_solve(z_init, x_context, memory_patterns)
        raise ValueError(f"Unknown solver type: {mode}")

    def _has_converged(
        self,
        fp_residual: float,
        energy_grad_norm: float,
        energy_history: Sequence[float],
    ) -> bool:
        if fp_residual > self.config.tol_fixedpoint:
            return False
        if energy_grad_norm > self.config.tol_energy:
            return False
        if len(energy_history) < 5:
            return False
        recent = torch.tensor(energy_history[-5:])
        return torch.std(recent).item() < 1e-4

    def _anderson_coefficients(self, residuals: Sequence[Tensor]) -> List[float]:
        if not residuals:
            return []
        stacked = torch.stack(residuals)
        flat = stacked.reshape(stacked.size(0), -1)
        gram = flat @ flat.t()
        eye = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        ones = torch.ones(gram.size(0), device=gram.device, dtype=gram.dtype)
        coeffs = torch.linalg.solve(gram + 1e-6 * eye, ones)
        coeffs = coeffs / coeffs.sum()
        return coeffs.tolist()

    def _beta_for_iteration(self, iteration: int) -> Optional[float]:
        if self.beta_schedule is None:
            return None
        total = self.config.max_iter
        return self.beta_schedule.value(iteration, total_steps=total)
