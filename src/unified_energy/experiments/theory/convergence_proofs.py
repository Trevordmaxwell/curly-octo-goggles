"""Empirical validation utilities for the unified equilibrium formulation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch
from torch import Tensor, nn


@dataclass(slots=True)
class ValidationConfig:
    """Configuration controlling convergence validation sweeps."""

    context_length: int = 8
    batch_size: int = 1
    lipschitz_threshold: float = 1.0
    energy_tolerance: float = 1e-4
    stability_radius: float = 0.1
    max_perturbation_trials: int = 5


class ConvergenceValidator:
    """Run sanity checks that mirror the theoretical guarantees in practice."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: Optional[torch.device | str] = None,
        config: Optional[ValidationConfig] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model = model.to(self.device) if hasattr(model, "to") else model
        self.config = config or ValidationConfig()
        self._d_model = self._infer_hidden_dim()

        if not hasattr(self.model, "dynamics"):
            msg = "Model must expose a dynamics(z, context, memory_patterns) method"
            raise ValueError(msg)
        if not hasattr(self.model, "energy_fn"):
            msg = "Model must expose an energy_fn compatible with UnifiedEnergyFunction"
            raise ValueError(msg)
        if not hasattr(self.model, "solver"):
            msg = "Model must expose a solver with a solve(...) method"
            raise ValueError(msg)
        if not hasattr(self.model, "memory_patterns"):
            msg = "Model must provide a memory_patterns tensor"
            raise ValueError(msg)

    # ------------------------------------------------------------------
    # Core helpers
    # ------------------------------------------------------------------
    def _infer_hidden_dim(self) -> int:
        if hasattr(self.model, "d_model"):
            return int(getattr(self.model, "d_model"))
        patterns = getattr(self.model, "memory_patterns", None)
        if isinstance(patterns, Tensor):
            return int(patterns.size(-1))
        raise ValueError("Unable to infer model hidden dimension")

    def _sample_context(self, batch_size: Optional[int] = None) -> Tensor:
        length = self.config.context_length
        batch = batch_size or self.config.batch_size
        return torch.randn(batch, length, self._d_model, device=self.device)

    # ------------------------------------------------------------------
    # Validation routines
    # ------------------------------------------------------------------
    @torch.no_grad()
    def test_contraction_property(self, num_samples: int = 32) -> Dict[str, float]:
        """Estimate the Lipschitz constant of the dynamics."""

        lipschitz_values = []
        for _ in range(num_samples):
            z1 = torch.randn(self.config.batch_size, self._d_model, device=self.device)
            z2 = torch.randn_like(z1)
            context = self._sample_context()
            f_z1 = self.model.dynamics(z1, context, self.model.memory_patterns)
            f_z2 = self.model.dynamics(z2, context, self.model.memory_patterns)
            numerator = torch.linalg.vector_norm(f_z1 - f_z2)
            denominator = torch.linalg.vector_norm(z1 - z2).clamp_min(1e-8)
            lipschitz_values.append((numerator / denominator).item())

        max_lip = max(lipschitz_values)
        mean_lip = sum(lipschitz_values) / len(lipschitz_values)
        contractive_ratio = sum(1.0 for v in lipschitz_values if v < 1.0) / len(lipschitz_values)
        return {
            "mean_lipschitz": mean_lip,
            "max_lipschitz": max_lip,
            "contractive_fraction": contractive_ratio,
            "is_contractive": max_lip < self.config.lipschitz_threshold,
        }

    @torch.no_grad()
    def test_energy_descent(
        self, num_trajectories: int = 16, num_steps: int = 20
    ) -> Dict[str, float]:
        """Check that the unified energy decreases along iterative updates."""

        violations = 0
        for _ in range(num_trajectories):
            z = torch.randn(self.config.batch_size, self._d_model, device=self.device)
            context = self._sample_context()
            last_energy: Optional[float] = None
            for _ in range(num_steps):
                z_next = self.model.dynamics(z, context, self.model.memory_patterns)
                energy, _ = self.model.energy_fn(z, z_next, self.model.memory_patterns)
                energy_value = float(energy.detach())
                if (
                    last_energy is not None
                    and energy_value > last_energy + self.config.energy_tolerance
                ):
                    violations += 1
                    break
                if torch.linalg.vector_norm(z_next - z).item() < self.config.energy_tolerance:
                    break
                last_energy = energy_value
                z = z_next
        violation_rate = violations / max(1, num_trajectories)
        return {
            "violation_rate": violation_rate,
            "monotonic_descent": violation_rate < 0.2,
        }

    @torch.no_grad()
    def test_fixed_point_stability(
        self, num_fixed_points: int = 8, num_perturbations: Optional[int] = None
    ) -> Dict[str, float]:
        """Probe stability of equilibria by perturbing converged states."""

        if num_perturbations is None:
            num_perturbations = self.config.max_perturbation_trials
        stable = 0
        attempted = 0
        for _ in range(num_fixed_points):
            z_init = torch.randn(self.config.batch_size, self._d_model, device=self.device)
            context = self._sample_context()
            z_eq, info = self.model.solver.solve(z_init, context, self.model.memory_patterns)
            if not info.get("converged", False):
                continue
            attempted += 1
            equilibrium_stable = True
            for _ in range(num_perturbations):
                noise = self.config.stability_radius * torch.randn_like(z_eq)
                z_perturbed = z_eq + noise
                z_return, info_return = self.model.solver.solve(
                    z_perturbed, context, self.model.memory_patterns
                )
                if not info_return.get("converged", False):
                    equilibrium_stable = False
                    break
                delta = torch.linalg.vector_norm(z_return - z_eq).item()
                if delta > self.config.stability_radius * 5:
                    equilibrium_stable = False
                    break
            if equilibrium_stable:
                stable += 1
        stability_rate = stable / max(1, attempted)
        return {
            "tested_equilibria": attempted,
            "stability_rate": stability_rate,
            "is_stable": stability_rate > 0.7,
        }

    @torch.no_grad()
    def test_lyapunov_function(self, num_samples: int = 16, horizon: int = 15) -> Dict[str, float]:
        """Validate that the energy acts as a Lyapunov function for the dynamics."""

        satisfied = 0
        for _ in range(num_samples):
            z = torch.randn(self.config.batch_size, self._d_model, device=self.device)
            context = self._sample_context()
            is_valid = True
            prev_energy: Optional[float] = None
            for _ in range(horizon):
                z_next = self.model.dynamics(z, context, self.model.memory_patterns)
                energy, _ = self.model.energy_fn(z, z_next, self.model.memory_patterns)
                energy_val = float(energy.detach())
                if (
                    prev_energy is not None
                    and energy_val > prev_energy + self.config.energy_tolerance
                ):
                    is_valid = False
                    break
                if torch.linalg.vector_norm(z_next - z).item() < self.config.energy_tolerance:
                    break
                prev_energy = energy_val
                z = z_next
            if is_valid:
                satisfied += 1
        lyapunov_rate = satisfied / max(1, num_samples)
        return {
            "lyapunov_rate": lyapunov_rate,
            "is_lyapunov": lyapunov_rate > 0.8,
        }

    # ------------------------------------------------------------------
    # Aggregate
    # ------------------------------------------------------------------
    def run_all_tests(self) -> Dict[str, Dict[str, float]]:
        """Execute the entire validation battery and collate metrics."""

        results = {
            "contraction": self.test_contraction_property(),
            "energy_descent": self.test_energy_descent(),
            "stability": self.test_fixed_point_stability(),
            "lyapunov": self.test_lyapunov_function(),
        }
        all_pass = all(
            [
                bool(results["contraction"].get("is_contractive", False)),
                bool(results["energy_descent"].get("monotonic_descent", False)),
                bool(results["stability"].get("is_stable", False)),
                bool(results["lyapunov"].get("is_lyapunov", False)),
            ]
        )
        results["summary"] = {"all_tests_passed": bool(all_pass)}
        return results
