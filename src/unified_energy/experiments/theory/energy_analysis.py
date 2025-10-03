"""Energy landscape interrogation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn


@dataclass(slots=True)
class LandscapeConfig:
    """Configuration parameters for landscape sweeps."""

    context_length: int = 8
    grid_resolution: int = 21
    exploration_radius: float = 1.5
    trajectory_steps: int = 20
    basin_resolution: int = 21
    basin_radius: float = 2.0


class EnergyLandscapeAnalyzer:
    """Numerically probe the unified model's energy landscape."""

    def __init__(
        self,
        model: nn.Module,
        *,
        device: Optional[torch.device | str] = None,
        config: Optional[LandscapeConfig] = None,
    ) -> None:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        self.model = model.to(self.device) if hasattr(model, "to") else model
        if not hasattr(self.model, "memory_patterns"):
            msg = "Model must expose a memory_patterns tensor"
            raise ValueError(msg)
        if not hasattr(self.model, "dynamics"):
            msg = "Model must provide a dynamics(z, context, memory_patterns) method"
            raise ValueError(msg)
        if not hasattr(self.model, "energy_fn"):
            msg = "Model must expose an energy_fn"
            raise ValueError(msg)
        if not hasattr(self.model, "solver"):
            msg = "Model must expose a solver with a solve(...) method"
            raise ValueError(msg)
        self.config = config or LandscapeConfig()
        self._d_model = self._infer_hidden_dim()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _infer_hidden_dim(self) -> int:
        if hasattr(self.model, "d_model"):
            return int(getattr(self.model, "d_model"))
        patterns = getattr(self.model, "memory_patterns", None)
        if isinstance(patterns, Tensor):
            return int(patterns.size(-1))
        raise ValueError("Unable to infer model hidden dimension")

    def _sample_context(self, batch_size: int = 1) -> Tensor:
        return torch.randn(
            batch_size,
            self.config.context_length,
            self._d_model,
            device=self.device,
        )

    def _principal_directions(self) -> Tuple[Tensor, Tensor]:
        patterns = self.model.memory_patterns.detach().to(self.device)
        if patterns.numel() == 0:
            raise ValueError("Memory patterns are empty; unable to derive principal directions")
        centered = patterns - patterns.mean(dim=0, keepdim=True)
        cov = centered.t() @ centered / max(1, centered.size(0) - 1)
        eigvals, eigvecs = torch.linalg.eigh(cov)
        v1 = eigvecs[:, -1]
        v2 = eigvecs[:, -2] if eigvecs.size(1) > 1 else torch.randn_like(v1)
        return v1 / v1.norm(), v2 / v2.norm()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def visualize_2d_slice(
        self,
        z_equilibrium: Tensor,
        context: Optional[Tensor] = None,
        *,
        basis_vectors: Optional[Tuple[Tensor, Tensor]] = None,
        resolution: Optional[int] = None,
        radius: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Evaluate energy and residuals on a 2D plane around an equilibrium."""

        if z_equilibrium.ndim != 2:
            raise ValueError("z_equilibrium must have shape (batch, d_model)")
        if z_equilibrium.size(0) != 1:
            raise ValueError("Only single-equilibrium visualisation is supported")
        resolution = resolution or self.config.grid_resolution
        radius = radius or self.config.exploration_radius
        if context is None:
            context = self._sample_context()
        if basis_vectors is None:
            basis_vectors = self._principal_directions()
        v1, v2 = (vec.to(self.device) for vec in basis_vectors)
        v1 = v1 / v1.norm().clamp_min(1e-8)
        v2 = v2 / v2.norm().clamp_min(1e-8)
        alphas = torch.linspace(-radius, radius, resolution, device=self.device)
        betas = torch.linspace(-radius, radius, resolution, device=self.device)
        energy_grid = torch.zeros(resolution, resolution, device=self.device)
        residual_grid = torch.zeros_like(energy_grid)
        base = z_equilibrium[0]
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                point = base + alpha * v1 + beta * v2
                point = point.unsqueeze(0)
                z_next = self.model.dynamics(point, context, self.model.memory_patterns)
                energy, _ = self.model.energy_fn(point, z_next, self.model.memory_patterns)
                energy_grid[i, j] = energy.detach()
                residual_grid[i, j] = torch.linalg.vector_norm(z_next - point)
        return {
            "alpha": alphas.cpu().numpy(),
            "beta": betas.cpu().numpy(),
            "energy": energy_grid.cpu().numpy(),
            "residual": residual_grid.cpu().numpy(),
        }

    @torch.no_grad()
    def visualize_convergence_trajectories(
        self,
        num_trajectories: int = 8,
        steps: Optional[int] = None,
    ) -> Dict[str, np.ndarray]:
        """Collect convergence trajectories and provide a 2D projection."""

        steps = steps or self.config.trajectory_steps
        trajectories: list[np.ndarray] = []
        energies: list[np.ndarray] = []
        for _ in range(num_trajectories):
            z = torch.randn(1, self._d_model, device=self.device)
            context = self._sample_context()
            points = []
            energy_values = []
            for _ in range(steps):
                points.append(z.detach().cpu().numpy())
                z_next = self.model.dynamics(z, context, self.model.memory_patterns)
                energy, _ = self.model.energy_fn(z, z_next, self.model.memory_patterns)
                energy_values.append(float(energy.detach()))
                if torch.linalg.vector_norm(z_next - z).item() < 1e-4:
                    z = z_next
                    break
                z = z_next
            trajectories.append(np.concatenate(points, axis=0))
            energies.append(np.array(energy_values))
        stacked = np.concatenate(trajectories, axis=0)
        # Principal component projection
        stacked_centered = stacked - stacked.mean(axis=0, keepdims=True)
        u, s, vh = np.linalg.svd(stacked_centered, full_matrices=False)
        components = vh[:2]
        projected: list[np.ndarray] = []
        cursor = 0
        for traj in trajectories:
            length = traj.shape[0]
            segment = stacked_centered[cursor : cursor + length] @ components.T
            projected.append(segment)
            cursor += length
        return {
            "trajectories_2d": projected,
            "energy_traces": energies,
            "components": components,
        }

    @torch.no_grad()
    def visualize_basin_of_attraction(
        self,
        z_equilibrium: Tensor,
        context: Optional[Tensor] = None,
        *,
        resolution: Optional[int] = None,
        radius: Optional[float] = None,
    ) -> Dict[str, np.ndarray]:
        """Sample a neighbourhood and record convergence success."""

        if z_equilibrium.ndim != 2 or z_equilibrium.size(0) != 1:
            raise ValueError("z_equilibrium must be of shape (1, d_model)")
        resolution = resolution or self.config.basin_resolution
        radius = radius or self.config.basin_radius
        if context is None:
            context = self._sample_context()
        v1, v2 = self._principal_directions()
        v1 = v1.to(self.device)
        v2 = v2.to(self.device)
        alphas = torch.linspace(-radius, radius, resolution, device=self.device)
        betas = torch.linspace(-radius, radius, resolution, device=self.device)
        convergence = torch.zeros(resolution, resolution, device=self.device)
        final_energy = torch.zeros_like(convergence)
        base = z_equilibrium[0]
        for i, alpha in enumerate(alphas):
            for j, beta in enumerate(betas):
                start = base + alpha * v1 + beta * v2
                start = start.unsqueeze(0)
                z_final, info = self.model.solver.solve(start, context, self.model.memory_patterns)
                success = 1.0 if info.get("converged", False) else 0.0
                convergence[i, j] = success
                final_energy[i, j] = float(info.get("final_energy", np.nan))
        return {
            "alpha": alphas.cpu().numpy(),
            "beta": betas.cpu().numpy(),
            "convergence": convergence.cpu().numpy(),
            "final_energy": final_energy.cpu().numpy(),
        }

    @torch.no_grad()
    def visualize_memory_organization(self) -> Dict[str, np.ndarray]:
        """Project memory patterns to 2D using PCA and compute similarities."""

        patterns = self.model.memory_patterns.detach().cpu().numpy()
        if patterns.size == 0:
            raise ValueError("No memory patterns available for analysis")
        centered = patterns - patterns.mean(axis=0, keepdims=True)
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        components = vh[:2]
        patterns_2d = centered @ components.T
        similarities = patterns @ patterns.T
        return {
            "patterns_2d": patterns_2d,
            "similarities": similarities,
            "components": components,
        }

    @torch.no_grad()
    def analyze_critical_points(
        self,
        num_samples: int = 16,
    ) -> Iterable[Dict[str, float]]:
        """Identify approximate critical points by converging random initialisations."""

        results = []
        for _ in range(num_samples):
            z0 = torch.randn(1, self._d_model, device=self.device)
            context = self._sample_context()
            z_eq, info = self.model.solver.solve(z0, context, self.model.memory_patterns)
            if not info.get("converged", False):
                continue
            grad = self.model.energy_fn.energy_gradient(z_eq, self.model.memory_patterns)
            grad_norm = float(torch.linalg.vector_norm(grad))
            eigenvalues = self._estimate_hessian_eigs(z_eq, context)
            point_type = self._classify_eigenvalues(eigenvalues)
            results.append(
                {
                    "final_energy": float(info.get("final_energy", np.nan)),
                    "grad_norm": grad_norm,
                    "type": point_type,
                }
            )
        return results

    def _estimate_hessian_eigs(
        self,
        z: Tensor,
        context: Tensor,
        *,
        probes: int = 3,
    ) -> list[float]:
        eigenvalues: list[float] = []
        with torch.enable_grad():
            z_param = z.detach().requires_grad_(True)
            z_next = self.model.dynamics(z_param, context, self.model.memory_patterns)
            energy, _ = self.model.energy_fn(z_param, z_next, self.model.memory_patterns)
            grad = torch.autograd.grad(energy, z_param, create_graph=True)[0]
            for _ in range(probes):
                v = torch.randn_like(z_param)
                v = v / v.norm().clamp_min(1e-8)
                hv = torch.autograd.grad(grad, z_param, v, retain_graph=True)[0]
                eigenvalues.append(float((v * hv).sum()))
        return eigenvalues

    @staticmethod
    def _classify_eigenvalues(eigenvalues: Iterable[float]) -> str:
        values = list(eigenvalues)
        positives = sum(1 for v in values if v > 1e-3)
        negatives = sum(1 for v in values if v < -1e-3)
        total = len(values)
        if total == positives:
            return "minimum"
        if total == negatives:
            return "maximum"
        if positives and negatives:
            return "saddle"
        return "flat"
