import types

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from unified_energy.experiments.theory import (
    ConvergenceValidator,
    EnergyLandscapeAnalyzer,
    LandscapeConfig,
    ValidationConfig,
)


class QuadraticEnergy:
    def __call__(self, z: torch.Tensor, z_next: torch.Tensor, memory_patterns: torch.Tensor):
        energy = 0.5 * torch.sum(z_next * z_next, dim=-1).mean()
        components = {
            "hopfield": float(energy),
            "consistency": float(torch.sum((z - z_next) ** 2, dim=-1).mean()),
            "regularization": 0.0,
            "total": float(energy),
        }
        return energy, components

    def energy_gradient(self, z: torch.Tensor, memory_patterns: torch.Tensor) -> torch.Tensor:
        return z


class LinearSolver:
    def __init__(self, dynamics, energy_fn) -> None:
        self.dynamics = dynamics
        self.energy_fn = energy_fn
        self.config = types.SimpleNamespace(max_iter=6, tol_fixedpoint=1e-4, tol_energy=1e-4)

    def solve(self, z_init, context, memory_patterns):
        z = z_init.clone()
        info = {
            "iterations": 0,
            "energy_history": [],
            "energy_components": {},
        }
        for iteration in range(self.config.max_iter):
            z_next = self.dynamics(z, context, memory_patterns)
            energy, components = self.energy_fn(z, z_next, memory_patterns)
            grad_norm = torch.linalg.vector_norm(
                self.energy_fn.energy_gradient(z_next, memory_patterns)
            ).item()
            fp_residual = torch.linalg.vector_norm(z_next - z).item()
            info.update(
                {
                    "iterations": iteration + 1,
                    "final_energy": float(energy.detach()),
                    "final_energy_grad": grad_norm,
                    "final_fp_residual": fp_residual,
                    "energy_components": components,
                }
            )
            info["energy_history"].append(float(energy.detach()))
            if fp_residual < self.config.tol_fixedpoint:
                info["converged"] = True
                return z_next.detach(), info
            z = z_next
        info.setdefault("converged", False)
        return z.detach(), info


class DummyUnifiedModel(nn.Module):
    def __init__(self, d_model: int = 4, memory_size: int = 6) -> None:
        super().__init__()
        self.d_model = d_model
        self.register_buffer("memory_patterns", torch.zeros(memory_size, d_model))
        self.energy_fn = QuadraticEnergy()
        self.solver = LinearSolver(self.dynamics, self.energy_fn)

    def dynamics(self, z: torch.Tensor, context: torch.Tensor, memory_patterns: torch.Tensor) -> torch.Tensor:
        return 0.5 * z


@pytest.fixture()
def dummy_model() -> DummyUnifiedModel:
    model = DummyUnifiedModel()
    return model


def test_convergence_validator_runs(dummy_model: DummyUnifiedModel) -> None:
    validator = ConvergenceValidator(
        dummy_model,
        config=ValidationConfig(context_length=3, batch_size=1, max_perturbation_trials=2, stability_radius=0.05),
        device="cpu",
    )
    contraction = validator.test_contraction_property(num_samples=4)
    assert set(contraction) >= {"mean_lipschitz", "max_lipschitz", "is_contractive"}

    energy = validator.test_energy_descent(num_trajectories=3, num_steps=5)
    assert "monotonic_descent" in energy

    stability = validator.test_fixed_point_stability(num_fixed_points=3, num_perturbations=2)
    assert "stability_rate" in stability

    lyapunov = validator.test_lyapunov_function(num_samples=3, horizon=4)
    assert "lyapunov_rate" in lyapunov

    summary = validator.run_all_tests()
    assert "summary" in summary


def test_energy_landscape_analyzer_shapes(dummy_model: DummyUnifiedModel) -> None:
    analyzer = EnergyLandscapeAnalyzer(
        dummy_model,
        config=LandscapeConfig(context_length=3, grid_resolution=5, exploration_radius=0.2, basin_resolution=5, basin_radius=0.3),
        device="cpu",
    )
    z_eq = torch.zeros(1, dummy_model.d_model)
    context = torch.zeros(1, analyzer.config.context_length, dummy_model.d_model)
    slice_data = analyzer.visualize_2d_slice(z_eq, context=context, resolution=5, radius=0.2)
    assert slice_data["energy"].shape == (5, 5)

    trajectories = analyzer.visualize_convergence_trajectories(num_trajectories=3, steps=4)
    assert "trajectories_2d" in trajectories
    assert len(trajectories["trajectories_2d"]) == 3

    basin = analyzer.visualize_basin_of_attraction(z_eq, context=context, resolution=5, radius=0.2)
    assert basin["convergence"].shape == (5, 5)

    patterns = analyzer.visualize_memory_organization()
    assert patterns["patterns_2d"].shape[1] == 2

    critical_points = list(analyzer.analyze_critical_points(num_samples=2))
    assert all("grad_norm" in point for point in critical_points)
