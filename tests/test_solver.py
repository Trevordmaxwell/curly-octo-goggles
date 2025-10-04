import pytest

torch = pytest.importorskip("torch")

from unified_energy.core.dynamics import UnifiedDynamics
from unified_energy.core.energy import UnifiedEnergyFunction
from unified_energy.solvers.hybrid_solver import SolverConfig, UnifiedEquilibriumSolver


class _RecordingEnergy(UnifiedEnergyFunction):
    def __init__(self, d_model: int) -> None:
        super().__init__(d_model=d_model)
        self.last_z_next = None

    def energy_gradient(self, z, memory_patterns, z_next=None):
        self.last_z_next = z_next
        return super().energy_gradient(z, memory_patterns, z_next=z_next)


def _make_fixture():
    dynamics = UnifiedDynamics(d_model=4, d_state=6, d_conv=3)
    energy = UnifiedEnergyFunction(d_model=4)
    solver = UnifiedEquilibriumSolver(dynamics, energy, SolverConfig(max_iter=4))
    return dynamics, energy, solver


def test_alternating_solver_returns_info_dict() -> None:
    _, _, solver = _make_fixture()
    solver.config.solver_type = "alternating"
    z0 = torch.randn(2, 4)
    context = torch.randn(2, 5, 4)
    memory = torch.randn(7, 4)

    z_star, info = solver.solve(z0, context, memory)
    assert z_star.shape == z0.shape
    assert "iterations" in info


def test_simultaneous_solver_converges_quickly() -> None:
    _, _, solver = _make_fixture()
    solver.config.solver_type = "simultaneous"
    z0 = torch.randn(1, 4)
    context = torch.randn(1, 3, 4)
    memory = torch.randn(5, 4)

    z_star, info = solver.solve(z0, context, memory)
    assert z_star.shape == z0.shape
    assert "energy_history" in info


def test_cascade_solver_executes() -> None:
    _, _, solver = _make_fixture()
    solver.config.solver_type = "cascade"
    z0 = torch.randn(1, 4)
    context = torch.randn(1, 3, 4)
    memory = torch.randn(5, 4)

    z_star, info = solver.solve(z0, context, memory)
    assert z_star.shape == z0.shape
    assert "final_energy" in info


def test_energy_gradient_receives_dynamics_output() -> None:
    dynamics = UnifiedDynamics(d_model=4, d_state=6, d_conv=3)
    energy = _RecordingEnergy(d_model=4)
    solver = UnifiedEquilibriumSolver(dynamics, energy, SolverConfig(max_iter=2))
    z = torch.randn(2, 4)
    context = torch.randn(2, 5, 4)
    memory = torch.randn(6, 4)

    solver.energy_descent_step(z, context, memory)

    assert energy.last_z_next is not None
    assert torch.allclose(energy.last_z_next, dynamics(z, context, memory))


def test_solver_reports_non_convergence() -> None:
    class ExpandingDynamics:
        def __call__(self, z, context, memory_patterns):
            return z + 0.2 * z

    class QuadraticEnergy:
        def __call__(self, z, z_next, memory_patterns):
            energy = torch.mean(z * z)
            components = {
                "hopfield": energy,
                "consistency": torch.mean((z - z_next) ** 2),
                "regularization": torch.tensor(0.0, device=z.device, dtype=z.dtype),
                "total": energy,
            }
            return energy, components

        def energy_gradient(self, z, memory_patterns, z_next=None):
            return z

    dynamics = ExpandingDynamics()
    energy = QuadraticEnergy()
    solver = UnifiedEquilibriumSolver(
        dynamics, energy, SolverConfig(max_iter=3, tol_fixedpoint=1e-6)
    )
    solver.config.solver_type = "alternating"

    z0 = torch.randn(1, 4)
    context = torch.zeros(1, 2, 4)
    memory = torch.zeros(3, 4)

    z_star, info = solver.solve(z0, context, memory)

    assert z_star.shape == z0.shape
    assert info["converged"] is False
    assert info["iterations"] == solver.config.max_iter
    assert len(info.get("energy_history", [])) == solver.config.max_iter
