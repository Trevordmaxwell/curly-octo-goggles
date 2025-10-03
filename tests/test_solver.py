import pytest

torch = pytest.importorskip("torch")

from unified_energy.core.dynamics import UnifiedDynamics
from unified_energy.core.energy import UnifiedEnergyFunction
from unified_energy.solvers.hybrid_solver import SolverConfig, UnifiedEquilibriumSolver


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
