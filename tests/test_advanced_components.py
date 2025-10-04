import pytest

torch = pytest.importorskip("torch")

from unified_energy.backbones import HybridBackbone, HybridBackboneConfig
from unified_energy.memory import HierarchicalMemory
from unified_energy.solvers.lbfgs import LBFGSConfig, LBFGSSolver


def test_hybrid_backbone_returns_sequence_and_state() -> None:
    config = HybridBackboneConfig(
        d_model=12,
        num_layers=4,
        attention_every=2,
        attention_type="local",
        window_size=2,
        num_heads=2,
    )
    backbone = HybridBackbone(config)
    context = torch.randn(2, 6, 12)
    processed, state = backbone(context, return_state=True)
    assert processed.shape == context.shape
    assert state.shape == (2, 12)
    assert torch.isfinite(processed).all()


def test_hierarchical_memory_store_and_retrieve() -> None:
    memory = HierarchicalMemory(d_model=8, l1_size=4, l2_size=4, l3_size=2, consolidation_freq=2)
    patterns = torch.randn(5, 8)
    memory.store(patterns)
    query = torch.randn(3, 8)
    retrieved, details = memory.retrieve(query, return_details=True)
    assert retrieved.shape == (3, 8)
    weights = details["level_weights"]
    assert weights.shape == (3, 3)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3), atol=1e-5)
    memory.consolidate()
    memory.consolidate()
    assert int(memory.level_counts[0].item()) > 0


def test_lbfgs_solver_converges_on_linear_dynamics() -> None:
    config = LBFGSConfig(max_iter=15, min_iter=2, tol=1e-4, confidence_threshold=0.4)
    solver = LBFGSSolver(config)

    def dynamics_fn(z: torch.Tensor, _context, _memory) -> torch.Tensor:
        return 0.5 * z

    class QuadraticEnergy:
        def __call__(self, z: torch.Tensor, z_next: torch.Tensor, _memory) -> tuple[torch.Tensor, dict[str, float]]:
            energy = 0.5 * torch.sum(z_next * z_next, dim=-1).mean()
            residual = torch.sum((z - z_next) ** 2, dim=-1).mean()
            components = {
                "hopfield": float(energy.detach()),
                "consistency": float(residual.detach()),
                "regularization": 0.0,
                "total": float((energy + residual).detach()),
            }
            return energy + residual, components

        def energy_gradient(self, z: torch.Tensor, _memory) -> torch.Tensor:
            return z

    energy_fn = QuadraticEnergy()
    z_init = torch.randn(2, 4)
    context = torch.zeros(2, 3, 4)
    memory_patterns = torch.zeros(6, 4)
    z_star, info = solver.solve(z_init, dynamics_fn, energy_fn, context=context, memory_patterns=memory_patterns)
    assert z_star.shape == z_init.shape
    assert info["iterations"] <= config.max_iter
    assert info["confidence"] >= 0.0
    assert len(info["energy_history"]) >= 1
    assert info["final_residual"] < 1.0

