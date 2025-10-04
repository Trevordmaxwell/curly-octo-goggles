import types

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from unified_energy.solvers.hybrid_solver import SolverConfig
from unified_energy.training.objectives import UnifiedTrainingObjective
from unified_energy.training.trainer import CurriculumConfig, UnifiedModelTrainer


def _make_dummy_solver_config() -> SolverConfig:
    return SolverConfig(
        max_iter=4,
        tol_fixedpoint=1e-3,
        tol_energy=1e-3,
        solver_type="alternating",
        anderson_memory=2,
        learning_rate=1e-2,
    )


class DummyModel(nn.Module):
    def __init__(self, vocab_size: int = 8, d_model: int = 6, seq_len: int = 5) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.seq_len = seq_len
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.memory_patterns = nn.Parameter(torch.zeros(7, d_model), requires_grad=False)
        self.solver = types.SimpleNamespace(config=_make_dummy_solver_config())
        self.energy_fn = _DummyEnergy()

    def dynamics(
        self, z: torch.Tensor, context: torch.Tensor, memory_patterns: torch.Tensor
    ) -> torch.Tensor:
        return z + 0.1 * torch.tanh(z)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        update_memory: bool = True,
        return_diagnostics: bool = False,
    ):
        context = self.embedding(input_ids)
        z_equilibrium = context.mean(dim=1)
        logits = self.output(context)
        solver_info = {
            "converged": True,
            "iterations": 2,
            "final_energy": 0.5,
            "energy_components": {"hopfield": 0.2, "consistency": 0.2, "regularization": 0.1},
            "final_fp_residual": 0.01,
            "final_energy_grad": 0.02,
            "energy_history": [0.7, 0.6, 0.5],
        }
        diagnostics = {
            "solver_info": solver_info,
            "z_equilibrium": z_equilibrium,
            "energy_trajectory": solver_info["energy_history"],
            "memory_usage": {},
            "context": context,
        }
        if return_diagnostics:
            return logits, diagnostics
        return logits


class _DummyEnergy:
    def __call__(self, z: torch.Tensor, z_next: torch.Tensor, memory_patterns: torch.Tensor):
        energy = 0.5 * (z**2).mean()
        components = {
            "hopfield": energy,
            "consistency": torch.mean((z - z_next) ** 2),
            "regularization": torch.tensor(0.0, device=z.device, dtype=z.dtype),
            "total": energy,
        }
        return energy, components

    def energy_gradient(
        self,
        z: torch.Tensor,
        memory_patterns: torch.Tensor,
        z_next: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return z


def test_training_objective_returns_all_components() -> None:
    model = DummyModel()
    objective = UnifiedTrainingObjective()
    input_ids = torch.randint(0, model.vocab_size, (2, model.seq_len))
    target_ids = input_ids.clone()
    logits, diagnostics = model(input_ids, return_diagnostics=True)
    diagnostics["logits"] = logits
    loss, components = objective.compute_loss(model, (input_ids, target_ids), diagnostics)
    assert loss.requires_grad
    for key in [
        "total",
        "task",
        "energy",
        "convergence",
        "stability",
        "num_iterations",
        "lipschitz_constant",
    ]:
        assert key in components
    assert "energy_components" in components
    loss.backward()


def test_trainer_step_and_validation_cycle() -> None:
    torch.manual_seed(0)
    model = DummyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train_data = [
        (
            torch.randint(0, model.vocab_size, (2, model.seq_len)),
            torch.randint(0, model.vocab_size, (2, model.seq_len)),
        )
        for _ in range(3)
    ]
    val_data = [
        (
            torch.randint(0, model.vocab_size, (2, model.seq_len)),
            torch.randint(0, model.vocab_size, (2, model.seq_len)),
        )
    ]
    trainer = UnifiedModelTrainer(
        model,
        optimizer=optimizer,
        train_loader=train_data,
        val_loader=val_data,
        curriculum=CurriculumConfig(
            warmup_steps=0,
            max_iter_schedule=(3, 5),
            tolerance_schedule=(1e-2, 5e-3),
            memory_enable_step=0,
        ),
    )
    max_iter, tolerance, memory_enabled = trainer.get_curriculum_params()
    assert max_iter == 3
    assert pytest.approx(tolerance) == 1e-2
    assert memory_enabled
    metrics = trainer.train_step(train_data[0])
    assert "total" in metrics
    val_stats = trainer.validate()
    assert "val/loss" in val_stats
    assert "val/convergence_rate" in val_stats
