"""Curriculum-aware training loop for the unified model."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.nn.utils import clip_grad_norm_

from .objectives import UnifiedTrainingObjective

try:  # pragma: no cover - optional dependency
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - fallback when tqdm is unavailable
    tqdm = None

try:  # pragma: no cover - optional dependency
    import wandb
except Exception:  # pragma: no cover - fallback when wandb is unavailable
    wandb = None  # type: ignore


@dataclass(slots=True)
class CurriculumConfig:
    """Schedules controlling solver depth and memory usage during training."""

    warmup_steps: int = 500
    max_iter_schedule: Sequence[int] = (10, 20, 30)
    tolerance_schedule: Sequence[float] = (1e-2, 5e-3, 1e-3)
    memory_enable_step: int = 1000

    def __post_init__(self) -> None:
        if len(self.max_iter_schedule) != len(self.tolerance_schedule):
            msg = "max_iter_schedule and tolerance_schedule must have equal length"
            raise ValueError(msg)
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be non-negative")
        if any(value <= 0 for value in self.max_iter_schedule):
            raise ValueError("max_iter_schedule entries must be positive")
        if any(value <= 0 for value in self.tolerance_schedule):
            raise ValueError("tolerance_schedule entries must be positive")
        if self.memory_enable_step < 0:
            raise ValueError("memory_enable_step must be non-negative")


class UnifiedModelTrainer:
    """High-level trainer orchestrating optimisation and curriculum scheduling."""

    def __init__(
        self,
        model,
        *,
        optimizer: torch.optim.Optimizer,
        train_loader: Iterable[Tuple[Tensor, Tensor]],
        val_loader: Optional[Iterable[Tuple[Tensor, Tensor]]] = None,
        objective: Optional[UnifiedTrainingObjective] = None,
        curriculum: Optional[CurriculumConfig] = None,
        device: Optional[torch.device | str] = None,
        log_wandb: bool = False,
    ) -> None:
        self.model = model
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.objective = objective or UnifiedTrainingObjective()
        self.curriculum = curriculum or CurriculumConfig()
        self.log_wandb = log_wandb and wandb is not None
        self.step = 0
        self.epoch = 0

    def get_curriculum_params(self) -> Tuple[int, float, bool]:
        """Return ``(max_iter, tolerance, memory_enabled)`` for the current step."""

        if self.step < self.curriculum.warmup_steps:
            return (
                self.curriculum.max_iter_schedule[0],
                self.curriculum.tolerance_schedule[0],
                False,
            )
        stage = min(
            len(self.curriculum.max_iter_schedule) - 1,
            (self.step - self.curriculum.warmup_steps) // 2000,
        )
        max_iter = self.curriculum.max_iter_schedule[stage]
        tolerance = self.curriculum.tolerance_schedule[stage]
        memory_enabled = self.step >= self.curriculum.memory_enable_step
        return max_iter, tolerance, memory_enabled

    def train_step(self, batch: Tuple[Tensor, Tensor]) -> Dict[str, float]:
        """Run a single optimisation step and return scalar metrics."""

        self.model.train()
        max_iter, tolerance, memory_enabled = self.get_curriculum_params()
        self.model.solver.config.max_iter = max_iter
        self.model.solver.config.tol_fixedpoint = tolerance
        self.model.solver.config.tol_energy = tolerance

        input_ids, target_ids = batch
        input_ids = input_ids.to(self.device)
        target_ids = target_ids.to(self.device)

        logits, diagnostics = self.model(
            input_ids,
            update_memory=memory_enabled,
            return_diagnostics=True,
        )
        diagnostics["logits"] = logits
        loss, loss_components = self.objective.compute_loss(
            self.model,
            (input_ids, target_ids),
            diagnostics,
        )

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()

        if self.log_wandb and self.step % 10 == 0:
            wandb.log(
                {
                    **loss_components,
                    "curriculum/max_iter": max_iter,
                    "curriculum/tolerance": tolerance,
                    "curriculum/memory_enabled": int(memory_enabled),
                    "step": self.step,
                }
            )
        self.step += 1
        return loss_components

    def validate(self) -> Dict[str, float]:
        """Evaluate the model on the validation loader."""

        if self.val_loader is None:
            return {}
        self.model.eval()
        total_loss = 0.0
        total_converged = 0
        iterations: List[float] = []
        energy_histories: List[Sequence[float]] = []
        num_batches = 0

        with torch.no_grad():
            for batch in self.val_loader:
                input_ids, target_ids = batch
                input_ids = input_ids.to(self.device)
                target_ids = target_ids.to(self.device)
                logits, diagnostics = self.model(
                    input_ids,
                    update_memory=False,
                    return_diagnostics=True,
                )
                diagnostics["logits"] = logits
                loss, _ = self.objective.compute_loss(
                    self.model,
                    (input_ids, target_ids),
                    diagnostics,
                )
                total_loss += float(loss.detach())
                solver_info = diagnostics.get("solver_info", {})
                total_converged += int(bool(solver_info.get("converged", False)))
                iterations.append(float(solver_info.get("iterations", 0) or 0))
                history = solver_info.get("energy_history", [])
                if isinstance(history, Sequence):
                    energy_histories.append(history)
                num_batches += 1

        if num_batches == 0:
            return {}

        avg_iterations = sum(iterations) / max(len(iterations), 1)
        stats = {
            "val/loss": total_loss / num_batches,
            "val/convergence_rate": total_converged / num_batches,
            "val/avg_iterations": avg_iterations,
        }
        if iterations:
            stats["val/median_iterations"] = float(torch.median(torch.tensor(iterations)).item())
            stats["val/max_iterations"] = float(max(iterations))
        if self.log_wandb:
            wandb.log(stats)
        return stats

    def train(self, num_epochs: int) -> None:
        """Full training loop with optional validation and checkpointing."""

        loader = self.train_loader
        for epoch in range(num_epochs):
            self.epoch = epoch
            iterator = loader
            if tqdm is not None:
                iterator = tqdm(loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            for batch in iterator:
                self.train_step(batch)
            if self.val_loader is not None:
                self.validate()

    def save_checkpoint(self, path: str) -> None:
        """Persist model, optimiser, and curriculum state."""

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
                "memory_patterns": self.model.memory_patterns,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Restore training state from a checkpoint."""

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = int(checkpoint.get("step", 0))
        self.epoch = int(checkpoint.get("epoch", 0))
        memory_patterns = checkpoint.get("memory_patterns")
        if isinstance(memory_patterns, Tensor):
            self.model.memory_patterns = memory_patterns.to(self.device)


def train_epoch(
    trainer: UnifiedModelTrainer,
    *,
    max_steps: Optional[int] = None,
) -> None:
    """Convenience wrapper to run a limited number of training steps."""

    for step, batch in enumerate(trainer.train_loader):
        trainer.train_step(batch)
        if max_steps is not None and step + 1 >= max_steps:
            break
