"""Curriculum-based training loop for the unified model."""
from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

wandb_spec = importlib.util.find_spec("wandb")
wandb = importlib.import_module("wandb") if wandb_spec is not None else None


class UnifiedModelTrainer:
    """Training loop with curriculum learning for stable DEQ training."""

    def __init__(
        self,
        model: Any,
        optimizer: torch.optim.Optimizer,
        train_loader: DataLoader,
        val_loader: DataLoader,
        objective: Any,
        device: str = "cuda",
        log_wandb: bool = True,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.objective = objective
        self.device = device
        self.log_wandb = log_wandb and wandb is not None

        self.step = 0
        self.epoch = 0

        self.curriculum = {
            "warmup_steps": 1000,
            "max_iter_schedule": [5, 10, 20, 30],
            "tolerance_schedule": [1e-2, 5e-3, 1e-3, 5e-4],
            "memory_enable_step": 2000,
        }

    def get_curriculum_params(self) -> Tuple[int, float, bool]:
        """Get current curriculum parameters based on training step."""

        if self.step < self.curriculum["warmup_steps"]:
            max_iter = self.curriculum["max_iter_schedule"][0]
            tolerance = self.curriculum["tolerance_schedule"][0]
        else:
            stage = min(
                len(self.curriculum["max_iter_schedule"]) - 1,
                (self.step - self.curriculum["warmup_steps"]) // 2000,
            )
            max_iter = self.curriculum["max_iter_schedule"][stage]
            tolerance = self.curriculum["tolerance_schedule"][stage]

        memory_enabled = self.step >= self.curriculum["memory_enable_step"]

        return max_iter, tolerance, memory_enabled

    def train_step(self, batch: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        """Single training step with curriculum adjustments."""

        self.model.train()

        max_iter, tolerance, memory_enabled = self.get_curriculum_params()

        self.model.solver.max_iter = max_iter
        self.model.solver.tol_fp = tolerance
        self.model.solver.tol_energy = tolerance

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

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

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
        """Validation loop with detailed diagnostics."""

        self.model.eval()

        total_loss = 0.0
        total_converged = 0
        total_iterations = []

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

                loss, loss_components = self.objective.compute_loss(
                    self.model,
                    (input_ids, target_ids),
                    diagnostics,
                )

                total_loss += float(loss.detach().item())
                solver_info = diagnostics.get("solver_info", {})
                total_converged += int(solver_info.get("converged", False))
                total_iterations.append(solver_info.get("iterations", 0))

        num_batches = max(1, len(self.val_loader))
        val_stats = {
            "val/loss": total_loss / num_batches,
            "val/convergence_rate": total_converged / num_batches,
            "val/avg_iterations": float(np.mean(total_iterations)) if total_iterations else 0.0,
            "val/median_iterations": float(np.median(total_iterations)) if total_iterations else 0.0,
            "val/max_iterations": float(np.max(total_iterations)) if total_iterations else 0.0,
        }

        if self.log_wandb:
            wandb.log(val_stats)

        return val_stats

    def train(self, num_epochs: int) -> None:
        """Full training loop with validation and checkpointing."""

        for epoch in range(num_epochs):
            self.epoch = epoch
            print(f"\nEpoch {epoch + 1}/{num_epochs}")

            for batch_idx, batch in enumerate(tqdm(self.train_loader)):
                loss_components = self.train_step(batch)

                if self.step % 500 == 0:
                    val_stats = self.validate()
                    print(
                        f"Step {self.step}: Val Loss = {val_stats['val/loss']:.4f}, "
                        f"Convergence = {val_stats['val/convergence_rate']:.2%}"
                    )

            val_stats = self.validate()

            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")

        print("Training complete!")

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint with training state."""

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "step": self.step,
                "epoch": self.epoch,
                "memory_patterns": getattr(self.model, "memory_patterns", None),
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""

        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.step = checkpoint.get("step", 0)
        self.epoch = checkpoint.get("epoch", 0)
        if "memory_patterns" in checkpoint:
            self.model.memory_patterns = checkpoint["memory_patterns"]
