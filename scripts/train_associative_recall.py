"""Train the unified model on associative recall sequences."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
try:
    from torch.utils.tensorboard import SummaryWriter
except Exception:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore

from unified_energy.models.unified import UnifiedMambaHopfieldDEQ, UnifiedModelConfig
from unified_energy.tasks import AssociativeRecallConfig, AssociativeRecallDataset
from unified_energy.training.trainer import CurriculumConfig, UnifiedModelTrainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Associative recall training")
    parser.add_argument("--steps", type=int, default=500, help="Training steps per run")
    parser.add_argument("--logdir", type=Path, default=Path("runs/associative"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    args.logdir.mkdir(parents=True, exist_ok=True)
    dataset = AssociativeRecallDataset(
        AssociativeRecallConfig(vocab_size=64, key_length=3, value_length=2, sequence_length=12)
    )
    config = UnifiedModelConfig(
        vocab_size=dataset.config.vocab_size,
        d_model=64,
        d_state=32,
        n_layers=2,
        memory_size=128,
        beta=1.5,
        alpha=1.0,
        solver_type="alternating",
        max_iterations=15,
        tolerance=1e-3,
    )
    model = UnifiedMambaHopfieldDEQ.from_config(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_batches = [dataset.sample_batch() for _ in range(args.steps)]
    trainer = UnifiedModelTrainer(
        model,
        optimizer=optimizer,
        train_loader=train_batches,
        curriculum=CurriculumConfig(
            warmup_steps=100,
            max_iter_schedule=(10, 15, 20),
            tolerance_schedule=(5e-3, 2e-3, 1e-3),
            memory_enable_step=50,
        ),
        device=args.device,
    )
    writer = SummaryWriter(log_dir=args.logdir) if SummaryWriter is not None else None
    for step, batch in enumerate(train_batches, start=1):
        metrics = trainer.train_step(batch)
        if writer is not None:
            for key, value in metrics.items():
                writer.add_scalar(f"train/{key}", value, step)
        if step % 100 == 0:
            print(f"Step {step}: {metrics}")
    if writer is not None:
        writer.close()
    torch.save(model.state_dict(), args.logdir / "model.pt")


if __name__ == "__main__":
    main()
