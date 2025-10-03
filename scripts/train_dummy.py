#!/usr/bin/env python3
"""Run a tiny CPU training smoke test using the curriculum trainer."""
from __future__ import annotations

import argparse
from typing import List, Tuple

import torch

from unified_energy.models.unified import UnifiedMambaHopfieldDEQ, UnifiedModelConfig
from unified_energy.training.objectives import UnifiedTrainingObjective
from unified_energy.training.trainer import CurriculumConfig, UnifiedModelTrainer


def _make_data(vocab_size: int, seq_len: int, batches: int, batch_size: int) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    return [
        (
            torch.randint(0, vocab_size, (batch_size, seq_len)),
            torch.randint(0, vocab_size, (batch_size, seq_len)),
        )
        for _ in range(batches)
    ]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=16)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--max-iter", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = UnifiedModelConfig(
        vocab_size=128,
        d_model=args.d_model,
        d_state=max(32, args.d_model // 2),
        d_conv=4,
        n_layers=2,
        memory_size=128,
        solver_type="alternating",
        max_iterations=args.max_iter,
        tolerance=1e-3,
    )
    model = UnifiedMambaHopfieldDEQ.from_config(cfg)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_data = _make_data(cfg.vocab_size, args.seq_len, args.batches, args.batch_size)
    val_data = _make_data(cfg.vocab_size, args.seq_len, 1, args.batch_size)
    trainer = UnifiedModelTrainer(
        model,
        optimizer=opt,
        train_loader=train_data,
        val_loader=val_data,
        objective=UnifiedTrainingObjective(),
        curriculum=CurriculumConfig(warmup_steps=0, max_iter_schedule=(args.max_iter,), tolerance_schedule=(1e-3,), memory_enable_step=0),
        device="cpu",
        log_wandb=False,
    )
    for epoch in range(args.epochs):
        for batch in train_data:
            metrics = trainer.train_step(batch)
        stats = trainer.validate()
        print({"epoch": epoch + 1, **stats})


if __name__ == "__main__":
    main()

