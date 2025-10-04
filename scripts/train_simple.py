"""Train the simple language model on synthetic data."""

from __future__ import annotations

import argparse
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from unified_energy.models.simple import SimpleLanguageModel, SimpleLanguageModelConfig
from unified_energy.training.simple_trainer import SimpleLanguageModelTrainer


def _synthetic_dataset(
    *,
    vocab_size: int,
    seq_len: int,
    num_batches: int,
    batch_size: int,
) -> Tuple[DataLoader, DataLoader]:
    total_samples = num_batches * batch_size
    inputs = torch.randint(0, vocab_size, (total_samples, seq_len))
    targets = inputs.clone()
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=batch_size)
    return loader, loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a simple language model")
    parser.add_argument("--vocab-size", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-batches", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SimpleLanguageModelConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )
    model = SimpleLanguageModel.from_config(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    train_loader, val_loader = _synthetic_dataset(
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
    )
    trainer = SimpleLanguageModelTrainer(
        model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
    )
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch()
        val_metrics = trainer.evaluate()
        print(f"Epoch {epoch + 1}: {metrics}")
        if val_metrics:
            print(f"Validation: {val_metrics}")


if __name__ == "__main__":
    main()
