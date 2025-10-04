"""Train the simple language model on synthetic or text data."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from typing import Optional, Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset

from unified_energy.data import create_byte_lm_dataloaders
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
    parser.add_argument("--text-path", type=str, default=None, help="UTF-8 text corpus")
    parser.add_argument("--vocab-size", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-batches", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--drop-last", action="store_true")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    train_loader: DataLoader
    val_loader: Optional[DataLoader]
    vocab_size = args.vocab_size

    if args.text_path:
        train_loader, val_loader, tokenizer = create_byte_lm_dataloaders(
            args.text_path,
            seq_len=args.seq_len,
            batch_size=args.batch_size,
            val_fraction=args.val_fraction,
            drop_last=args.drop_last,
            num_workers=args.num_workers,
        )
        vocab_size = tokenizer.vocab_size
        if val_loader is None:
            print("Warning: validation split skipped because the corpus is too small.")
    else:
        train_loader, synthetic_val = _synthetic_dataset(
            vocab_size=vocab_size,
            seq_len=args.seq_len,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
        )
        val_loader = synthetic_val

    config = SimpleLanguageModelConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
    )
    model = SimpleLanguageModel.from_config(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    trainer = SimpleLanguageModelTrainer(
        model,
        optimizer=optimizer,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
    )
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset) if val_loader is not None else 0
    print(
        f"Training with vocab={vocab_size}, train_sequences={train_size}, val_sequences={val_size}"
    )
    for epoch in range(args.epochs):
        metrics = trainer.train_epoch()
        val_metrics = trainer.evaluate()
        print(f"Epoch {epoch + 1}: {metrics}")
        if val_metrics:
            print(f"Validation: {val_metrics}")

    if args.save_path:
        checkpoint = {
            "config": asdict(config),
            "state_dict": model.state_dict(),
            "tokenizer": "byte" if args.text_path else "synthetic",
            "metadata": {
                "train_sequences": train_size,
                "val_sequences": val_size,
                "seed": args.seed,
            },
        }
        torch.save(checkpoint, args.save_path)
        print(f"Saved checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
