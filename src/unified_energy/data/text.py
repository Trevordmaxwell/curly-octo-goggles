"""Byte-level text helpers for simple language modelling."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset


class ByteTokenizer:
    """Map UTF-8 text to integer tokens in ``[0, 255]``."""

    vocab_size: int = 256

    def encode(self, text: str) -> Tensor:
        data = text.encode("utf-8")
        return torch.tensor(list(data), dtype=torch.long)

    def decode(self, tokens: Tensor) -> str:
        flat = tokens.reshape(-1).to(torch.long)
        return bytes(int(t) % 256 for t in flat).decode("utf-8", errors="ignore")


def _build_sequences(tokens: Tensor, seq_len: int) -> Tuple[Tensor, Tensor]:
    if seq_len <= 0:
        raise ValueError("seq_len must be positive")
    if tokens.numel() <= seq_len:
        raise ValueError("token stream is shorter than one sequence")
    usable = (tokens.numel() - 1) // seq_len * seq_len
    if usable == 0:
        raise ValueError("token stream is shorter than one sequence")
    inputs = tokens[:usable].reshape(-1, seq_len)
    targets = tokens[1 : usable + 1].reshape(-1, seq_len)
    return inputs, targets


def create_byte_lm_dataloaders(
    text_path: str | Path,
    *,
    seq_len: int,
    batch_size: int,
    val_fraction: float = 0.1,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 0,
) -> Tuple[DataLoader, Optional[DataLoader], ByteTokenizer]:
    """Return train/validation loaders for byte-level language modelling."""

    path = Path(text_path)
    if not path.exists():
        raise FileNotFoundError(f"Text file not found: {path}")
    text = path.read_text(encoding="utf-8")
    tokenizer = ByteTokenizer()
    tokens = tokenizer.encode(text)
    inputs, targets = _build_sequences(tokens, seq_len=seq_len)
    num_sequences = inputs.size(0)

    val_loader: Optional[DataLoader]
    if val_fraction <= 0 or num_sequences < 2:
        train_inputs = inputs
        train_targets = targets
        val_loader = None
    else:
        val_count = max(1, int(num_sequences * val_fraction))
        val_count = min(val_count, num_sequences - 1)
        train_inputs = inputs[:-val_count]
        train_targets = targets[:-val_count]
        val_inputs = inputs[-val_count:]
        val_targets = targets[-val_count:]
        val_dataset = TensorDataset(val_inputs.contiguous(), val_targets.contiguous())
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=drop_last,
            num_workers=num_workers,
        )

    train_dataset = TensorDataset(train_inputs.contiguous(), train_targets.contiguous())
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    return train_loader, val_loader, tokenizer


__all__ = ["ByteTokenizer", "create_byte_lm_dataloaders"]
