"""Associative recall dataset for sequence-to-sequence evaluation."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch import Tensor


@dataclass(slots=True)
class AssociativeRecallConfig:
    """Configuration for synthetic associative recall sequences."""

    vocab_size: int = 32
    key_length: int = 3
    value_length: int = 1
    sequence_length: int = 7
    batch_size: int = 32
    pad_token: int = 0
    query_token: int = 1
    answer_token: int = 2

    def __post_init__(self) -> None:
        if self.vocab_size <= 3:
            raise ValueError("vocab_size must exceed reserved tokens")
        if self.key_length <= 0 or self.value_length <= 0:
            raise ValueError("key/value lengths must be positive")
        if self.sequence_length <= self.key_length + self.value_length:
            raise ValueError("sequence_length must accommodate key/value pairs")


class AssociativeRecallDataset:
    """Generate key-value sequences and query recall targets."""

    def __init__(self, config: AssociativeRecallConfig, *, device: torch.device | str = "cpu") -> None:
        self.config = config
        self.device = torch.device(device)

    def sample_batch(self) -> Tuple[Tensor, Tensor]:
        inputs: List[List[int]] = []
        targets: List[List[int]] = []
        cfg = self.config
        for _ in range(cfg.batch_size):
            pairs = self._generate_pairs()
            keys = [token for pair in pairs for token in pair[0]]
            values = [token for pair in pairs for token in pair[1]]
            context = keys + values
            random.shuffle(context)
            query_pair = random.choice(pairs)
            query = [cfg.query_token] + query_pair[0]
            answer = query_pair[1]
            seq = context + query
            seq = seq[: cfg.sequence_length]
            seq += [cfg.pad_token] * (cfg.sequence_length - len(seq))
            target_seq = [cfg.pad_token] * cfg.sequence_length
            target_seq[-cfg.value_length :] = answer
            inputs.append(seq)
            targets.append(target_seq)
        return torch.tensor(inputs, dtype=torch.long, device=self.device), torch.tensor(
            targets, dtype=torch.long, device=self.device
        )

    def _generate_pairs(self) -> List[Tuple[List[int], List[int]]]:
        cfg = self.config
        alphabet = list(range(3, cfg.vocab_size))
        random.shuffle(alphabet)
        pairs: List[Tuple[List[int], List[int]]] = []
        num_pairs = max(1, (cfg.sequence_length - cfg.key_length - cfg.value_length) // cfg.key_length)
        for idx in range(num_pairs):
            key = alphabet[idx * cfg.key_length : (idx + 1) * cfg.key_length]
            value = alphabet[(idx + num_pairs) * cfg.value_length : (idx + num_pairs + 1) * cfg.value_length]
            if len(key) < cfg.key_length:
                key = key + [alphabet[-1]] * (cfg.key_length - len(key))
            if len(value) < cfg.value_length:
                value = value + [alphabet[-1]] * (cfg.value_length - len(value))
            pairs.append((key, value))
        return pairs


__all__ = ["AssociativeRecallDataset", "AssociativeRecallConfig"]
