#!/usr/bin/env python3
"""Randomly sample hyperparameters and search for positive validation scores."""
from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from unified_energy.experiments.theory import ConvergenceValidator, ValidationConfig
from unified_energy.models.unified import UnifiedMambaHopfieldDEQ, UnifiedModelConfig


@dataclass
class Result:
    config: dict
    score: int
    summary: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=None)
    return parser.parse_args()


def build_model(config: dict) -> UnifiedMambaHopfieldDEQ:
    unified_cfg = UnifiedModelConfig(
        vocab_size=config["vocab_size"],
        d_model=config["d_model"],
        d_state=config["d_state"],
        d_conv=3,
        n_layers=config["n_layers"],
        memory_size=config["memory_size"],
        beta=config["beta"],
        alpha=config["alpha"],
        solver_type=config["solver_type"],
        max_iterations=config["max_iterations"],
        tolerance=config["tolerance"],
    )
    model = UnifiedMambaHopfieldDEQ.from_config(unified_cfg)
    model.solver.config.learning_rate = config["learning_rate"]
    return model


def score(summary: dict) -> int:
    if "error" in summary:
        return -1
    metrics = [
        summary["contraction"].get("is_contractive", False),
        summary["energy_descent"].get("monotonic_descent", False),
        summary["stability"].get("is_stable", False),
        summary["lyapunov"].get("is_lyapunov", False),
    ]
    return sum(bool(m) for m in metrics)


def sample_config(rng: random.Random) -> dict:
    d_model = rng.choice([6, 8, 10, 12])
    return {
        "vocab_size": rng.choice([16, 32, 64]),
        "d_model": d_model,
        "d_state": max(6, d_model // 2),
        "n_layers": rng.choice([1, 2]),
        "memory_size": rng.choice([d_model * 2, d_model * 4, 64]),
        "beta": rng.uniform(0.3, 1.2),
        "alpha": rng.uniform(0.05, 0.4),
        "solver_type": "alternating",
        "max_iterations": rng.choice([4, 6, 8]),
        "tolerance": rng.choice([1e-2, 5e-3, 1e-3]),
        "learning_rate": rng.uniform(0.005, 0.05),
    }


def run_validation(config: dict) -> dict:
    torch.manual_seed(0)
    model = build_model(config)
    validator = ConvergenceValidator(
        model,
        config=ValidationConfig(
            context_length=3, batch_size=1, max_perturbation_trials=1, stability_radius=0.02
        ),
        device="cpu",
    )
    try:
        return validator.run_all_tests()
    except RuntimeError as exc:
        return {"error": str(exc)}


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)
    best: Result | None = None
    history: list[Result] = []
    for idx in range(1, args.samples + 1):
        cfg = sample_config(rng)
        summary = run_validation(cfg)
        cfg_score = score(summary)
        history.append(Result(config=cfg, score=cfg_score, summary=summary))
        if best is None or cfg_score > best.score:
            best = history[-1]
        print(f"sample={idx:04d} score={cfg_score} cfg={cfg}")
        if cfg_score >= 1:
            break
    if best is None:
        print("No configurations evaluated.")
        return
    print("\nBest configuration:")
    print(
        json.dumps({"config": best.config, "score": best.score, "summary": best.summary}, indent=2)
    )
    if args.out:
        args.out.write_text(
            json.dumps(
                {
                    "best": {"config": best.config, "score": best.score, "summary": best.summary},
                    "history": [
                        {"config": h.config, "score": h.score, "summary": h.summary}
                        for h in history
                    ],
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
