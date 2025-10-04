#!/usr/bin/env python3
"""Scan small model configurations and report validation outcomes."""
from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from unified_energy.experiments.theory import ConvergenceValidator, ValidationConfig
from unified_energy.models.unified import UnifiedMambaHopfieldDEQ, UnifiedModelConfig


@dataclass
class SearchConfig:
    d_model: int
    memory_size: int
    solver_type: str
    max_iter: int
    tolerance: float
    learning_rate: float
    beta: float
    alpha: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=None, help="Optional path to save JSON results")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--runs", type=int, default=3, help="Number of random seeds per configuration")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap on configurations to evaluate")
    return parser.parse_args()


def candidate_configs() -> Iterable[SearchConfig]:
    solver_types = ["alternating", "simultaneous"]
    d_models = [12, 16, 24, 32]
    learning_rates = [0.01, 0.02, 0.05]
    tolerances = [1e-2, 5e-3, 1e-3]
    betas = [1.0, 1.5, 2.0]
    alphas = [0.5, 0.8, 1.2]
    max_iters = [6, 10]
    for solver_type in solver_types:
        for d_model in d_models:
            for lr in learning_rates:
                for tol in tolerances:
                    for beta in betas:
                        for alpha in alphas:
                            for max_iter in max_iters:
                                yield SearchConfig(
                                    d_model=d_model,
                                    memory_size=max(32, d_model * 4),
                                    solver_type=solver_type,
                                    max_iter=max_iter,
                                    tolerance=tol,
                                    learning_rate=lr,
                                    beta=beta,
                                    alpha=alpha,
                                )


def run_single(cfg: SearchConfig, seed: int) -> dict:
    torch.manual_seed(seed)
    model_config = UnifiedModelConfig(
        vocab_size=128,
        d_model=cfg.d_model,
        d_state=max(16, cfg.d_model // 2),
        d_conv=4,
        n_layers=2,
        memory_size=cfg.memory_size,
        beta=cfg.beta,
        alpha=cfg.alpha,
        solver_type=cfg.solver_type,
        max_iterations=cfg.max_iter,
        tolerance=cfg.tolerance,
    )
    model = UnifiedMambaHopfieldDEQ.from_config(model_config)
    model.solver.config.learning_rate = cfg.learning_rate
    validator = ConvergenceValidator(
        model,
        config=ValidationConfig(
            context_length=4,
            batch_size=1,
            max_perturbation_trials=2,
            stability_radius=0.05,
        ),
        device="cpu",
    )
    try:
        return validator.run_all_tests()
    except RuntimeError as exc:
        return {"error": str(exc)}


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


def aggregate_results(cfg: SearchConfig, args: argparse.Namespace) -> dict:
    summaries = []
    for run_idx in range(args.runs):
        seed = args.seed + run_idx
        summaries.append(run_single(cfg, seed))
    scores = [score(s) if "error" not in s else -1 for s in summaries]
    best = max(zip(scores, summaries), key=lambda t: t[0])[1]
    result = {
        "config": cfg.__dict__,
        "scores": scores,
        "mean_score": statistics.fmean(scores),
        "best_summary": best,
    }
    return result


def main() -> None:
    args = parse_args()
    results = []
    for idx, cfg in enumerate(candidate_configs(), start=1):
        if args.limit is not None and idx > args.limit:
            break
        result = aggregate_results(cfg, args)
        results.append(result)
        print(
            f"config={cfg.__dict__} scores={result['scores']} mean_score={result['mean_score']:.2f}"
        )
    results.sort(key=lambda r: (r["mean_score"], score(r["best_summary"])), reverse=True)
    print("\nTop configurations:")
    for idx, result in enumerate(results[:5], start=1):
        cfg = result["config"]
        best = result["best_summary"]
        print(
            f"{idx:02d}: score_mean={result['mean_score']:.2f} config={cfg} best_summary={json.dumps(best)}"
        )
    if args.out:
        args.out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
