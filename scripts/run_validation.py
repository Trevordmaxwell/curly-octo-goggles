#!/usr/bin/env python3
"""Run the convergence validation battery on CPU and dump JSON results."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from unified_energy.experiments.theory import ConvergenceValidator, ValidationConfig
from unified_energy.models.unified import UnifiedMambaHopfieldDEQ, UnifiedModelConfig


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--out", type=Path, default=Path("validation_results.json"), help="Output JSON path"
    )
    p.add_argument("--vocab-size", type=int, default=256)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--d-state", type=int, default=None)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--memory-size", type=int, default=128)
    p.add_argument("--beta", type=float, default=1.5)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument(
        "--solver-type", choices=["alternating", "simultaneous", "cascade"], default="alternating"
    )
    p.add_argument("--max-iter", type=int, default=15)
    p.add_argument("--tolerance", type=float, default=1e-3)
    p.add_argument("--learning-rate", type=float, default=0.02)
    p.add_argument("--context-length", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--max-perturbations", type=int, default=2)
    p.add_argument("--stability-radius", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = UnifiedModelConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        d_state=args.d_state or max(16, args.d_model // 2),
        d_conv=4,
        n_layers=args.n_layers,
        memory_size=args.memory_size,
        beta=args.beta,
        alpha=args.alpha,
        solver_type=args.solver_type,
        max_iterations=args.max_iter,
        tolerance=args.tolerance,
    )
    model = UnifiedMambaHopfieldDEQ.from_config(cfg)
    model.solver.config.learning_rate = args.learning_rate
    validator = ConvergenceValidator(
        model,
        config=ValidationConfig(
            context_length=args.context_length,
            batch_size=args.batch_size,
            max_perturbation_trials=args.max_perturbations,
            stability_radius=args.stability_radius,
        ),
        device="cpu",
    )
    results = validator.run_all_tests()
    args.out.write_text(json.dumps(results, indent=2))
    print(f"Wrote results to {args.out.resolve()}")


if __name__ == "__main__":
    main()
