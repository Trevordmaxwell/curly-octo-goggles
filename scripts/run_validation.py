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
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--memory-size", type=int, default=128)
    p.add_argument("--context-length", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    cfg = UnifiedModelConfig(
        vocab_size=256,
        d_model=args.d_model,
        d_state=max(32, args.d_model // 2),
        d_conv=4,
        n_layers=2,
        memory_size=args.memory_size,
        solver_type="alternating",
        max_iterations=15,
        tolerance=1e-3,
    )
    model = UnifiedMambaHopfieldDEQ.from_config(cfg)
    validator = ConvergenceValidator(
        model,
        config=ValidationConfig(
            context_length=args.context_length,
            batch_size=args.batch_size,
            max_perturbation_trials=2,
            stability_radius=0.05,
        ),
        device="cpu",
    )
    results = validator.run_all_tests()
    args.out.write_text(json.dumps(results, indent=2))
    print(f"Wrote results to {args.out.resolve()}")


if __name__ == "__main__":
    main()
