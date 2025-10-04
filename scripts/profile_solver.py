#!/usr/bin/env python3
"""Profile the unified equilibrium solver on CPU."""
from __future__ import annotations

import argparse
import statistics
import time

import torch

from unified_energy.models.unified import UnifiedMambaHopfieldDEQ, UnifiedModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5, help="Number of forward passes to profile")
    parser.add_argument(
        "--seq-len", type=int, default=64, help="Sequence length for the synthetic batch"
    )
    parser.add_argument(
        "--batch-size", type=int, default=2, help="Batch size for the synthetic batch"
    )
    parser.add_argument(
        "--d-model", type=int, default=128, help="Hidden size used for the test model"
    )
    parser.add_argument(
        "--solver-type", choices=["alternating", "simultaneous", "cascade"], default="alternating"
    )
    parser.add_argument(
        "--max-iter", type=int, default=15, help="Maximum solver iterations to profile"
    )
    parser.add_argument("--seed", type=int, default=13, help="Random seed for reproducibility")
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> UnifiedMambaHopfieldDEQ:
    config = UnifiedModelConfig(
        vocab_size=256,
        d_model=args.d_model,
        d_state=max(32, args.d_model // 2),
        d_conv=4,
        n_layers=2,
        memory_size=256,
        solver_type=args.solver_type,
        max_iterations=args.max_iter,
        tolerance=1e-3,
    )
    model = UnifiedMambaHopfieldDEQ.from_config(config)
    model.solver.config.max_iter = args.max_iter
    model.solver.config.solver_type = args.solver_type
    return model.to("cpu")


def profile_solver() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    model = build_model(args)
    input_ids = torch.randint(
        0, model.output_proj[-1].out_features, (args.batch_size, args.seq_len)
    )

    durations: list[float] = []
    for run in range(args.runs):
        start = time.perf_counter()
        with torch.no_grad():
            logits, diagnostics = model(
                input_ids,
                update_memory=False,
                return_diagnostics=True,
            )
        end = time.perf_counter()
        duration_ms = (end - start) * 1_000
        durations.append(duration_ms)
        solver_info = diagnostics.get("solver_info", {})
        print(
            f"run={run:02d} time_ms={duration_ms:7.2f} iterations={solver_info.get('iterations')} "
            f"converged={solver_info.get('converged')} final_energy={solver_info.get('final_energy')}"
        )

    mean = statistics.fmean(durations)
    stdev = statistics.pstdev(durations) if len(durations) > 1 else 0.0
    print(f"\nmean_time_ms={mean:7.2f} stdev_ms={stdev:6.2f} runs={args.runs}")


if __name__ == "__main__":
    profile_solver()
