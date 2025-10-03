# Architecture Overview

The unified architecture is organised around three interacting subsystems:

1. **Dynamics core** – `src/unified_energy/core/` contains the Mamba-inspired
   temporal processor, the Hopfield energy functional, and the joint
   equilibrium solver that enforces both fixed-point and energy optimality.
2. **Model wrapper** – `src/unified_energy/models/unified.py` assembles the
   dynamics, solver, and memory bank into a full sequence model capable of
   iterative equilibrium inference and implicit-gradient backpropagation.
3. **Training and evaluation tooling** – `src/unified_energy/training/` now
   provides curriculum-aware training utilities that expose diagnostics needed
   for energy-based regularisation.

## Training Objective

`UnifiedTrainingObjective` combines four components:

* **Task loss** – standard cross-entropy between logits and targets.
* **Energy penalty** – encourages equilibria with low unified energy using the
  solver diagnostics.
* **Convergence cost** – normalises the solver iteration count to bias the
  model towards faster fixed-point attainment.
* **Stability regulariser** – estimates a local Lipschitz constant of the
  dynamics and penalises departures from the contraction target.

All scalar metrics are surfaced via the returned component dictionary to make
experiment tracking straightforward.

## Curriculum Trainer

`UnifiedModelTrainer` orchestrates solver parameter schedules, gradient-based
optimisation, validation, and lightweight checkpointing. Its curriculum
configuration controls the equilibrium solver depth and tolerance per training
phase, while optional Weights & Biases logging can be enabled without becoming
a hard dependency. The helper `train_epoch` function runs a bounded number of
steps for smoke tests or interactive prototyping.

## Notes for Future Contributors

* The solver diagnostics dictionary now always includes final energy statistics
  (even when convergence fails) so downstream utilities can rely on a consistent
  interface.
* Diagnostics returned from the unified model expose the processed input
  context, making it easy to compute Jacobian-vector products or additional
  stability metrics without repeating the forward pass.
* See `tests/test_training.py` for minimal examples demonstrating how to plug
  the trainer and objective together with a lightweight dummy model.

## Theoretical Validation Toolkit

The `unified_energy.experiments.theory` package introduces two lightweight
utilities designed to make the mathematical guarantees of the architecture
observable in practice:

* `ConvergenceValidator` executes four batteries that mirror the project plan:
  contraction estimates, monotonic energy descent checks, equilibrium stability
  perturbations, and Lyapunov-style energy tracking. Results are returned as
  plain dictionaries so higher level experiment runners (or notebooks) can log
  summaries without depending on plotting libraries.
* `EnergyLandscapeAnalyzer` samples two-dimensional slices of the energy
  surface, collects projected convergence trajectories, inspects basins of
  attraction, and projects stored memories via PCA. The helper is careful to run
  entirely on CPU-friendly tensor operations, making it safe to call inside unit
  tests while still returning NumPy arrays that can be visualised interactively
  when optional plotting packages are available.

Both utilities require only the public interfaces already exposed by the unified
model (`dynamics`, `energy_fn`, `solver`, and `memory_patterns`), so they remain
applicable to future architectural variants.
