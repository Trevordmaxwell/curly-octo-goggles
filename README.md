# Unified Energy Framework

This repository implements a research-grade energy-based framework that unifies Mamba temporal processing, Modern Hopfield Network memory, and Deep Equilibrium Model (DEQ) reasoning.

## Project Phases

The development roadmap follows ten sequential phases:

1. Repository structure and dependencies *(completed)*
2. Energy function design combining Hopfield retrieval with equilibrium consistency terms *(completed)*
3. Unified dynamics integrating Mamba-style state-space updates with Hopfield retrieval *(completed)*
4. Hybrid equilibrium solver supporting alternating, simultaneous, and cascade modes *(completed)*
5. Full model assembly with implicit-depth wrapper *(completed)*
6. Training infrastructure with curriculum learning for energy-based objectives
7. Theoretical validation experiments (convergence, Lyapunov, contraction analyses)
8. Practical task experiments (associative recall, continual learning benchmarks)
9. Energy landscape visualization tools
10. Documentation and interactive demo notebook

Each phase will be implemented in order with rigorous testing and documentation.

## Installation

The project uses `pyproject.toml` for dependency management. Install in editable mode with:

```bash
pip install -e .[dev]
```

## Repository Layout

```
src/unified_energy/
  analysis/
  core/
  models/
  solvers/
  training/
  utils/
  visualization/
experiments/
  theory/
  tasks/
  configs/
notebooks/
docs/
tests/
```

Each directory aligns with the architect's plan and now contains scaffolding or prototype implementations for the initial phases.
