# Theoretical Foundations

This document will compile the mathematical derivations underpinning the unified energy framework.

## CPU Validation Workflow

While waiting for accelerator access you can still exercise most of the theoretical guarantees on CPU. The following snippet runs the full validation battery using the lightweight dummy fixtures shipped with the repository:

```python
import torch

from unified_energy.experiments.theory import ConvergenceValidator, ValidationConfig
from unified_energy.models.unified import UnifiedModelConfig, UnifiedMambaHopfieldDEQ

torch.manual_seed(0)

config = UnifiedModelConfig(vocab_size=64, d_model=32, d_state=16, n_layers=2, memory_size=64)
model = UnifiedMambaHopfieldDEQ.from_config(config)

validator = ConvergenceValidator(
    model,
    config=ValidationConfig(context_length=4, batch_size=1, max_perturbation_trials=2, stability_radius=0.05),
    device="cpu",
)

results = validator.run_all_tests()
print(results["summary"])
```

All of the validation utilities purposefully avoid GPU-only kernels, so the script above typically finishes in under a minute on a modern laptop. Individual tests can be invoked directly (e.g. `validator.test_contraction_property()`) when you want to focus on a single guarantee.

## Interpreting Diagnostics Without a GPU

Every solver invocation now surfaces the raw energy terms as tensors and reports fixed-point residuals even when convergence fails. The expanded test suite (`pytest tests/test_solver.py`) demonstrates two CPU-only scenarios you can replicate and adapt:

- `test_energy_gradient_receives_dynamics_output` ensures the energy gradient is evaluated against the actual dynamics output, confirming that the consistency penalty participates in descent.
- `test_solver_reports_non_convergence` drives a deliberately non-contractive system and asserts that the solver returns a detailed diagnostic payload instead of silently failing.

These tests double as reference code for building minimal reproduction cases when you iterate on new solver logic without hardware acceleration.

## Finding Friendly CPU Hyperparameters

The default random configuration rarely satisfies all theoretical checks. Use `scripts/scan_validation.py` to sweep a handful of small Mamba/Hopfield/solver settings and capture the best performers:

```bash
python scripts/scan_validation.py --runs 2 --out validation_scan.json
```

The tool prints a ranked table (highest mean score first) and optionally saves the full JSON report. Once you identify a configuration with a few passing metrics, pass its values back to `scripts/run_validation.py` (or set them inline in a notebook) to generate reproducible diagnostics.

For a quicker but noisier search, `scripts/random_validation.py` randomly samples the same space and stops when it finds a configuration with at least one successful metric:

```bash
python scripts/random_validation.py --samples 300 --out random_search.json
```

Because the models are randomly initialised, rerunning the validation may not reproduce the exact metric mix, but the saved JSON captures the full history so you can replay promising seeds.
