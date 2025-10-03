# Unified Mamba-Hopfield-DEQ Architecture

A research implementation unifying three powerful paradigms:
- **Mamba**: Efficient selective state space models for sequence processing
- **Modern Hopfield Networks**: Associative memory with exponential capacity
- **Deep Equilibrium Models**: Implicit depth through fixed-point computation

## Installation

```bash
git clone https://github.com/your-repo/unified-mamba-hopfield-deq
cd unified-mamba-hopfield-deq
pip install -r requirements.txt
pip install -e .
```

## Quick Start

### Basic Usage

```python
import torch
from src.models.unified import UnifiedMambaHopfieldDEQ

# Initialize model
model = UnifiedMambaHopfieldDEQ(
    vocab_size=10000,
    d_model=512,
    d_state=64,
    memory_size=5000,
    solver_type='alternating'
)

# Forward pass
input_ids = torch.randint(0, 10000, (2, 128))  # (batch, seq_len)
logits = model(input_ids)  # (batch, seq_len, vocab_size)

# With diagnostics
logits, diagnostics = model(input_ids, return_diagnostics=True)
print(f"Converged: {diagnostics['solver_info']['converged']}")
print(f"Iterations: {diagnostics['solver_info']['iterations']}")
print(f"Final energy: {diagnostics['solver_info']['final_energy']:.4f}")
```

### Training

```python
from src.training.trainer import UnifiedModelTrainer
from src.training.objectives import UnifiedTrainingObjective

# Setup training
objective = UnifiedTrainingObjective(
    task_weight=1.0,
    energy_weight=0.1,
    convergence_weight=0.05
)

trainer = UnifiedModelTrainer(
    model=model,
    optimizer=torch.optim.AdamW(model.parameters(), lr=1e-4),
    train_loader=train_loader,
    val_loader=val_loader,
    objective=objective
)

# Train with curriculum
trainer.train(num_epochs=10)
```

### Generation

```python
# Autoregressive generation
prompt = "Once upon a time"
prompt_ids = tokenizer.encode(prompt)
prompt_tensor = torch.tensor(prompt_ids).unsqueeze(0)

generated = model.generate(
    prompt_tensor,
    max_length=100,
    temperature=0.8,
    top_k=50
)

print(tokenizer.decode(generated[0]))
```

## Architecture Details

### Information Flow

```
Input Tokens
    ↓
Embedding
    ↓
Mamba Layers (process sequence)
    ↓
Extract State z₀
    ↓
┌─────────────────────────────┐
│  DEQ Equilibrium Finding    │
│                             │
│  Iterate until convergence: │
│  1. Query Hopfield memory   │
│  2. Update Mamba state      │
│  3. Compute energy          │
│  4. Check convergence       │
└─────────────────────────────┘
    ↓
Equilibrium State z*
    ↓
Output Projection
    ↓
Logits
```

### Solver Options

Three solver modes available:

1. **Alternating** (default): Alternates between fixed-point and energy descent
   - Best for stability
   - Slower convergence

2. **Simultaneous**: Jointly optimizes both objectives
   - Faster convergence
   - Requires careful hyperparameter tuning

3. **Cascade**: Fixed-point first, then energy refinement
   - Good when objectives are well-aligned
   - Most efficient when it works

## Experiments

### 1. Theoretical Validation

Verify convergence properties:

```python
from experiments.theory.convergence_proofs import ConvergenceValidator

validator = ConvergenceValidator(model)
results = validator.run_all_tests()
```

### 2. Energy Landscape Analysis

Visualize energy surfaces:

```python
from experiments.theory.energy_analysis import EnergyLandscapeAnalyzer

analyzer = EnergyLandscapeAnalyzer(model)
analyzer.visualize_2d_slice(z_equilibrium, context)
analyzer.visualize_convergence_trajectories()
analyzer.visualize_basin_of_attraction(z_equilibrium, context)
```

### 3. Associative Recall

Test memory capabilities:

```python
from experiments.tasks.memory_tasks import AssociativeRecallExperiment

experiment = AssociativeRecallExperiment(model)
accuracies = experiment.evaluate(num_trials=100)
```

### 4. Continual Learning

Measure catastrophic forgetting:

```python
from experiments.tasks.memory_tasks import ContinualLearningExperiment

experiment = ContinualLearningExperiment(model, num_tasks=5)
history = experiment.run_continual_learning()
```

## Configuration

Key hyperparameters:

```python
config = {
    # Model architecture
    'd_model': 512,              # Hidden dimension
    'd_state': 64,               # SSM state dimension
    'd_conv': 4,                 # Convolution width
    'n_layers': 6,               # Number of Mamba layers

    # Memory
    'memory_size': 10000,        # Number of storable patterns
    'beta': 2.0,                 # Hopfield inverse temperature

    # DEQ solver
    'solver_type': 'alternating',
    'max_iterations': 30,
    'tol_fixedpoint': 1e-3,
    'tol_energy': 1e-3,

    # Training
    'learning_rate': 1e-4,
    'batch_size': 32,
    'gradient_clip': 1.0
}
```

## Performance Tips

### Memory Efficiency

```python
# Enable gradient checkpointing
model.dynamics.mamba.gradient_checkpointing_enable()

# Limit convergence iterations during training
model.solver.max_iter = 10  # Increase gradually

# Use mixed precision
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    logits = model(input_ids)
```

### Faster Convergence

```python
# Warm-start with previous equilibrium
previous_z = None
for batch in dataloader:
    logits, diagnostics = model(batch, z_init=previous_z)
    previous_z = diagnostics['z_equilibrium'].detach()
```

### Better Stability

```python
# Increase regularization
objective = UnifiedTrainingObjective(
    stability_weight=0.1,  # Encourage contractive dynamics
    contraction_target=0.8
)

# Reduce learning rate for memory patterns
optimizer = torch.optim.AdamW([
    {'params': [p for n, p in model.named_parameters() if 'memory' not in n]},
    {'params': [model.memory_patterns], 'lr': 1e-5}
], lr=1e-4)
```

## Troubleshooting

### Issue: DEQ doesn’t converge

**Solutions:**

- Reduce `max_iter` initially and gradually increase
- Increase `tolerance` during warmup
- Check Lipschitz constant (should be < 1)
- Add stability regularization

### Issue: NaN losses

**Solutions:**

- Enable gradient clipping (max_norm=1.0)
- Reduce learning rate
- Check energy function components (one might be exploding)
- Use mixed precision cautiously

### Issue: Slow training

**Solutions:**

- Reduce `max_iter` (quality vs speed tradeoff)
- Use ‘cascade’ solver (faster than ‘alternating’)
- Enable gradient checkpointing
- Batch multiple sequences efficiently

## Citation

If you use this code in your research, please cite:

```bibtex
@software{unified_mamba_hopfield_deq,
  title={Unified Mamba-Hopfield-DEQ Architecture},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/unified-mamba-hopfield-deq}
}
```

## License

MIT License - see LICENSE file for details

## Acknowledgments

Built on top of:

- [Mamba](https://github.com/state-spaces/mamba) by Gu & Dao
- Modern Hopfield Networks theory by Ramsauer et al.
- DEQ framework by Bai et al.
