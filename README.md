# Curly Octo Goggles Model Optimisations

This repository now contains a compact neural network that
showcases both permanent and optional optimisation ideas. The goal is to make
it easy to experiment with small feature additions while keeping the training
loop transparent, dependency-free, and built entirely on the Python standard library.

## Highlights

- **Permanent upgrades**
  - Xavier/Glorot parameter initialisation for stable convergence.
  - Gradient clipping applied to every update to avoid exploding gradients.
  - Learning-rate decay using an inverse time schedule for smoother late-epoch
    optimisation.
- **Optional toggles**
  - Dropout regularisation on the hidden layer with configurable rate.
  - L2 weight decay for further control over model capacity.
  - Early stopping governed by patience/min-delta thresholds.

## Layout

```
model/
├── __init__.py          # Package exports
├── config.py            # Dataclasses describing model/training configuration
└── network.py           # Two-layer neural net with configurable optimisations

tests/
└── test_network.py      # Unit tests covering dropout and non-dropout training
```

The tests focus on learning the XOR pattern, ensuring that the optimisation
settings improve the model instead of destabilising it.

## Running the tests

```
python -m pytest
```

Running the suite only requires the Python standard library, so no external packages need to be installed.
