"""Plotting utilities for energy landscapes and convergence."""

from __future__ import annotations

from typing import Sequence

import matplotlib.pyplot as plt


def plot_energy_history(energies: Sequence[float]) -> None:
    """Plot energy values across solver iterations."""

    plt.figure()
    plt.plot(energies)
    plt.xlabel("Iteration")
    plt.ylabel("Energy")
    plt.title("Energy Trajectory")
    plt.tight_layout()
