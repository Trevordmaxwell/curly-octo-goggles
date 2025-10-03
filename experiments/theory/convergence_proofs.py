"""Convergence validation experiments for the unified model."""
from __future__ import annotations

from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm


class ConvergenceValidator:
    """Empirically validate theoretical convergence properties."""

    def __init__(self, model: Any, device: str = "cuda") -> None:
        self.model = model.to(device)
        self.device = device

    def test_contraction_property(self, num_samples: int = 100) -> Dict[str, float]:
        """Test if dynamics are contractive."""

        print("Testing contraction property...")

        lipschitz_constants = []

        with torch.no_grad():
            for _ in tqdm(range(num_samples)):
                z1 = torch.randn(1, self.model.d_model, device=self.device)
                z2 = torch.randn(1, self.model.d_model, device=self.device)

                context = torch.randn(1, 10, self.model.d_model, device=self.device)

                f_z1 = self.model.dynamics(z1, context, self.model.memory_patterns)
                f_z2 = self.model.dynamics(z2, context, self.model.memory_patterns)

                numerator = torch.norm(f_z1 - f_z2)
                denominator = torch.norm(z1 - z2)

                if denominator > 1e-6:
                    lipschitz_constants.append((numerator / denominator).item())

        mean_L = float(np.mean(lipschitz_constants)) if lipschitz_constants else float("nan")
        max_L = float(np.max(lipschitz_constants)) if lipschitz_constants else float("nan")
        pct_contractive = float(np.mean([L < 1.0 for L in lipschitz_constants])) if lipschitz_constants else 0.0

        print(f"Lipschitz constant: mean={mean_L:.4f}, max={max_L:.4f}")
        print(f"Contractive: {pct_contractive:.1%} of samples")

        return {
            "mean_lipschitz": mean_L,
            "max_lipschitz": max_L,
            "contractive_percentage": pct_contractive,
            "is_contraction": max_L < 1.0 if lipschitz_constants else False,
        }

    def test_energy_descent(
        self, num_trajectories: int = 50, num_steps: int = 30
    ) -> Dict[str, Any]:
        """Verify that energy decreases along trajectories."""

        print("Testing energy descent property...")

        descent_violations = 0
        energy_trajectories = []

        with torch.no_grad():
            for _ in tqdm(range(num_trajectories)):
                z = torch.randn(1, self.model.d_model, device=self.device)
                context = torch.randn(1, 10, self.model.d_model, device=self.device)

                energies = []

                for _ in range(num_steps):
                    z_next = self.model.dynamics(z, context, self.model.memory_patterns)
                    E, _ = self.model.energy_fn(z, z_next, self.model.memory_patterns)
                    energies.append(E.item())
                    z = z_next

                energy_trajectories.append(energies)

                for i in range(len(energies) - 1):
                    if energies[i + 1] > energies[i] + 1e-4:
                        descent_violations += 1
                        break

        violation_rate = descent_violations / max(1, num_trajectories)

        print(f"Energy descent violations: {violation_rate:.1%}")

        plt.figure(figsize=(10, 6))
        for traj in energy_trajectories[:10]:
            plt.plot(traj, alpha=0.5)
        plt.xlabel("Iteration")
        plt.ylabel("Energy")
        plt.title("Energy Trajectories")
        plt.savefig("energy_descent.png")

        return {
            "violation_rate": violation_rate,
            "energy_trajectories": energy_trajectories,
            "monotonic_descent": violation_rate < 0.1,
        }

    def test_fixed_point_stability(
        self, num_fixed_points: int = 20, num_perturbations: int = 10
    ) -> Dict[str, Any]:
        """Test stability of converged fixed points."""

        print("Testing fixed-point stability...")

        stable_count = 0

        with torch.no_grad():
            for _ in tqdm(range(num_fixed_points)):
                z_init = torch.randn(1, self.model.d_model, device=self.device)
                context = torch.randn(1, 10, self.model.d_model, device=self.device)

                z_eq, info = self.model.solver.solve(z_init, context, self.model.memory_patterns)

                if not info.get("converged", False):
                    continue

                is_stable = True

                for _ in range(num_perturbations):
                    epsilon = 0.01
                    perturbation = epsilon * torch.randn_like(z_eq)
                    z_perturbed = z_eq + perturbation

                    z_reconverged, _ = self.model.solver.solve(
                        z_perturbed, context, self.model.memory_patterns
                    )

                    distance = torch.norm(z_reconverged - z_eq)

                    if distance > 0.1:
                        is_stable = False
                        break

                if is_stable:
                    stable_count += 1

        stability_rate = stable_count / max(1, num_fixed_points)

        print(f"Stable fixed points: {stability_rate:.1%}")

        return {
            "stability_rate": stability_rate,
            "is_stable": stability_rate > 0.8,
        }

    def test_lyapunov_function(self, num_samples: int = 50) -> Dict[str, Any]:
        """Verify energy function acts as Lyapunov function."""

        print("Testing Lyapunov property...")

        lyapunov_satisfied = 0

        with torch.no_grad():
            for _ in tqdm(range(num_samples)):
                z = torch.randn(1, self.model.d_model, device=self.device)
                context = torch.randn(1, 10, self.model.d_model, device=self.device)

                is_lyapunov = True

                for _ in range(20):
                    z_next = self.model.dynamics(z, context, self.model.memory_patterns)

                    E_current, _ = self.model.energy_fn(z, z_next, self.model.memory_patterns)
                    E_next, _ = self.model.energy_fn(z_next, z_next, self.model.memory_patterns)

                    if E_next > E_current + 1e-4:
                        is_lyapunov = False
                        break

                    if torch.norm(z_next - z) < 1e-3:
                        break

                    z = z_next

                if is_lyapunov:
                    lyapunov_satisfied += 1

        lyapunov_rate = lyapunov_satisfied / max(1, num_samples)

        print(f"Lyapunov property satisfied: {lyapunov_rate:.1%}")

        return {
            "lyapunov_rate": lyapunov_rate,
            "is_lyapunov": lyapunov_rate > 0.9,
        }

    def run_all_tests(self) -> Dict[str, Dict[str, Any]]:
        """Run complete convergence validation suite."""

        print("=" * 50)
        print("CONVERGENCE VALIDATION SUITE")
        print("=" * 50)

        results = {}

        results["contraction"] = self.test_contraction_property()
        results["energy_descent"] = self.test_energy_descent()
        results["stability"] = self.test_fixed_point_stability()
        results["lyapunov"] = self.test_lyapunov_function()

        all_pass = (
            results["contraction"]["is_contraction"]
            and results["energy_descent"]["monotonic_descent"]
            and results["stability"]["is_stable"]
            and results["lyapunov"]["is_lyapunov"]
        )

        print("\n" + "=" * 50)
        print("SUMMARY")
        print("=" * 50)
        print(f"Contraction: {'✓' if results['contraction']['is_contraction'] else '✗'}")
        print(f"Energy Descent: {'✓' if results['energy_descent']['monotonic_descent'] else '✗'}")
        print(f"Stability: {'✓' if results['stability']['is_stable'] else '✗'}")
        print(f"Lyapunov: {'✓' if results['lyapunov']['is_lyapunov'] else '✗'}")
        print(f"\nOverall: {'✓ ALL TESTS PASSED' if all_pass else '✗ SOME TESTS FAILED'}")
        print("=" * 50)

        return results
