"""Energy landscape analysis utilities for the unified model."""
from __future__ import annotations

import importlib
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm.auto import tqdm

plotly_spec = importlib.util.find_spec("plotly.graph_objects")
go = importlib.import_module("plotly.graph_objects") if plotly_spec is not None else None

sklearn_decomp_spec = importlib.util.find_spec("sklearn.decomposition")
if sklearn_decomp_spec is not None:
    sklearn_decomp = importlib.import_module("sklearn.decomposition")
    PCA = getattr(sklearn_decomp, "PCA")
else:
    PCA = None

sklearn_manifold_spec = importlib.util.find_spec("sklearn.manifold")
if sklearn_manifold_spec is not None:
    sklearn_manifold = importlib.import_module("sklearn.manifold")
    TSNE = getattr(sklearn_manifold, "TSNE")
else:
    TSNE = None

sklearn_cluster_spec = importlib.util.find_spec("sklearn.cluster")
if sklearn_cluster_spec is not None:
    sklearn_cluster = importlib.import_module("sklearn.cluster")
    KMeans = getattr(sklearn_cluster, "KMeans")
else:
    KMeans = None


class EnergyLandscapeAnalyzer:
    """Visualize and analyze the energy landscape of the unified model."""

    def __init__(self, model: Any, device: str = "cuda") -> None:
        self.model = model.to(device)
        self.device = device

    def _require_module(self, module: Optional[Any], name: str) -> Any:
        if module is None:
            raise ImportError(
                f"Optional dependency '{name}' is required for this analysis but is not installed."
            )
        return module

    def visualize_2d_slice(
        self,
        z_equilibrium: torch.Tensor,
        context: torch.Tensor,
        basis_vectors: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        resolution: int = 50,
        radius: float = 2.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Visualize 2D slice of energy landscape around an equilibrium."""

        print("Computing 2D energy slice...")

        PCA_module = self._require_module(PCA, "sklearn.decomposition.PCA")
        go_module = self._require_module(go, "plotly.graph_objects")

        if basis_vectors is None:
            patterns = self.model.memory_patterns.cpu().numpy()
            pca = PCA_module(n_components=2)
            pca.fit(patterns)
            v1 = torch.tensor(pca.components_[0], device=self.device, dtype=torch.float32)
            v2 = torch.tensor(pca.components_[1], device=self.device, dtype=torch.float32)
        else:
            v1, v2 = basis_vectors

        alpha = np.linspace(-radius, radius, resolution)
        beta = np.linspace(-radius, radius, resolution)
        A, B = np.meshgrid(alpha, beta)

        energies = np.zeros_like(A)
        fp_residuals = np.zeros_like(A)

        with torch.no_grad():
            for i in tqdm(range(resolution)):
                for j in range(resolution):
                    z = z_equilibrium + A[i, j] * v1 + B[i, j] * v2
                    z = z.unsqueeze(0)

                    z_next = self.model.dynamics(z, context, self.model.memory_patterns)
                    E, _ = self.model.energy_fn(z, z_next, self.model.memory_patterns)

                    energies[i, j] = E.item()
                    fp_residuals[i, j] = torch.norm(z - z_next).item()

        fig = go_module.Figure()
        fig.add_trace(
            go_module.Surface(
                x=A,
                y=B,
                z=energies,
                colorscale="Viridis",
                name="Energy",
                showscale=True,
                opacity=0.9,
            )
        )

        fig.add_trace(
            go_module.Scatter3d(
                x=[0],
                y=[0],
                z=[energies[resolution // 2, resolution // 2]],
                mode="markers",
                marker=dict(size=10, color="red"),
                name="Equilibrium",
            )
        )

        fig.update_layout(
            title="Energy Landscape (2D Slice)",
            scene=dict(
                xaxis_title="Direction 1",
                yaxis_title="Direction 2",
                zaxis_title="Energy",
            ),
            width=900,
            height=700,
        )

        fig.write_html("energy_landscape_2d.html")
        print("Saved to energy_landscape_2d.html")

        fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        im1 = ax1.contourf(A, B, energies, levels=20, cmap="viridis")
        ax1.set_title("Energy Landscape")
        ax1.set_xlabel("Direction 1")
        ax1.set_ylabel("Direction 2")
        ax1.plot(0, 0, "r*", markersize=15, label="Equilibrium")
        ax1.legend()
        plt.colorbar(im1, ax=ax1)

        im2 = ax2.contourf(A, B, fp_residuals, levels=20, cmap="plasma")
        ax2.set_title("Fixed-Point Residual")
        ax2.set_xlabel("Direction 1")
        ax2.set_ylabel("Direction 2")
        ax2.plot(0, 0, "r*", markersize=15)
        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()
        plt.savefig("energy_landscape_contours.png", dpi=150)
        print("Saved to energy_landscape_contours.png")

        return energies, fp_residuals

    def visualize_convergence_trajectories(
        self,
        num_trajectories: int = 10,
        num_steps: int = 50,
    ) -> Tuple[List[np.ndarray], List[List[float]]]:
        """Visualize multiple convergence trajectories in state space."""

        print("Computing convergence trajectories...")

        PCA_module = self._require_module(PCA, "sklearn.decomposition.PCA")

        trajectories: List[np.ndarray] = []
        energies_list: List[List[float]] = []

        with torch.no_grad():
            for _ in tqdm(range(num_trajectories)):
                z = torch.randn(1, self.model.d_model, device=self.device)
                context = torch.randn(1, 10, self.model.d_model, device=self.device)

                traj = [z.cpu().numpy().flatten()]
                energies: List[float] = []

                for _ in range(num_steps):
                    z_next = self.model.dynamics(z, context, self.model.memory_patterns)
                    E, _ = self.model.energy_fn(z, z_next, self.model.memory_patterns)

                    traj.append(z_next.cpu().numpy().flatten())
                    energies.append(E.item())

                    if torch.norm(z_next - z) < 1e-4:
                        break

                    z = z_next

                trajectories.append(np.array(traj))
                energies_list.append(energies)

        all_points = np.vstack(trajectories)
        pca = PCA_module(n_components=2)
        all_points_2d = pca.fit_transform(all_points)

        trajectories_2d: List[np.ndarray] = []
        idx = 0
        for traj in trajectories:
            traj_len = len(traj)
            trajectories_2d.append(all_points_2d[idx : idx + traj_len])
            idx += traj_len

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        for traj_2d in trajectories_2d:
            ax1.plot(traj_2d[:, 0], traj_2d[:, 1], "o-", alpha=0.6, markersize=3)
            ax1.plot(traj_2d[0, 0], traj_2d[0, 1], "go", markersize=8)
            ax1.plot(traj_2d[-1, 0], traj_2d[-1, 1], "r*", markersize=12)

        ax1.set_title("Convergence Trajectories (PCA projection)")
        ax1.set_xlabel("PC1")
        ax1.set_ylabel("PC2")
        ax1.grid(True, alpha=0.3)

        for energies in energies_list:
            ax2.plot(energies, alpha=0.6)

        ax2.set_title("Energy During Convergence")
        ax2.set_xlabel("Iteration")
        ax2.set_ylabel("Energy")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("convergence_trajectories.png", dpi=150)
        print("Saved to convergence_trajectories.png")

        return trajectories_2d, energies_list

    def visualize_basin_of_attraction(
        self,
        z_equilibrium: torch.Tensor,
        context: torch.Tensor,
        resolution: int = 30,
        radius: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Identify and visualize basin of attraction around an equilibrium."""

        print("Analyzing basin of attraction...")

        PCA_module = self._require_module(PCA, "sklearn.decomposition.PCA")

        patterns = self.model.memory_patterns.cpu().numpy()
        pca = PCA_module(n_components=2)
        pca.fit(patterns)
        v1 = torch.tensor(pca.components_[0], device=self.device, dtype=torch.float32)
        v2 = torch.tensor(pca.components_[1], device=self.device, dtype=torch.float32)

        alpha = np.linspace(-radius, radius, resolution)
        beta = np.linspace(-radius, radius, resolution)
        A, B = np.meshgrid(alpha, beta)

        convergence_map = np.zeros_like(A)
        final_energies = np.zeros_like(A)

        with torch.no_grad():
            for i in tqdm(range(resolution)):
                for j in range(resolution):
                    z_init = z_equilibrium + A[i, j] * v1 + B[i, j] * v2
                    z_init = z_init.unsqueeze(0)

                    z_final, info = self.model.solver.solve(
                        z_init, context, self.model.memory_patterns
                    )

                    convergence_map[i, j] = 1 if info.get("converged", False) else 0
                    final_energies[i, j] = info.get("final_energy", np.nan)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.contourf(A, B, convergence_map, levels=[-0.5, 0.5, 1.5], colors=["red", "green"], alpha=0.6)
        ax1.set_title("Basin of Attraction\n(Green = Converged, Red = Failed)")
        ax1.set_xlabel("Direction 1")
        ax1.set_ylabel("Direction 2")
        ax1.plot(0, 0, "k*", markersize=15, label="Target Equilibrium")
        ax1.legend()

        im2 = ax2.contourf(A, B, final_energies, levels=20, cmap="viridis")
        ax2.set_title("Final Energy Values")
        ax2.set_xlabel("Direction 1")
        ax2.set_ylabel("Direction 2")
        ax2.plot(0, 0, "r*", markersize=15)
        plt.colorbar(im2, ax=ax2)

        plt.tight_layout()
        plt.savefig("basin_of_attraction.png", dpi=150)
        print("Saved to basin_of_attraction.png")

        convergence_rate = float(convergence_map.mean())
        print(f"Convergence rate in explored region: {convergence_rate:.1%}")

        return convergence_map, final_energies

    def visualize_memory_organization(self) -> Tuple[np.ndarray, np.ndarray]:
        """Visualize how memory patterns are organized in state space."""

        print("Visualizing memory organization...")

        TSNE_module = self._require_module(TSNE, "sklearn.manifold.TSNE")

        patterns = self.model.memory_patterns.cpu().numpy()

        tsne = TSNE_module(n_components=2, random_state=42)
        patterns_2d = tsne.fit_transform(patterns)

        with torch.no_grad():
            similarities = torch.matmul(
                self.model.memory_patterns, self.model.memory_patterns.T
            ).cpu().numpy()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

        ax1.scatter(
            patterns_2d[:, 0],
            patterns_2d[:, 1],
            c=np.arange(len(patterns)),
            cmap="tab20",
            alpha=0.6,
        )
        ax1.set_title("Memory Pattern Organization (t-SNE)")
        ax1.set_xlabel("t-SNE 1")
        ax1.set_ylabel("t-SNE 2")

        im = ax2.imshow(similarities, cmap="viridis", aspect="auto")
        ax2.set_title("Memory Pattern Similarity Matrix")
        ax2.set_xlabel("Pattern Index")
        ax2.set_ylabel("Pattern Index")
        plt.colorbar(im, ax=ax2)

        plt.tight_layout()
        plt.savefig("memory_organization.png", dpi=150)
        print("Saved to memory_organization.png")

        KMeans_module = KMeans
        if KMeans_module is not None:
            n_clusters = min(10, max(1, len(patterns) // 100))
            if n_clusters > 1:
                kmeans = KMeans_module(n_clusters=n_clusters, random_state=42)
                clusters = kmeans.fit_predict(patterns)

                print("\nMemory clustering ({} clusters):".format(n_clusters))
                for i in range(n_clusters):
                    count = int((clusters == i).sum())
                    print(f"  Cluster {i}: {count} patterns ({count / len(patterns):.1%})")

        return patterns_2d, similarities

    def analyze_critical_points(self, num_samples: int = 100) -> List[Dict[str, Any]]:
        """Find and classify critical points of the energy function."""

        print("Analyzing critical points...")

        critical_points: List[Dict[str, Any]] = []

        with torch.no_grad():
            for _ in tqdm(range(num_samples)):
                z = torch.randn(1, self.model.d_model, device=self.device)
                context = torch.randn(1, 10, self.model.d_model, device=self.device)

                z_eq, info = self.model.solver.solve(z, context, self.model.memory_patterns)

                if not info.get("converged", False):
                    continue

                grad_norm = torch.norm(
                    self.model.energy_fn.energy_gradient(z_eq, self.model.memory_patterns)
                ).item()

                eigenvalues = self._estimate_hessian_eigenvalues(z_eq, context)

                critical_points.append(
                    {
                        "position": z_eq.cpu().numpy(),
                        "energy": info.get("final_energy"),
                        "grad_norm": grad_norm,
                        "eigenvalues": eigenvalues,
                        "type": self._classify_critical_point(eigenvalues),
                    }
                )

        types = [cp["type"] for cp in critical_points]
        type_counts = {t: types.count(t) for t in set(types)}

        print("\nCritical Point Analysis:")
        for cp_type, count in type_counts.items():
            print(f"  {cp_type}: {count} ({count / len(critical_points):.1%})")

        return critical_points

    def _estimate_hessian_eigenvalues(
        self,
        z: torch.Tensor,
        context: torch.Tensor,
        num_samples: int = 5,
    ) -> List[float]:
        """Estimate Hessian eigenvalues using finite differences."""

        eigenvalues: List[float] = []

        with torch.enable_grad():
            for _ in range(num_samples):
                v = torch.randn_like(z)
                v = v / torch.norm(v)

                z_param = z.detach().requires_grad_(True)
                z_next = self.model.dynamics(z_param, context, self.model.memory_patterns)
                E, _ = self.model.energy_fn(z_param, z_next, self.model.memory_patterns)

                grad_E = torch.autograd.grad(E, z_param, create_graph=True)[0]
                Hv = torch.autograd.grad(grad_E, z_param, v, retain_graph=False)[0]

                eigenval = (v * Hv).sum().item()
                eigenvalues.append(eigenval)

        return eigenvalues

    def _classify_critical_point(self, eigenvalues: Sequence[float]) -> str:
        """Classify critical point based on Hessian eigenvalues."""

        pos = sum(1 for e in eigenvalues if e > 0.01)
        neg = sum(1 for e in eigenvalues if e < -0.01)

        if pos == len(eigenvalues):
            return "Minimum (Stable)"
        if neg == len(eigenvalues):
            return "Maximum (Unstable)"
        return "Saddle Point"
