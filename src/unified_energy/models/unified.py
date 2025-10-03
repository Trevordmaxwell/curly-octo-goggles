"""Full unified architecture combining Mamba, Hopfield, and DEQ reasoning."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..core.dynamics import UnifiedDynamics
from ..core.energy import EnergyHyperParameters, UnifiedEnergyFunction
from ..core.mamba import MambaLayer
from ..solvers.hybrid_solver import SolverConfig, UnifiedEquilibriumSolver


@dataclass(slots=True)
class UnifiedModelConfig:
    """Configuration container for :class:`UnifiedMambaHopfieldDEQ`."""

    vocab_size: int
    d_model: int = 512
    d_state: int = 64
    d_conv: int = 4
    n_layers: int = 6
    memory_size: int = 1024
    beta: float = 2.0
    alpha: float = 1.0
    lambda_l2: float = 1e-2
    lambda_smooth: float = 1e-3
    solver_type: str = "alternating"
    max_iterations: int = 30
    tolerance: float = 1e-3

    def solver_config(self) -> SolverConfig:
        """Create a :class:`SolverConfig` reflecting the model settings."""

        return SolverConfig(
            max_iter=self.max_iterations,
            tol_fixedpoint=self.tolerance,
            tol_energy=self.tolerance,
            solver_type=self.solver_type,
        )

    def energy_hyperparameters(self) -> EnergyHyperParameters:
        """Return hyper-parameters for :class:`UnifiedEnergyFunction`."""

        return EnergyHyperParameters(
            beta=self.beta,
            alpha=self.alpha,
            lambda_l2=self.lambda_l2,
            lambda_smooth=self.lambda_smooth,
        )

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.d_state <= 0:
            raise ValueError("d_state must be positive")
        if self.d_conv <= 0:
            raise ValueError("d_conv must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.memory_size <= 0:
            raise ValueError("memory_size must be positive")
        if self.beta <= 0:
            raise ValueError("beta must be positive")
        if self.alpha < 0:
            raise ValueError("alpha must be non-negative")
        if self.lambda_l2 < 0:
            raise ValueError("lambda_l2 must be non-negative")
        if self.lambda_smooth < 0:
            raise ValueError("lambda_smooth must be non-negative")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.tolerance <= 0:
            raise ValueError("tolerance must be positive")


class UnifiedMambaHopfieldDEQ(nn.Module):
    """Unified architecture with Mamba dynamics, Hopfield memory, and DEQ solver."""

    @classmethod
    def from_config(
        cls,
        config: UnifiedModelConfig,
        **overrides: object,
    ) -> "UnifiedMambaHopfieldDEQ":
        """Instantiate the model from :class:`UnifiedModelConfig` values."""

        params: Dict[str, object] = {
            "d_model": config.d_model,
            "d_state": config.d_state,
            "d_conv": config.d_conv,
            "n_layers": config.n_layers,
            "memory_size": config.memory_size,
            "beta": config.beta,
            "alpha": config.alpha,
            "lambda_l2": config.lambda_l2,
            "lambda_smooth": config.lambda_smooth,
            "solver_type": config.solver_type,
            "max_iterations": config.max_iterations,
            "tol": config.tolerance,
        }
        extra = dict(overrides)
        solver_config = extra.pop("solver_config", config.solver_config())
        params.update(extra)
        return cls(
            config.vocab_size,
            solver_config=solver_config,  # type: ignore[arg-type]
            **params,
        )

    def __init__(
        self,
        vocab_size: int,
        *,
        d_model: int = 512,
        d_state: int = 64,
        d_conv: int = 4,
        n_layers: int = 6,
        memory_size: int = 1024,
        beta: float = 2.0,
        alpha: float = 1.0,
        lambda_l2: float = 1e-2,
        lambda_smooth: float = 1e-3,
        solver_type: str = "alternating",
        max_iterations: int = 30,
        tol: float = 1e-3,
        solver_config: Optional[SolverConfig] = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if memory_size <= 0:
            raise ValueError("memory_size must be positive")
        if n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if beta <= 0:
            raise ValueError("beta must be positive")
        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if lambda_l2 < 0:
            raise ValueError("lambda_l2 must be non-negative")
        if lambda_smooth < 0:
            raise ValueError("lambda_smooth must be non-negative")

        self.d_model = d_model
        self.memory_size = memory_size
        self.n_layers = n_layers
        self.novelty_threshold = 0.7

        # Input embedding and Mamba backbone
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.mamba_layers = nn.ModuleList(
            [MambaLayer(d_model, d_state, d_conv) for _ in range(n_layers)]
        )

        # Unified core components
        self.dynamics = UnifiedDynamics(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            beta=beta,
        )
        energy_hyper = EnergyHyperParameters(
            beta=beta,
            alpha=alpha,
            lambda_l2=lambda_l2,
            lambda_smooth=lambda_smooth,
        )
        self.energy_hyper = energy_hyper
        self.energy_fn = UnifiedEnergyFunction(d_model=d_model, hyper=energy_hyper)
        self.solver = UnifiedEquilibriumSolver(
            self.dynamics,
            self.energy_fn,
            solver_config
            or SolverConfig(
                max_iter=max_iterations,
                tol_fixedpoint=tol,
                tol_energy=tol,
                solver_type=solver_type,
            ),
        )

        # Learnable memory bank and writing mechanism
        memory_init = torch.randn(memory_size, d_model) / np.sqrt(d_model)
        self.register_buffer("memory_patterns", memory_init)
        self.register_buffer("memory_cursor", torch.zeros(1, dtype=torch.long))
        self.memory_write_gate = nn.Linear(d_model, 1)

        # Output projection to vocabulary space
        self.output_proj = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, vocab_size))

        # Runtime tracking utilities
        self.convergence_stats: List[Dict[str, object]] = []
        self._last_context: Optional[Tensor] = None

    def process_context(self, input_ids: Tensor) -> Tuple[Tensor, Tensor]:
        """Process token inputs with Mamba layers to obtain context and initial state."""

        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape (batch, length)")
        context = self.embedding(input_ids)
        state: Optional[Tensor] = None
        for layer in self.mamba_layers:
            state = layer(context, state=state)
            context = context + state.unsqueeze(1)
        assert state is not None  # for mypy
        return context, state

    def forward(
        self,
        input_ids: Tensor,
        *,
        update_memory: bool = True,
        return_diagnostics: bool = False,
    ) -> Tensor | Tuple[Tensor, Dict[str, object]]:
        """Run the full unified model on ``input_ids``."""

        _batch_size, seq_len = input_ids.shape
        context, z_init = self.process_context(input_ids)
        self._last_context = context.detach()

        z_equilibrium, solver_info = self.solver.solve(
            z_init=z_init,
            x_context=context,
            memory_patterns=self.memory_patterns,
        )

        if self.training:
            stats = {
                "converged": solver_info.get("converged", False),
                "iterations": solver_info.get("iterations", None),
                "final_energy": solver_info.get("final_energy", None),
                "energy_components": solver_info.get("energy_components", None),
            }
            self.convergence_stats.append(stats)
            if len(self.convergence_stats) > 100:
                self.convergence_stats.pop(0)

        if update_memory and (self.training or return_diagnostics):
            self.update_memory(z_equilibrium)

        z_expanded = z_equilibrium.unsqueeze(1).expand(-1, seq_len, -1)
        combined = context + z_expanded
        logits = self.output_proj(combined)

        if return_diagnostics:
            diagnostics = {
                "solver_info": solver_info,
                "z_equilibrium": z_equilibrium,
                "z_init": z_init,
                "energy_trajectory": solver_info.get("energy_history", []),
                "memory_usage": self._compute_memory_usage(z_equilibrium),
                "context": context,
            }
            return logits, diagnostics
        return logits

    def update_memory(self, z_equilibrium: Tensor, should_store: Optional[Tensor] = None) -> None:
        """Update the memory bank using a queue-based policy."""

        with torch.no_grad():
            if should_store is None:
                similarities = z_equilibrium @ self.memory_patterns.t()
                max_similarity = similarities.max(dim=-1).values
                gate_logits = self.memory_write_gate(z_equilibrium).squeeze(-1)
                gate_prob = torch.sigmoid(gate_logits)
                should_store = (max_similarity < self.novelty_threshold) & (gate_prob > 0.5)

            if should_store.dtype != torch.bool:
                should_store = should_store.to(torch.bool)

            indices = torch.nonzero(should_store, as_tuple=False).flatten()
            if indices.numel() == 0:
                return

            cursor = int(self.memory_cursor.item())
            for idx in indices:
                pattern = z_equilibrium[idx].detach()
                self.memory_patterns[cursor] = pattern
                cursor = (cursor + 1) % self.memory_size
            self.memory_cursor.fill_(cursor)

    def _compute_memory_usage(self, z_equilibrium: Tensor) -> Dict[str, float]:
        """Return statistics describing memory retrieval behaviour."""

        with torch.no_grad():
            similarities = z_equilibrium @ self.memory_patterns.t()
            attention = F.softmax(similarities, dim=-1)
            entropy = -torch.sum(attention * torch.log(attention + 1e-10), dim=-1).mean()
            top_k = min(10, attention.shape[-1])
            top_mass = attention.topk(top_k, dim=-1).values.sum(dim=-1).mean()
            return {
                "attention_entropy": float(entropy),
                "top_10_mass": float(top_mass),
                "num_patterns": float(self.memory_patterns.shape[0]),
            }

    def get_implicit_gradients(self, z_equilibrium: Tensor, loss: Tensor) -> Tensor:
        """Compute gradients via implicit differentiation as in DEQ models."""

        if self._last_context is None:
            raise RuntimeError("Context is unavailable for implicit gradient computation")

        with torch.enable_grad():
            z_eq = z_equilibrium.detach().requires_grad_(True)

            def jvp(vector: Tensor) -> Tensor:
                z_next = self.dynamics(z_eq, self._last_context, self.memory_patterns)
                return torch.autograd.grad(
                    z_next,
                    z_eq,
                    vector,
                    retain_graph=True,
                    create_graph=True,
                    allow_unused=False,
                )[0]

            g_loss = torch.autograd.grad(loss, z_eq, retain_graph=True)[0]
            implicit_grad = self._solve_implicit_gradient(jvp, g_loss)
        return implicit_grad

    def _solve_implicit_gradient(
        self,
        jvp_fn,
        g: Tensor,
        *,
        max_iter: int = 10,
        tol: float = 1e-3,
    ) -> Tensor:
        """Solve ``(I - J) x = g`` using conjugate gradient with matrix-free ops."""

        x = g.clone()
        r = g - (x - jvp_fn(x))
        p = r.clone()
        for _ in range(max_iter):
            Ap = p - jvp_fn(p)
            r_dot = torch.dot(r.flatten(), r.flatten())
            if r_dot < tol:
                break
            alpha = r_dot / torch.dot(p.flatten(), Ap.flatten())
            x = x + alpha * p
            r_new = r - alpha * Ap
            beta = torch.dot(r_new.flatten(), r_new.flatten()) / r_dot
            p = r_new + beta * p
            r = r_new
        return x

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: Tensor,
        *,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
    ) -> Tensor:
        """Autoregressive generation using equilibrium-driven decoding."""

        generated = prompt_ids.clone()
        for _ in range(max_length):
            logits, diagnostics = self.forward(
                generated,
                update_memory=False,
                return_diagnostics=True,
            )
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-5)
            if top_k > 0:
                kth_values = torch.topk(next_token_logits, top_k, dim=-1).values[..., -1, None]
                mask = next_token_logits < kth_values
                next_token_logits = next_token_logits.masked_fill(mask, float("-inf"))
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_token], dim=-1)

            solver_info = diagnostics["solver_info"]
            if solver_info.get("converged", False):
                should_store = torch.ones(
                    next_token.shape[0], dtype=torch.bool, device=generated.device
                )
                self.update_memory(diagnostics["z_equilibrium"], should_store=should_store)
        return generated


# Backwards compatibility for earlier imports
UnifiedModel = UnifiedMambaHopfieldDEQ

