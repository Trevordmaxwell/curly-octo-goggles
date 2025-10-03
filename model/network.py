"""Two-layer neural network implemented with only the Python standard library."""
from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
from typing import List, Sequence

from .config import EarlyStoppingConfig, ModelConfig

Matrix = List[List[float]]
Vector = List[float]


@dataclass
class TrainingHistory:
    """Container storing training metrics collected during :meth:`TwoLayerNN.fit`."""

    losses: list[float] = field(default_factory=list)


def zeros(rows: int, cols: int) -> Matrix:
    return [[0.0 for _ in range(cols)] for _ in range(rows)]


def matmul(a: Matrix, b: Matrix) -> Matrix:
    rows = len(a)
    shared = len(a[0])
    cols = len(b[0])
    result = zeros(rows, cols)
    for i in range(rows):
        for k in range(shared):
            aik = a[i][k]
            for j in range(cols):
                result[i][j] += aik * b[k][j]
    return result


def transpose(a: Matrix) -> Matrix:
    return [list(col) for col in zip(*a)]


def add_bias(matrix: Matrix, bias: Vector) -> Matrix:
    return [[value + bias[j] for j, value in enumerate(row)] for row in matrix]


def relu(matrix: Matrix) -> tuple[Matrix, Matrix]:
    activated = []
    mask = []
    for row in matrix:
        activated_row = []
        mask_row = []
        for value in row:
            if value > 0.0:
                activated_row.append(value)
                mask_row.append(1.0)
            else:
                activated_row.append(0.0)
                mask_row.append(0.0)
        activated.append(activated_row)
        mask.append(mask_row)
    return activated, mask


def apply_dropout(matrix: Matrix, rate: float, rng: random.Random) -> tuple[Matrix, Matrix | None]:
    if rate <= 0.0:
        return matrix, None
    keep_prob = 1.0 - rate
    if not 0.0 < keep_prob <= 1.0:
        raise ValueError("Dropout keep probability must be within (0, 1].")
    dropped = []
    mask = []
    scale = 1.0 / keep_prob
    for row in matrix:
        dropped_row = []
        mask_row = []
        for value in row:
            if rng.random() < keep_prob:
                dropped_row.append(value * scale)
                mask_row.append(scale)
            else:
                dropped_row.append(0.0)
                mask_row.append(0.0)
        dropped.append(dropped_row)
        mask.append(mask_row)
    return dropped, mask


def softmax(logits: Matrix) -> Matrix:
    probs: Matrix = []
    for row in logits:
        max_val = max(row)
        exps = [math.exp(value - max_val) for value in row]
        total = sum(exps)
        probs.append([value / total for value in exps])
    return probs


def cross_entropy(probs: Matrix, labels: Sequence[int]) -> float:
    eps = 1e-12
    total = 0.0
    for prob_row, label in zip(probs, labels):
        total -= math.log(prob_row[label] + eps)
    return total / len(labels)


def subtract_matrices(a: Matrix, b: Matrix) -> Matrix:
    return [[va - vb for va, vb in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def sum_columns(matrix: Matrix) -> Vector:
    cols = len(matrix[0])
    totals = [0.0 for _ in range(cols)]
    for row in matrix:
        for j in range(cols):
            totals[j] += row[j]
    return totals


def scalar_multiply(matrix: Matrix, scalar: float) -> Matrix:
    return [[value * scalar for value in row] for row in matrix]


def scalar_multiply_vector(vector: Vector, scalar: float) -> Vector:
    return [value * scalar for value in vector]


def add_matrices(a: Matrix, b: Matrix) -> Matrix:
    return [[va + vb for va, vb in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def add_vectors(a: Vector, b: Vector) -> Vector:
    return [va + vb for va, vb in zip(a, b)]


def elementwise_multiply(a: Matrix, b: Matrix) -> Matrix:
    return [[va * vb for va, vb in zip(row_a, row_b)] for row_a, row_b in zip(a, b)]


def l2_norm_squared(matrix: Matrix) -> float:
    return sum(value * value for row in matrix for value in row)


class TwoLayerNN:
    """A tiny neural network with optional regularisation features.

    The implementation purposefully avoids external dependencies so it can run
    in constrained environments. It supports the following improvements:

    * **Permanent**: Xavier/Glorot parameter initialisation and gradient
      clipping for stable training.
    * **Permanent**: Softmax cross-entropy with inverse time learning rate
      decay.
    * **Optional**: Dropout on the hidden activations.
    * **Optional**: L2 weight decay.
    * **Optional**: Early stopping on the training loss.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.params = self._initialise_parameters()
        self._learning_rate = config.learning_rate

    def _initialise_parameters(self) -> dict[str, Matrix | Vector]:
        def glorot(scale_in: int, scale_out: int) -> float:
            limit = math.sqrt(6.0 / (scale_in + scale_out))
            return self.rng.uniform(-limit, limit)

        w1 = [[glorot(self.config.input_dim, self.config.hidden_dim) for _ in range(self.config.hidden_dim)] for _ in range(self.config.input_dim)]
        bias_init = 0.1
        b1 = [bias_init for _ in range(self.config.hidden_dim)]
        w2 = [[glorot(self.config.hidden_dim, self.config.output_dim) for _ in range(self.config.output_dim)] for _ in range(self.config.hidden_dim)]
        b2 = [0.0 for _ in range(self.config.output_dim)]
        return {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

    def _forward(self, x: Matrix, training: bool = False) -> tuple[Matrix, dict[str, Matrix | None]]:
        z1 = add_bias(matmul(x, self.params["w1"]), self.params["b1"])  # type: ignore[arg-type]
        hidden, relu_mask = relu(z1)
        cache: dict[str, Matrix | None] = {"x": x, "z1": z1, "relu_mask": relu_mask}

        dropout_mask: Matrix | None
        if self.config.use_dropout and training:
            hidden, dropout_mask = apply_dropout(hidden, self.config.dropout_rate, self.rng)
        else:
            dropout_mask = None
        cache["dropout_mask"] = dropout_mask
        cache["hidden"] = hidden

        logits = add_bias(matmul(hidden, self.params["w2"]), self.params["b2"])  # type: ignore[arg-type]
        probs = softmax(logits)
        cache["probs"] = probs
        return probs, cache

    def _loss_and_grads(self, x: Matrix, y: Sequence[int]) -> tuple[float, dict[str, Matrix | Vector]]:
        probs, cache = self._forward(x, training=True)
        num_samples = len(x)
        loss = cross_entropy(probs, y)

        # Gradient on output weights
        y_one_hot = zeros(num_samples, self.config.output_dim)
        for i, label in enumerate(y):
            y_one_hot[i][label] = 1.0
        dz2 = scalar_multiply(subtract_matrices(probs, y_one_hot), 1.0 / num_samples)
        hidden = cache["hidden"]  # type: ignore[assignment]
        grads_w2 = matmul(transpose(hidden), dz2)
        grads_b2 = scalar_multiply_vector(sum_columns(dz2), 1.0)

        # Backpropagate into hidden layer
        w2 = self.params["w2"]  # type: ignore[assignment]
        dhidden = matmul(dz2, transpose(w2))
        relu_mask = cache["relu_mask"]  # type: ignore[assignment]
        dhidden = elementwise_multiply(dhidden, relu_mask)

        dropout_mask = cache["dropout_mask"]
        if dropout_mask is not None:
            dhidden = elementwise_multiply(dhidden, dropout_mask)

        x_matrix = cache["x"]  # type: ignore[assignment]
        grads_w1 = matmul(transpose(x_matrix), dhidden)
        grads_b1 = scalar_multiply_vector(sum_columns(dhidden), 1.0)

        if self.config.l2_strength:
            loss += 0.5 * self.config.l2_strength * (
                l2_norm_squared(self.params["w1"]) + l2_norm_squared(self.params["w2"])  # type: ignore[arg-type]
            )
            grads_w1 = add_matrices(grads_w1, scalar_multiply(self.params["w1"], self.config.l2_strength))  # type: ignore[arg-type]
            grads_w2 = add_matrices(grads_w2, scalar_multiply(self.params["w2"], self.config.l2_strength))  # type: ignore[arg-type]

        grads = {"w1": grads_w1, "b1": grads_b1, "w2": grads_w2, "b2": grads_b2}
        return loss, grads

    def _apply_gradients(self, grads: dict[str, Matrix | Vector]) -> None:
        clip_value = self.config.gradient_clip
        for key, grad in grads.items():
            if key.startswith('w'):
                grad_matrix = grad  # type: ignore[assignment]
                if clip_value:
                    grad_matrix = [[max(min(value, clip_value), -clip_value) for value in row] for row in grad_matrix]
                param_matrix = self.params[key]  # type: ignore[assignment]
                updated = add_matrices(param_matrix, scalar_multiply(grad_matrix, -self._learning_rate))
                self.params[key] = updated  # type: ignore[assignment]
            else:
                grad_vector = grad  # type: ignore[assignment]
                if clip_value:
                    grad_vector = [max(min(value, clip_value), -clip_value) for value in grad_vector]  # type: ignore[list-item]
                param_vector = self.params[key]  # type: ignore[assignment]
                updated_vec = add_vectors(param_vector, scalar_multiply_vector(grad_vector, -self._learning_rate))
                self.params[key] = updated_vec  # type: ignore[assignment]

    def fit(
        self,
        x: Matrix,
        y: Sequence[int],
        *,
        epochs: int = 500,
        early_stopping: EarlyStoppingConfig | None = None,
    ) -> TrainingHistory:
        """Train the model on the provided dataset."""

        history = TrainingHistory()
        best_loss = float("inf")
        epochs_without_improvement = 0

        for epoch in range(1, epochs + 1):
            loss, grads = self._loss_and_grads(x, y)
            history.losses.append(loss)
            self._apply_gradients(grads)

            if self.config.learning_rate_decay:
                self._learning_rate = self.config.learning_rate / (1.0 + self.config.learning_rate_decay * epoch)

            if early_stopping is not None:
                if loss + early_stopping.min_delta < best_loss:
                    best_loss = loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    if epochs_without_improvement >= early_stopping.patience:
                        break

        return history

    def predict(self, x: Matrix) -> list[int]:
        probs, _ = self._forward(x, training=False)
        return [int(max(range(len(row)), key=row.__getitem__)) for row in probs]

    def predict_proba(self, x: Matrix) -> Matrix:
        probs, _ = self._forward(x, training=False)
        return probs

    def parameters(self) -> dict[str, Matrix | Vector]:
        def copy_param(value: Matrix | Vector) -> Matrix | Vector:
            if isinstance(value, list) and value and isinstance(value[0], list):
                return [row.copy() for row in value]  # type: ignore[list-item]
            return value.copy()  # type: ignore[return-value]

        return {name: copy_param(value) for name, value in self.params.items()}
