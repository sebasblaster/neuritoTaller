from dataclasses import dataclass

import numpy as np


@dataclass
class Network:
    weights1: np.ndarray
    bias1: np.ndarray
    weights2: np.ndarray
    bias2: np.ndarray

    def predict(self, data: np.ndarray) -> np.ndarray:
        hidden = relu(data @ self.weights1 + self.bias1)
        return softmax(hidden @ self.weights2 + self.bias2)


def relu(values: np.ndarray) -> np.ndarray:
    return np.maximum(0, values)


def softmax(values: np.ndarray) -> np.ndarray:
    exp_values = np.exp(values - np.max(values, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def make_dataset(size: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    data = rng.random((size, 3))
    targets = np.argmax(
        np.column_stack((data[:, 0], 1 - data[:, 1], data[:, 2])), axis=1
    )
    return data, targets


def train_network(
    data: np.ndarray,
    targets: np.ndarray,
    hidden_size: int = 6,
    epochs: int = 1200,
    learning_rate: float = 0.1,
    seed: int = 42,
) -> tuple[Network, list[float]]:
    rng = np.random.default_rng(seed)
    weights1 = rng.standard_normal((3, hidden_size)) * 0.5
    bias1 = np.zeros((1, hidden_size))
    weights2 = rng.standard_normal((hidden_size, 3)) * 0.5
    bias2 = np.zeros((1, 3))
    history = []
    count = len(data)

    for _ in range(epochs):
        z1 = data @ weights1 + bias1
        hidden = relu(z1)
        probabilities = softmax(hidden @ weights2 + bias2)
        loss = -np.mean(np.log(probabilities[np.arange(count), targets] + 1e-10))
        history.append(float(loss))

        delta2 = probabilities.copy()
        delta2[np.arange(count), targets] -= 1
        delta2 /= count
        grad_weights2 = hidden.T @ delta2
        grad_bias2 = np.sum(delta2, axis=0, keepdims=True)
        delta1 = (delta2 @ weights2.T) * (z1 > 0)
        grad_weights1 = data.T @ delta1
        grad_bias1 = np.sum(delta1, axis=0, keepdims=True)
        weights1 -= learning_rate * grad_weights1
        bias1 -= learning_rate * grad_bias1
        weights2 -= learning_rate * grad_weights2
        bias2 -= learning_rate * grad_bias2

    return Network(weights1, bias1, weights2, bias2), history


def evaluate(
    network: Network, data: np.ndarray, targets: np.ndarray
) -> tuple[float, float]:
    probabilities = network.predict(data)
    loss = -np.mean(np.log(probabilities[np.arange(len(data)), targets] + 1e-10))
    accuracy = np.mean(np.argmax(probabilities, axis=1) == targets)
    return float(loss), float(accuracy)
