"""Red neuronal manual para decidir cuándo saltar."""

import numpy as np


class RedSalto:
    def __init__(self, semilla: int = 42, ocultas: int = 12, tasa: float = 0.8):
        self.rng = np.random.default_rng(semilla)
        self.ocultas = ocultas
        self.tasa = tasa
        self.reiniciar()

    def reiniciar(self) -> None:
        self.W1 = self.rng.normal(0, 0.65, (3, self.ocultas))
        self.b1 = np.zeros((1, self.ocultas))
        self.W2 = self.rng.normal(0, 0.65, (self.ocultas, 1))
        self.b2 = np.zeros((1, 1))
        self.epocas = 0
        self.historial: list[float] = []

    @staticmethod
    def preparar_entradas(
        distancia: float | np.ndarray,
        velocidad: float | np.ndarray,
    ) -> np.ndarray:
        """Normaliza distancia, velocidad y tiempo estimado hasta el obstáculo."""
        d = np.asarray(distancia, dtype=float)
        v = np.asarray(velocidad, dtype=float)
        tiempo = np.maximum(d, 0) / np.maximum(v, 0.001)
        return np.column_stack((
            np.clip(d / 550, 0, 1),
            np.clip(v / 9, 0, 1),
            np.clip(tiempo / 60, 0, 1),
        ))

    @staticmethod
    def sigmoide(z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -40, 40)))

    def adelante(
        self, X: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.W2 + self.b2
        return z1, a1, self.sigmoide(z2)

    def probabilidad(self, distancia: float, velocidad: float) -> float:
        X = self.preparar_entradas([distancia], [velocidad])
        return float(self.adelante(X)[2][0, 0])

    @staticmethod
    def etiqueta(
        distancia: float | np.ndarray,
        velocidad: float | np.ndarray,
    ) -> np.ndarray:
        """Etiqueta supervisada: saltar si faltan entre 7 y 13 fotogramas."""
        ttc = np.asarray(distancia) / np.maximum(np.asarray(velocidad), 0.001)
        return ((ttc >= 7) & (ttc <= 13)).astype(float).reshape(-1, 1)

    def generar_datos(self, n: int = 5000) -> tuple[np.ndarray, np.ndarray]:
        velocidad = self.rng.uniform(4.5, 8.5, n)
        positivos = n // 2
        ttc = np.concatenate((
            self.rng.uniform(7.5, 12.5, positivos),
            self.rng.uniform(0, 60, n - positivos),
        ))
        self.rng.shuffle(ttc)
        distancia = velocidad * ttc
        return self.preparar_entradas(distancia, velocidad), self.etiqueta(
            distancia, velocidad
        )

    def entrenar_epoca(self, X: np.ndarray, y: np.ndarray) -> float:
        z1, a1, a2 = self.adelante(X)
        p = np.clip(a2, 1e-8, 1 - 1e-8)
        perdida = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))

        dz2 = (a2 - y) / len(X)
        dW2 = a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)
        dz1 = (dz2 @ self.W2.T) * (z1 > 0)
        dW1 = X.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        self.W2 -= self.tasa * dW2
        self.b2 -= self.tasa * db2
        self.W1 -= self.tasa * dW1
        self.b1 -= self.tasa * db1
        self.epocas += 1
        self.historial.append(float(perdida))
        return float(perdida)

    def precision(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean((self.adelante(X)[2] >= 0.5) == y))
