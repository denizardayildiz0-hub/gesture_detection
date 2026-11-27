import numpy as np


class Param:
    def __init__(self, value: np.ndarray):
        self.value = value.astype(np.float32)
        self.grad = np.zeros_like(self.value, dtype=np.float32)


class Module:
    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def parameters(self):
        return []

    def zero_grad(self):
        for p in self.parameters():
            p.grad.fill(0.0)

