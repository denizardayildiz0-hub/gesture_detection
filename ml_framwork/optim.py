import numpy as np
from typing import List, Dict
from .module import Param



class SGD:
    def __init__(self,lr:float=0.01,weight_decay:float=0.0):
        self.lr=lr
        self.weight_decay=weight_decay


    def step(self,parameters:List[Param]):
        for p in parameters:
            if self.weight_decay !=0.0:
                p.grad += self.weight_decay *p.value

            p.value -= self.lr *p.grad


    def zero_grad(self,parameters:List[Param]):

        for p in parameters:
            p.grad.fill(0)

class VanillaGD:

    def __init__(self, lr: float = 0.01, weight_decay: float = 0.0):
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self, parameters: List[Param]):
        for p in parameters:
            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.value

            p.value -= self.lr * grad

    def zero_grad(self, parameters: List[Param]):
        for p in parameters:
            p.grad.fill(0.0)

class SGDMomentum:
    def __init__(self, lr: float = 0.01, momentum: float = 0.9, weight_decay: float = 0.0):
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.v: Dict[int, np.ndarray] = {}

    def step(self, parameters: List[Param]):
        for p in parameters:
            pid = id(p)
            if pid not in self.v:
                self.v[pid] = np.zeros_like(p.value, dtype=np.float32)

            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.value

            # v = mu*v - lr*grad ; w += v
            self.v[pid] = self.momentum * self.v[pid] - self.lr * grad
            p.value += self.v[pid]

    def zero_grad(self, parameters: List[Param]):
        for p in parameters:
            p.grad.fill(0.0)


class RMSProp:
    def __init__(self, lr: float = 0.001, beta: float = 0.9, eps: float = 1e-8, weight_decay: float = 0.0):
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.weight_decay = weight_decay
        self.v = {}  # running average of squared gradients

    def step(self, parameters: List[Param]):
        for p in parameters:
            pid = id(p)

            if pid not in self.v:
                self.v[pid] = np.zeros_like(p.value, dtype=np.float32)

            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.value

            # v = beta*v + (1-beta)*g^2
            self.v[pid] = self.beta * self.v[pid] + (1.0 - self.beta) * (grad * grad)

            # update
            p.value -= self.lr * grad / (np.sqrt(self.v[pid]) + self.eps)

    def zero_grad(self, parameters: List[Param]):
        for p in parameters:
            p.grad.fill(0.0)

class Adam:
    def __init__(self, lr: float = 1e-3, betas=(0.9, 0.999), eps: float = 1e-8, weight_decay: float = 0.0):
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m: Dict[int, np.ndarray] = {}
        self.v: Dict[int, np.ndarray] = {}

    def step(self, parameters: List[Param]):
        self.t += 1
        for p in parameters:
            pid = id(p)
            if pid not in self.m:
                self.m[pid] = np.zeros_like(p.value, dtype=np.float32)
                self.v[pid] = np.zeros_like(p.value, dtype=np.float32)

            grad = p.grad
            if self.weight_decay != 0.0:
                grad = grad + self.weight_decay * p.value

            self.m[pid] = self.beta1 * self.m[pid] + (1 - self.beta1) * grad
            self.v[pid] = self.beta2 * self.v[pid] + (1 - self.beta2) * (grad * grad)

            m_hat = self.m[pid] / (1 - self.beta1 ** self.t)
            v_hat = self.v[pid] / (1 - self.beta2 ** self.t)

            p.value -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def zero_grad(self, parameters: List[Param]):
        for p in parameters:
            p.grad.fill(0.0)

