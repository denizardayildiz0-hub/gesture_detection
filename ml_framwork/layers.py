import numpy as np
from .module import Module,Param


class Dense(Module):
    def __init__(self,in_features:int,out_features:int,seed: int=42):
        rng=np.random.default_rng(seed)
        W=rng.normal(
            loc=0.0,
            scale=np.sqrt(2.0/ in_features),
            size=(in_features,out_features)
        ).astype(np.float32)


        b=np.zeros((1,out_features),dtype=np.float32)

        self.W=Param(W)
        self.b=Param(b)

        self._input=None


    def forward(self,x:np.ndarray) ->np.ndarray:
        self._input=x
        return x @ self.W.value+self.b.value

    def backward(self,grad_output:np.ndarray) ->np.ndarray:
        self.W.grad +=self._input.T @ grad_output
        self.b.grad +=np.sum(grad_output,axis=0,keepdims=True)
        grad_input=grad_output @ self.W.value.T
        return grad_input


    def parameters(self):
        return (self.W,self.b)

class Relu(Module):
    def __init__(self):
        self._mask=None
    def forward(self,x:np.ndarray) ->np.ndarray:
        self._mask=(x>0).astype(np.float32)
        return x *self._mask
    def backward(self,grad_output:np.ndarray) ->np.ndarray:
        return grad_output * self._mask

class Tanh(Module):
    def __init__(self):
        self._out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._out = np.tanh(x).astype(np.float32)
        return self._out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        # d/dx tanh = 1 - tanh^2
        return grad_output * (1.0 - self._out**2)


class Sigmoid(Module):
    def __init__(self):
        self._out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        # stable sigmoid
        x = x.astype(np.float32)
        self._out = 1.0 / (1.0 + np.exp(-x))
        return self._out.astype(np.float32)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * (self._out * (1.0 - self._out))



