import numpy as np


class MSELoss:
    def forward(self, y_pred, y_true):
        self.diff = (y_pred - y_true).astype(np.float32)
        return float(np.mean(self.diff ** 2))

    def backward(self):
        n = self.diff.shape[0]
        m = self.diff.shape[1] if self.diff.ndim > 1 else 1
        return (2.0 / (n * m)) * self.diff

class MAELoss:
    def forward(self, y_pred, y_true):
        self.diff = (y_pred - y_true).astype(np.float32)
        return float(np.mean(np.abs(self.diff)))

    def backward(self):
        n = self.diff.shape[0]
        m = self.diff.shape[1] if self.diff.ndim > 1 else 1
        return (1.0 / (n * m)) * np.sign(self.diff)




class SoftmaxCrossEntropyLoss:
    def forward(self, logits, y_true):
        self.logits = logits
        self.n = logits.shape[0]

        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        self.probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)

        self.y_true = y_true

        loss = -np.log(self.probs[np.arange(self.n), y_true] + 1e-12)
        return float(np.mean(loss))

    def backward(self):
        grad = self.probs.copy()
        grad[np.arange(self.n), self.y_true] -= 1
        return grad / self.n



