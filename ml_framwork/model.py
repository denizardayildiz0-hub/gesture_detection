import numpy as np
from typing import List
from .module import Module,Param
from .metrics import accuracy, f1_macro

class Sequential:
    def __init__(self,layers: List[Module]):
        self.layers=layers
        self.history = {
            "loss": [],
            "val_loss": [],
            "acc": [],
            "val_acc": [],
            "f1": [],
            "val_f1": [],
        }

    def forward(self,x: np.ndarray) ->np.ndarray:
        for layer in self.layers:
            x=layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad


    def parameters(self)->List[Param]:

        params=[]
        for layer in self.layers:
            params.extend(layer.parameters())
        return params

    def get_state(self):
        return [p.value.copy() for p in self.parameters()]

    def set_state(self, state):
        params = self.parameters()
        assert len(params) == len(state)
        for p, v in zip(params, state):
            p.value[...] = v

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = self.forward(x)
        return np.argmax(logits, axis=1)

    def fit(
            self,
            x_train,
            y_train,
            loss_fn,
            optimizer,
            epochs: int = 20,
            batch_size: int = 32,
            verbose: bool = True,
            x_val=None,
            y_val=None,
    ):
        n_samples = x_train.shape[0]
        num_classes = int(np.max(y_train)) + 1  # assumes labels 0..C-1

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]
            epoch_loss = 0.0

            for start in range(0, n_samples, batch_size):
                end = start + batch_size
                x_batch = x_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                logits = self.forward(x_batch)
                loss = loss_fn.forward(logits, y_batch)
                epoch_loss += loss * x_batch.shape[0]

                grad = loss_fn.backward()
                self.backward(grad)

                params = self.parameters()
                optimizer.step(params)
                optimizer.zero_grad(params)

            epoch_loss /= n_samples

            # --- train metrics ---
            y_pred_train = self.predict(x_train)
            train_acc = accuracy(y_pred_train, y_train)
            train_f1 = f1_macro(y_pred_train, y_train, num_classes=num_classes)

            self.history["loss"].append(float(epoch_loss))
            self.history["acc"].append(float(train_acc))
            self.history["f1"].append(float(train_f1))

            # --- validation ---
            val_loss = None
            val_acc = None
            val_f1 = None

            if x_val is not None and y_val is not None:
                val_logits = self.forward(x_val)
                val_loss = float(loss_fn.forward(val_logits, y_val))

                y_pred_val = self.predict(x_val)
                val_acc = accuracy(y_pred_val, y_val)
                val_f1 = f1_macro(y_pred_val, y_val, num_classes=num_classes)

                self.history["val_loss"].append(val_loss)
                self.history["val_acc"].append(float(val_acc))
                self.history["val_f1"].append(float(val_f1))

            if verbose:
                if val_loss is None:
                    print(
                        f"epoch: {epoch + 1} - loss: {epoch_loss:.4f} - acc: {train_acc:.4f} - f1: {train_f1:.4f}"
                    )
                else:
                    print(
                        f"epoch: {epoch + 1} - loss: {epoch_loss:.4f} - acc: {train_acc:.4f} - f1: {train_f1:.4f} "
                        f"- val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f} - val_f1: {val_f1:.4f}"
                    )