import pickle
from typing import Any


def save_model(model: Any, filepath: str):
    """
    Save the entire model object using pickle.
    """
    with open(filepath, "wb") as f:
        pickle.dump(model, f)


def load_model(filepath: str) -> Any:
    """
    Load a previously saved model.
    """
    with open(filepath, "rb") as f:
        model = pickle.load(f)
    return model