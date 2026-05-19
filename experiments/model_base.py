"""
Model Base Interface for unified neural and symbolic modeling
"""
import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Union, Any
import numpy.typing as npt


class ModelBase(ABC):
    """Abstract base class for model interfaces."""
    
    @abstractmethod
    def fit(self, X_train: Any, y_train: Any):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: Any) -> np.ndarray:
        """Make predictions."""
        pass


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Mean Squared Error."""
    y_true, y_pred = _validate_arrays(y_true, y_pred)
    return float(np.mean((y_true - y_pred) ** 2))


def compute_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R-squared score."""
    y_true, y_pred = _validate_arrays(y_true, y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1 - (ss_res / ss_tot))


def compute_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute correlation coefficient."""
    y_true, y_pred = _validate_arrays(y_true, y_pred)
    if len(y_true) < 2:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def compute_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute cross entropy with numerical stability."""
    y_true, y_pred = _validate_arrays(y_true, y_pred)
    # Clip predictions to prevent log(0)
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    # Normalize true values to [0,1]
    y_true = np.clip(y_true, 0, 1)
    return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))


def _validate_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Validate and convert inputs to numpy arrays."""
    if not isinstance(y_true, np.ndarray):
        y_true = np.asarray(y_true)
    if not isinstance(y_pred, np.ndarray):
        y_pred = np.asarray(y_pred)
        
    # Ensure same shape
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    
    # Handle edge cases
    if y_true.size == 0:
        return y_true, y_pred
    
    return y_true, y_pred