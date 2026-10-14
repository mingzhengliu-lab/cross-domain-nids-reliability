"""Abstract base class for anomaly detectors. Higher score = more anomalous."""
from abc import ABC, abstractmethod
import numpy as np


class AnomalyDetector(ABC):
    name: str = 'base'

    @abstractmethod
    def fit(self, X_benign: np.ndarray) -> None:
        """Fit on source benign training samples only.

        Args:
            X_benign: (N, D) float32, LODO-scaled, label-free benign rows from
                3 source domains' training splits.
        """

    @abstractmethod
    def score(self, X: np.ndarray) -> np.ndarray:
        """Return per-row anomaly scores (higher = more anomalous).

        Args:
            X: (N, D) float32, same scaling as fit data.

        Returns:
            (N,) float32.
        """
