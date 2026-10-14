"""
Mahalanobis distance detector — closed-form Gaussian baseline.

  score(x) = (x - mu)^T Sigma^{-1} (x - mu)

Fit-time cost: O(N D^2) for covariance + O(D^3) for inversion. Trivial for D=39.
No tunable HPs other than ridge.
"""
import numpy as np
from .base import AnomalyDetector


class MahalanobisDetector(AnomalyDetector):
    name = 'mahalanobis'

    def __init__(self, ridge: float = 1e-4, **kwargs):
        """
        Args:
            ridge: Tikhonov regularization added to covariance for numerical
                stability (covariance can be near-singular if features are
                strongly collinear after StandardScaler).
        """
        self.ridge = ridge
        self.mu = None
        self.inv_cov = None

    def fit(self, X_benign):
        X = np.asarray(X_benign, dtype=np.float64)
        self.mu = X.mean(axis=0)
        cov = np.cov(X, rowvar=False)
        cov = cov + self.ridge * np.eye(cov.shape[0])
        # pinv for safety against rank deficiency
        self.inv_cov = np.linalg.pinv(cov)

    def score(self, X):
        X = np.asarray(X, dtype=np.float64)
        d = X - self.mu
        # batched (x - mu)^T Sigma^{-1} (x - mu)
        s = np.einsum('ni,ij,nj->n', d, self.inv_cov, d)
        return s.astype(np.float32)
