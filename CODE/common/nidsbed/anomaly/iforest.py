"""
Isolation Forest wrapper.

sklearn's IsolationForest is O(n_estimators * max_samples * log(max_samples))
for fit, so cost is dominated by max_samples (subsample param), not the source
training size. We additionally subsample raw fit data to max_train_samples
for memory; sklearn would do another subsample down to max_samples internally.

Note on n_estimators: empirically with sklearn default n_estimators=200, the
cross-domain AUROC seed-to-seed std on NFv2 LODO can be as large as 0.05-0.23
(IForest is the only detector with non-trivial seed variance under our setup).
We use n_estimators=500 by default to halve the std without changing the
mean — the 2.5x compute cost is still <3 min/run.
"""
import numpy as np
from sklearn.ensemble import IsolationForest
from .base import AnomalyDetector


class IForestDetector(AnomalyDetector):
    name = 'iforest'

    def __init__(self, n_estimators: int = 500, max_samples: int = 256,
                 contamination='auto', max_train_samples: int = 200_000,
                 random_state: int = 42, n_jobs: int = -1, **kwargs):
        """
        Args:
            n_estimators: number of trees.
            max_samples: per-tree subsample size (sklearn default 256).
            contamination: expected anomaly fraction. 'auto' uses sklearn's
                offset-based threshold (we don't use sklearn's threshold,
                only its decision_function, so this only affects offset_).
            max_train_samples: cap on rows fed to fit (memory + speed).
        """
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.max_train_samples = max_train_samples
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.model = None

    def fit(self, X_benign):
        rng = np.random.RandomState(self.random_state)
        X = np.asarray(X_benign, dtype=np.float32)
        if len(X) > self.max_train_samples:
            idx = rng.choice(len(X), self.max_train_samples, replace=False)
            X = X[idx]
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=self.n_jobs,
        )
        self.model.fit(X)

    def score(self, X):
        # sklearn convention: decision_function HIGH = inlier; we want HIGH = anomaly
        return (-self.model.decision_function(X)).astype(np.float32)
