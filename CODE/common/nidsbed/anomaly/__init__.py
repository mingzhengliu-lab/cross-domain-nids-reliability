"""Source-benign-only anomaly detectors.

All detectors satisfy:
  - fit(X_benign) sees ONLY source-domain benign training samples.
  - No source-attack labels at fit time.
  - No target-domain X at fit time.
  - Higher score = more anomalous.

OneClassSVM intentionally omitted: O(n^2) RBF training infeasible at NFv2 scale.
"""
from .mahalanobis import MahalanobisDetector
from .iforest import IForestDetector
from .autoencoder import AEDetector
from .deep_svdd import DeepSVDDDetector

DETECTOR_REGISTRY = {
    'mahalanobis': MahalanobisDetector,
    'iforest':     IForestDetector,
    'autoencoder': AEDetector,
    'deep_svdd':   DeepSVDDDetector,
}

__all__ = [
    'MahalanobisDetector', 'IForestDetector', 'AEDetector', 'DeepSVDDDetector',
    'DETECTOR_REGISTRY',
]
