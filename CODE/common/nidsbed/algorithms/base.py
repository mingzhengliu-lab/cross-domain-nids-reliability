"""Abstract base class for domain-generalization algorithms.

Subclasses implement ``update(batch)`` for one training step and inherit
``predict(x)`` for batched logits inference.
"""
from abc import ABC, abstractmethod
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ..networks import MLPBackbone, Classifier
from ..lib.misc import get_class_weights


class Algorithm(nn.Module, ABC):
    """Shared DG algorithm interface.

    Concrete methods must implement ``update()``; inference goes through
    ``predict()``.
    """

    def __init__(self, in_dim: int, n_classes: int, hparams: dict,
                 class_weights_per_domain: Optional[dict] = None,
                 device: str = 'cuda'):
        """Initialize backbone/classifier placeholders and per-domain weights.

        Args:
            in_dim: input feature dimension.
            n_classes: number of classes (2 for binary NIDS).
            hparams: training hyperparameters such as lr, batch_size,
                weight_decay, and dropout, plus method-specific keys.
            class_weights_per_domain: mapping from domain label to a
                length-2 tensor ``[w_benign, w_attack]`` used by the
                per-domain weighted cross-entropy loss.
            device: ``'cuda'`` or ``'cpu'``.
        """
        super().__init__()
        self.in_dim = in_dim
        self.n_classes = n_classes
        self.hparams = hparams
        self.device = device
        self.class_weights_per_domain = class_weights_per_domain or {}

        self.backbone: Optional[nn.Module] = None
        self.classifier: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None

    @abstractmethod
    def update(self, batch) -> Dict[str, float]:
        """Run one optimizer step and return scalar loss terms.

        Args:
            batch: tuple ``(X, y, domain_labels)`` where
                X is ``(B, in_dim)``,
                y is ``(B,)`` int64 labels, and
                domain_labels is ``(B,)`` int64 source-domain ids.

        Returns:
            Dict of floats, always including ``loss_cls`` / ``loss_total``
            and optionally method-specific keys such as ``loss_dann``,
            ``loss_irm``, or ``loss_coral``.
        """
        raise NotImplementedError

    def build_default_backbone(self, hidden: Optional[int] = None,
                               feat_dim: Optional[int] = None,
                               dropout: Optional[float] = None):
        """Instantiate the shared MLP backbone and linear classifier."""
        h = hidden if hidden is not None else self.hparams.get('hidden', 256)
        fd = feat_dim if feat_dim is not None else self.hparams.get('feat_dim', 64)
        dp = dropout if dropout is not None else self.hparams.get('dropout', 0.3)

        self.backbone = MLPBackbone(
            in_dim=self.in_dim, hidden=h, feat_dim=fd, dropout=dp
        ).to(self.device)
        self.classifier = Classifier(feat_dim=fd, n_classes=self.n_classes).to(self.device)

    def build_optimizer(self, params=None):
        """Create an Adam optimizer over backbone + classifier parameters.

        DANN subclasses may pass a broader ``params`` iterable that also
        includes the domain discriminator.
        """
        if params is None:
            params = list(self.backbone.parameters()) + list(self.classifier.parameters())

        lr = self.hparams.get('lr', 1e-4)
        wd = self.hparams.get('weight_decay', 0)
        self.optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return class logits for a feature batch."""
        self.eval()
        x = x.to(self.device)
        h = self.backbone(x)
        logits = self.classifier(h)
        return logits

    @torch.no_grad()
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return softmax probabilities for a feature batch."""
        logits = self.predict(x)
        return F.softmax(logits, dim=1)

    def compute_per_domain_cls_loss(self, logits: torch.Tensor, y: torch.Tensor,
                                    domain_labels: torch.Tensor) -> torch.Tensor:
        """Average per-domain weighted cross-entropy across domains present.

        Each domain uses its own class-weight vector from
        ``class_weights_per_domain`` when available.

        Args:
            logits: ``(B, n_classes)`` raw scores.
            y: ``(B,)`` int64 labels.
            domain_labels: ``(B,)`` int64 domain ids.

        Returns:
            Scalar mean loss across domains with at least one sample.
        """
        unique_domains = torch.unique(domain_labels)
        losses = []

        for d in unique_domains:
            mask = (domain_labels == d)
            if mask.sum() == 0:
                continue
            logits_d = logits[mask]
            y_d = y[mask]

            w = self.class_weights_per_domain.get(int(d.item()))
            if w is not None:
                w = w.to(self.device)

            loss_d = F.cross_entropy(logits_d, y_d, weight=w)
            losses.append(loss_d)

        if len(losses) == 0:
            return torch.tensor(0.0, device=self.device, requires_grad=True)

        return torch.stack(losses).mean()
