"""Shared neural network modules for the LODO benchmark."""

import torch.nn as nn
from torch.autograd import Function


class _GradReverseFn(Function):
    """Autograd function for gradient reversal used by DANN."""

    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


def grad_reverse(x, lambd: float = 1.0):
    """Return x unchanged in forward pass and reverse its gradient."""
    return _GradReverseFn.apply(x, lambd)


class MLPBackbone(nn.Module):
    """Three-layer MLP feature extractor shared by the benchmark methods."""

    def __init__(
        self,
        in_dim: int = 39,
        hidden: int = 256,
        feat_dim: int = 64,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden = hidden
        self.feat_dim = feat_dim

        self.features = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, feat_dim),
            nn.BatchNorm1d(feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.features(x)


class Classifier(nn.Module):
    """Linear classifier head."""

    def __init__(self, feat_dim: int = 64, n_classes: int = 2):
        super().__init__()
        self.fc = nn.Linear(feat_dim, n_classes)

    def forward(self, h):
        return self.fc(h)


class DomainDiscriminator(nn.Module):
    """Domain classifier used by DANN."""

    def __init__(
        self,
        feat_dim: int = 64,
        hidden: int = 64,
        n_domains: int = 3,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, n_domains),
        )

    def forward(self, h, lambd: float = 1.0):
        h_rev = grad_reverse(h, lambd)
        return self.net(h_rev)
