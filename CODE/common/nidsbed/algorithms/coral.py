"""
DeepCORAL (Deep Correlation Alignment)
    Sun & Saenko
    ECCV 2016 Workshops, "Deep CORAL: Correlation Alignment for Deep Domain Adaptation"

Source-only multi-source variant: align feature covariances across all
pairs of source domains with a pairwise CORAL penalty (Sun & Saenko 2016).
"""
import itertools
import torch
import torch.nn.functional as F

from .base import Algorithm


class DeepCORAL(Algorithm):
    """DeepCORAL trained on source domains only."""

    def __init__(self, in_dim: int, n_classes: int, hparams: dict,
                 class_weights_per_domain=None, device: str = 'cuda'):
        super().__init__(in_dim, n_classes, hparams, class_weights_per_domain, device)

        self.build_default_backbone()
        self.build_optimizer()

        self.coral_weight = float(
            hparams.get('coral_lambda',
                        hparams.get('coral_weight', 1.0))
        )

    @staticmethod
    def _cov(h: torch.Tensor) -> torch.Tensor:
        """Unbiased feature covariance for batch h (n, d)."""
        n = h.size(0)
        if n < 2:
            d = h.size(1)
            return torch.zeros(d, d, device=h.device)
        mean = h.mean(dim=0, keepdim=True)
        hc = h - mean
        cov = hc.t() @ hc / (n - 1)
        return cov

    def _coral_loss(self, h_a: torch.Tensor, h_b: torch.Tensor) -> torch.Tensor:
        """
        DeepCORAL loss between two source feature tensors.
        Sun & Saenko Eq. 1: ℓ = ||C_a - C_b||_F² / (4·d²)
        """
        c_a = self._cov(h_a)
        c_b = self._cov(h_b)
        d = h_a.size(1)
        loss = (c_a - c_b).pow(2).sum() / (4 * d * d)
        return loss

    def update(self, batch) -> dict:
        x, y, domain_labels = batch
        x = x.to(self.device)
        y = y.to(self.device)
        domain_labels = domain_labels.to(self.device)

        self.train()
        h = self.backbone(x)
        logits = self.classifier(h)

        loss_cls = self.compute_per_domain_cls_loss(logits, y, domain_labels)

        unique_domains = torch.unique(domain_labels)
        if len(unique_domains) < 2:
            coral_loss = torch.tensor(0.0, device=self.device)
        else:
            domain_features = {}
            for d in unique_domains:
                mask = (domain_labels == d)
                domain_features[int(d.item())] = h[mask]

            pair_losses = []
            for d1, d2 in itertools.combinations(domain_features.keys(), 2):
                pair_losses.append(self._coral_loss(domain_features[d1],
                                                     domain_features[d2]))
            coral_loss = torch.stack(pair_losses).mean() if pair_losses \
                else torch.tensor(0.0, device=self.device)

        loss_total = loss_cls + self.coral_weight * coral_loss

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        return {
            'loss_cls': loss_cls.item(),
            'loss_coral': coral_loss.item(),
            'loss_total': loss_total.item(),
        }
