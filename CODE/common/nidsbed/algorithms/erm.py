"""Empirical Risk Minimization baseline for the LODO NIDS benchmark."""

import torch
from .base import Algorithm


class ERM(Algorithm):
    """Class-weighted ERM averaged across source domains."""

    def __init__(self, in_dim: int, n_classes: int, hparams: dict,
                 class_weights_per_domain=None, device: str = 'cuda'):
        super().__init__(in_dim, n_classes, hparams, class_weights_per_domain, device)

        self.build_default_backbone()
        self.build_optimizer()

    def update(self, batch) -> dict:
        x, y, domain_labels = batch
        x = x.to(self.device)
        y = y.to(self.device)
        domain_labels = domain_labels.to(self.device)

        self.train()
        h = self.backbone(x)
        logits = self.classifier(h)

        loss_cls = self.compute_per_domain_cls_loss(logits, y, domain_labels)

        self.optimizer.zero_grad()
        loss_cls.backward()
        self.optimizer.step()

        return {
            'loss_cls': loss_cls.item(),
            'loss_total': loss_cls.item(),
        }
