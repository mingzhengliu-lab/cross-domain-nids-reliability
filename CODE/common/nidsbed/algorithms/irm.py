"""Invariant Risk Minimization (IRMv1) for source-only LODO training.

Reference: Arjovsky et al., arXiv:1907.02893, 2019. Adds a scalar-scale IRM
penalty on per-domain weighted CE using a dummy scale parameter w=1.0.
"""
import torch
import torch.nn.functional as F
from torch import autograd

from .base import Algorithm


class IRM(Algorithm):
    """IRMv1 with optional penalty annealing (Arjovsky 2019 Algorithm 1)."""

    def __init__(self, in_dim: int, n_classes: int, hparams: dict,
                 class_weights_per_domain=None, device: str = 'cuda'):
        super().__init__(in_dim, n_classes, hparams, class_weights_per_domain, device)

        self.build_default_backbone()
        self.build_optimizer()

        self.irm_lambda = float(hparams.get('irm_lambda', 1.0))
        self.irm_anneal_iters = int(hparams.get('irm_anneal_iters', 500))

        self._step = 0

    @staticmethod
    def _irm_penalty(logits: torch.Tensor, y: torch.Tensor,
                     class_weight: torch.Tensor = None) -> torch.Tensor:
        """IRM penalty on a single domain (Arjovsky 2019 Eq. 3)."""
        scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
        loss = F.cross_entropy(logits * scale, y, weight=class_weight)
        grad = autograd.grad(loss, [scale], create_graph=True)[0]
        return grad.pow(2)

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
        penalties = []
        for d in unique_domains:
            mask = (domain_labels == d)
            if mask.sum() < 2:
                continue  # ,
            w = self.class_weights_per_domain.get(int(d.item()))
            if w is not None:
                w = w.to(self.device)
            penalty_d = self._irm_penalty(logits[mask], y[mask], class_weight=w)
            penalties.append(penalty_d)

        if penalties:
            irm_penalty = torch.stack(penalties).mean()
        else:
            irm_penalty = torch.tensor(0.0, device=self.device)

        if self._step < self.irm_anneal_iters:
            effective_lambda = 1.0
        else:
            effective_lambda = self.irm_lambda

        loss_total = loss_cls + effective_lambda * irm_penalty

        if effective_lambda > 1.0:
            loss_total = loss_total / effective_lambda

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        self._step += 1

        return {
            'loss_cls': loss_cls.item(),
            'loss_irm_penalty': irm_penalty.item(),
            'loss_total': loss_total.item(),
            'effective_lambda': effective_lambda,
        }
