"""
Fishr: Invariant Gradient Variances for Out-of-Distribution Generalization

Ramé, Dancette, Cord, ICML 2022.

Fishr-lite: match per-domain classifier-head gradient variances using a
closed-form per-sample gradient for the linear head (no BackPACK).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .base import Algorithm


class Fishr(Algorithm):
    """Fishr-lite: classifier-head gradient-variance matching."""

    def __init__(self, in_dim: int, n_classes: int, hparams: dict,
                 class_weights_per_domain=None, device: str = 'cuda'):
        super().__init__(in_dim, n_classes, hparams, class_weights_per_domain, device)

        self.build_default_backbone()
        self.build_optimizer()

        self.fishr_lambda = float(hparams.get('fishr_lambda', 1.0))
        self.fishr_anneal_iters = int(hparams.get('fishr_anneal_iters', 1500))
        self.fishr_weighted_grads = bool(hparams.get('fishr_weighted_grads', True))
        self._step = 0

    def _per_sample_class_weights(
        self, y: torch.Tensor, domain_labels: torch.Tensor
    ) -> torch.Tensor:
        weights = torch.ones_like(y, dtype=torch.float32, device=self.device)
        if not self.fishr_weighted_grads:
            return weights

        for d in torch.unique(domain_labels):
            mask = domain_labels == d
            w = self.class_weights_per_domain.get(int(d.item()))
            if w is None or mask.sum() == 0:
                continue
            sample_w = w.to(self.device)[y[mask]].float()
            # PyTorch's weighted CE mean normalizes by the batch weight sum.
            # Keeping a per-domain mean of one makes the Fishr penalty reflect
            # class imbalance without letting different domain class priors
            # dominate the variance scale.
            sample_w = sample_w / sample_w.mean().clamp_min(1e-8)
            weights[mask] = sample_w
        return weights

    def _per_sample_classifier_grad(
        self, logits: torch.Tensor, y: torch.Tensor, h: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Closed-form per-sample gradients of the linear classifier head.

        Args:
            logits: (N, C) classifier outputs.
            y: (N,) integer labels.
            h: (N, d) backbone features.

        Returns:
            grads: (N, C*d) flattened per-sample gradients.
        """
        N, C = logits.shape
        d = h.shape[1]
        p = F.softmax(logits, dim=1)                    # (N, C)
        y_onehot = F.one_hot(y, num_classes=C).float()  # (N, C)
        err = p - y_onehot                              # (N, C)
        if sample_weights is not None:
            err = err * sample_weights.view(-1, 1)
        grads = torch.einsum('nc,nd->ncd', err, h).reshape(N, C * d)
        return grads

    def update(self, batch) -> dict:
        x, y, domain_labels = batch
        x = x.to(self.device)
        y = y.to(self.device)
        domain_labels = domain_labels.to(self.device)

        self.train()
        h = self.backbone(x)
        logits = self.classifier(h)

        loss_cls = self.compute_per_domain_cls_loss(logits, y, domain_labels)

        # ---- Fishr penalty ----
        sample_weights = self._per_sample_class_weights(y, domain_labels)
        grads = self._per_sample_classifier_grad(logits, y, h, sample_weights)  # (N, C*d)

        unique_domains = torch.unique(domain_labels)
        per_domain_vars = []
        for d in unique_domains:
            mask = (domain_labels == d)
            if mask.sum() < 2:
                continue
            g_d = grads[mask]                           # (n_d, C*d)
            var_d = g_d.var(dim=0, unbiased=False)      # (C*d,)
            per_domain_vars.append(var_d)

        if len(per_domain_vars) >= 2:
            stacked = torch.stack(per_domain_vars, dim=0)   # (E, C*d)
            mean_var = stacked.mean(dim=0, keepdim=True)    # (1, C*d)
            # L_Fishr = (1/E) * Σ_e || V_e - V̄ ||²
            fishr_penalty = ((stacked - mean_var) ** 2).sum(dim=1).mean()
        else:
            fishr_penalty = torch.tensor(0.0, device=self.device)

        if self._step < self.fishr_anneal_iters:
            effective_lambda = 0.0
        else:
            effective_lambda = self.fishr_lambda

        loss_total = loss_cls + effective_lambda * fishr_penalty

        if effective_lambda > 1.0:
            loss_total = loss_total / effective_lambda

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        self._step += 1

        return {
            'loss_cls': loss_cls.item(),
            'loss_fishr': fishr_penalty.item(),
            'loss_total': loss_total.item(),
            'effective_lambda': effective_lambda,
            'fishr_weighted_grads': float(self.fishr_weighted_grads),
        }
