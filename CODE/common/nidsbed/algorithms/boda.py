"""
BoDA: Balanced Domain-Class Distribution Alignment

Yang, Wang, Katabi, "On Multi-Domain Long-Tailed Recognition, Imbalanced
Domain Generalization and Beyond", ECCV 2022.

BoDA-style: same-class cross-domain EMA prototype alignment across source
domains (inspired by Yang et al., ECCV 2022). Uses warmup and a per-batch
minimum sample count per (domain, class) pair.
"""
import itertools

import torch
import torch.nn.functional as F

from .base import Algorithm


class BoDA(Algorithm):
    """BoDA-style prototype alignment with EMA and warmup."""

    def __init__(self, in_dim: int, n_classes: int, hparams: dict,
                 class_weights_per_domain=None, device: str = 'cuda',
                 n_source_domains: int = 3):
        super().__init__(in_dim, n_classes, hparams, class_weights_per_domain, device)

        self.build_default_backbone()
        self.build_optimizer()

        self.boda_lambda = float(hparams.get('boda_lambda', 1.0))
        self.boda_ema_momentum = float(hparams.get('boda_ema_momentum', 0.9))
        self.boda_warmup_iters = int(hparams.get('boda_warmup_iters', 500))
        self.boda_min_samples = int(hparams.get('boda_min_samples', 8))

        self.n_source_domains = n_source_domains
        self.n_classes = n_classes

        feat_dim = hparams.get('feat_dim', 64)

        self.register_prototype_buffers(feat_dim)

        self._step = 0

    def register_prototype_buffers(self, feat_dim: int):
        """Register EMA prototype buffers (uninitialized until enough samples)."""
        self.prototypes = torch.zeros(
            self.n_source_domains, self.n_classes, feat_dim,
            device=self.device,
        )
        self.proto_initialized = torch.zeros(
            self.n_source_domains, self.n_classes,
            dtype=torch.bool, device=self.device,
        )

    def _update_prototypes(self, h: torch.Tensor, y: torch.Tensor,
                           domain_labels: torch.Tensor) -> dict:
        """Update EMA prototypes from the current batch per (domain, class)."""
        counts = {}
        with torch.no_grad():
            for d in range(self.n_source_domains):
                for c in range(self.n_classes):
                    mask = (domain_labels == d) & (y == c)
                    n_dc = int(mask.sum().item())
                    counts[(d, c)] = n_dc
                    if n_dc < self.boda_min_samples:
                        continue
                    current = h[mask].mean(dim=0)
                    if not self.proto_initialized[d, c]:
                        self.prototypes[d, c] = current
                        self.proto_initialized[d, c] = True
                    else:
                        m = self.boda_ema_momentum
                        self.prototypes[d, c] = m * self.prototypes[d, c] + (1 - m) * current
        return counts

    def _boda_penalty(self) -> torch.Tensor:
        """EMA prototype alignment loss across same-class source-domain pairs.

        For each class c, average L2 distance between initialized prototypes
        μ_{d,c} and μ_{d',c} over all source-domain pairs (d, d').
        """
        feat_dim = self.prototypes.shape[-1]
        losses = []
        for c in range(self.n_classes):
            init_domains = [d for d in range(self.n_source_domains)
                            if bool(self.proto_initialized[d, c].item())]
            if len(init_domains) < 2:
                continue
            for d1, d2 in itertools.combinations(init_domains, 2):
                diff = self.prototypes[d1, c] - self.prototypes[d2, c]
                losses.append((diff ** 2).sum() / feat_dim)
        if not losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(losses).mean()

    def _contrastive_current_batch(
        self, h: torch.Tensor, y: torch.Tensor, domain_labels: torch.Tensor
    ) -> torch.Tensor:
        """Contrastive pull of batch features toward other domains' EMA prototypes.

        For each sample, pull h_i toward the mean same-class prototype from
        domains other than the sample's own domain.
        """
        n_dom = self.n_source_domains
        per_sample_losses = []
        for d in range(n_dom):
            for c in range(self.n_classes):
                mask = (domain_labels == d) & (y == c)
                if mask.sum() < 1:
                    continue
                other_domains = [dd for dd in range(n_dom)
                                 if dd != d and bool(self.proto_initialized[dd, c].item())]
                if not other_domains:
                    continue
                target_proto = self.prototypes[other_domains, c].mean(dim=0).detach()
                diff = h[mask] - target_proto.unsqueeze(0)
                per_sample_losses.append((diff ** 2).sum(dim=1))
        if not per_sample_losses:
            return torch.tensor(0.0, device=self.device)
        return torch.cat(per_sample_losses).mean()

    def update(self, batch) -> dict:
        x, y, domain_labels = batch
        x = x.to(self.device)
        y = y.to(self.device)
        domain_labels = domain_labels.to(self.device)

        self.train()
        h = self.backbone(x)
        logits = self.classifier(h)

        loss_cls = self.compute_per_domain_cls_loss(logits, y, domain_labels)

        counts = self._update_prototypes(h.detach(), y, domain_labels)

        if self._step < self.boda_warmup_iters:
            effective_lambda = 0.0
            boda_penalty = torch.tensor(0.0, device=self.device)
        else:
            effective_lambda = self.boda_lambda
            boda_penalty = self._contrastive_current_batch(h, y, domain_labels)

        loss_total = loss_cls + effective_lambda * boda_penalty

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        self._step += 1

        n_initialized = int(self.proto_initialized.sum().item())

        return {
            'loss_cls': loss_cls.item(),
            'loss_boda': boda_penalty.item(),
            'loss_total': loss_total.item(),
            'effective_lambda': effective_lambda,
            'n_initialized_dc': n_initialized,
        }

