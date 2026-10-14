"""Multitask ERM with an auxiliary attack-family head.

The primary task stays binary attack-vs-benign classification. A second head
predicts source-domain attack families, including benign as its own family
class, to discourage the backbone from collapsing all attacks into a single
source-specific shortcut direction.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ..networks import Classifier
from .base import Algorithm


class MultitaskERM(Algorithm):
    """ERM with a shared backbone and an auxiliary attack-family objective."""

    def __init__(self, in_dim: int, n_classes: int, hparams: dict,
                 class_weights_per_domain=None, device: str = "cuda"):
        super().__init__(in_dim, n_classes, hparams, class_weights_per_domain, device)

        self.build_default_backbone()

        feat_dim = int(hparams.get("feat_dim", 64))
        self.n_attack_families = int(hparams.get("n_attack_families", 0))
        if self.n_attack_families <= 1:
            raise ValueError(
                "MultitaskERM requires hparams['n_attack_families'] > 1 "
                f"(got {self.n_attack_families})."
            )

        self.family_lambda = float(hparams.get("family_lambda", 0.3))
        self.family_classifier = Classifier(
            feat_dim=feat_dim,
            n_classes=self.n_attack_families,
        ).to(self.device)

        family_weights = hparams.get("attack_family_class_weights")
        self.attack_family_class_weights = None
        if family_weights is not None:
            family_weights = torch.as_tensor(
                family_weights,
                dtype=torch.float32,
                device=self.device,
            )
            if family_weights.numel() != self.n_attack_families:
                raise ValueError(
                    "attack_family_class_weights size mismatch: "
                    f"expected {self.n_attack_families}, got {family_weights.numel()}"
                )
            self.attack_family_class_weights = family_weights

        params = (
            list(self.backbone.parameters())
            + list(self.classifier.parameters())
            + list(self.family_classifier.parameters())
        )
        self.build_optimizer(params=params)

    def _compute_family_loss(self, logits_family: torch.Tensor,
                             attack_family: torch.Tensor,
                             domain_labels: torch.Tensor) -> torch.Tensor:
        valid = (attack_family >= 0) & (attack_family < self.n_attack_families)
        unique_domains = torch.unique(domain_labels)
        losses = []
        for d in unique_domains:
            mask = (domain_labels == d) & valid
            if mask.sum() == 0:
                continue
            loss_d = F.cross_entropy(
                logits_family[mask],
                attack_family[mask],
                weight=self.attack_family_class_weights,
            )
            losses.append(loss_d)
        if not losses:
            return torch.tensor(0.0, device=self.device, requires_grad=True)
        return torch.stack(losses).mean()

    def update(self, batch) -> dict:
        if len(batch) != 4:
            raise ValueError(
                "MultitaskERM expects batches of "
                "(x, y, domain_labels, attack_family_labels)."
            )

        x, y, domain_labels, attack_family = batch
        x = x.to(self.device)
        y = y.to(self.device)
        domain_labels = domain_labels.to(self.device)
        attack_family = attack_family.to(self.device)

        self.train()
        h = self.backbone(x)
        logits_cls = self.classifier(h)
        logits_family = self.family_classifier(h)

        loss_cls = self.compute_per_domain_cls_loss(logits_cls, y, domain_labels)
        loss_family = self._compute_family_loss(logits_family, attack_family, domain_labels)
        loss_total = loss_cls + self.family_lambda * loss_family

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        with torch.no_grad():
            valid = (attack_family >= 0) & (attack_family < self.n_attack_families)
            if valid.any():
                family_acc = (
                    logits_family[valid].argmax(dim=1) == attack_family[valid]
                ).float().mean().item()
            else:
                family_acc = float("nan")

        return {
            "loss_cls": loss_cls.item(),
            "loss_family": loss_family.item(),
            "loss_total": loss_total.item(),
            "family_lambda": self.family_lambda,
            "family_acc": family_acc,
        }
