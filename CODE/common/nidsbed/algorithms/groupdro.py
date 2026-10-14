"""
GroupDRO (Distributionally Robust Optimization with Group Structure)
    Sagawa, Koh, Hashimoto, Liang
    ICLR 2020, "Distributionally Robust Neural Networks for Group Shifts"

Groups are source domains. Domain losses are reweighted with an
exponentiated-gradient update (Sagawa et al., ICLR 2020 Appendix A).
"""
import torch
import torch.nn.functional as F

from .base import Algorithm


class GroupDRO(Algorithm):
    """GroupDRO with source domains as groups."""

    def __init__(self, in_dim: int, n_classes: int, hparams: dict,
                 class_weights_per_domain=None, device: str = 'cuda',
                 n_domains: int = 3):
        super().__init__(in_dim, n_classes, hparams, class_weights_per_domain, device)

        self.build_default_backbone()
        self.build_optimizer()

        self.n_domains = n_domains
        self.groupdro_eta = float(hparams.get('groupdro_eta', 0.01))

        self.q = torch.ones(n_domains, device=self.device) / n_domains

    def update(self, batch) -> dict:
        x, y, domain_labels = batch
        x = x.to(self.device)
        y = y.to(self.device)
        domain_labels = domain_labels.to(self.device)

        self.train()
        h = self.backbone(x)
        logits = self.classifier(h)

        unique_domains = torch.unique(domain_labels)
        per_domain_losses = {}
        for d in unique_domains:
            mask = (domain_labels == d)
            if mask.sum() == 0:
                continue
            w = self.class_weights_per_domain.get(int(d.item()))
            if w is not None:
                w = w.to(self.device)
            loss_d = F.cross_entropy(logits[mask], y[mask], weight=w)
            per_domain_losses[int(d.item())] = loss_d

        with torch.no_grad():
            for d_id, loss_d in per_domain_losses.items():
                self.q[d_id] *= torch.exp(self.groupdro_eta * loss_d.detach())
            self.q /= self.q.sum()

        assert len(per_domain_losses) == self.n_domains, (
            f"GroupDRO expects all {self.n_domains} source domains in each batch, "
            f"got {len(per_domain_losses)}. Check DomainBalancedBatchSampler."
        )

        loss_total = torch.stack([
            self.q[d_id] * loss_d
            for d_id, loss_d in per_domain_losses.items()
        ]).sum()

        loss_cls_unweighted = torch.stack(list(per_domain_losses.values())).mean()

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        return {
            'loss_cls': loss_cls_unweighted.item(),
            'loss_total': loss_total.item(),
            'q_max': float(self.q.max().item()),
            'q_min': float(self.q.min().item()),
        }
