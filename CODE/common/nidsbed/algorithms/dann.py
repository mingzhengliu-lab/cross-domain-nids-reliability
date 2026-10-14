"""
DANN (Domain-Adversarial Neural Networks) - source-only multi-source DG 

Ganin et al., "Domain-Adversarial Training of Neural Networks",
JMLR 2016, vol. 17, no. 59, pp. 1-35.

Source-only multi-source variant: adversarially align source domains via a
gradient-reversal layer and a domain discriminator (Ganin et al., JMLR 2016).
No target-domain samples are used.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

from .base import Algorithm


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------
class _GradReverse(Function):
    """Gradient-reversal layer: identity on forward, multiplies the
    incoming gradient by ``-lambda`` on backward (DANN, Ganin 2015)."""

    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambda_, None


def grad_reverse(x: torch.Tensor, lambda_: float) -> torch.Tensor:
    return _GradReverse.apply(x, lambda_)


# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
class _DomainDiscriminator(nn.Module):
    """Two-layer MLP domain discriminator: feat_dim -> hidden -> n_domains."""

    def __init__(self, feat_dim: int = 64, hidden: int = 64, n_source_domains: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_source_domains),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return self.net(h)


# ---------------------------------------------------------------------------
# DANN algorithm
# ---------------------------------------------------------------------------
class DANN(Algorithm):
    """DANN trained on source domains only (no target UDA)."""

    def __init__(self, in_dim: int, n_classes: int, hparams: dict,
                 class_weights_per_domain=None, device: str = 'cuda',
                 n_source_domains: int = 3):
        super().__init__(in_dim, n_classes, hparams, class_weights_per_domain, device)

        self.build_default_backbone()

        feat_dim = hparams.get('feat_dim', 64)
        disc_hidden = hparams.get('disc_hidden', 64)
        self.discriminator = _DomainDiscriminator(
            feat_dim=feat_dim,
            hidden=disc_hidden,
            n_source_domains=n_source_domains,
        ).to(self.device)

        params = (list(self.backbone.parameters())
                  + list(self.classifier.parameters())
                  + list(self.discriminator.parameters()))
        self.optimizer = torch.optim.Adam(
            params,
            lr=hparams.get('lr', 1e-3),
            weight_decay=hparams.get('weight_decay', 0.0),
        )

        # lambda_mode ∈ {'fixed', 'schedule'}
        # - 'schedule': 2 / (1 + exp(-10 * p)) - 1,p = _step / total_steps_estimate
        self.lambda_mode = str(hparams.get('dann_lambda_mode', 'fixed'))
        self.lambda_value = float(hparams.get('dann_lambda_value', 1.0))

        self._step = 0
        self._total_steps_estimate = int(hparams.get('dann_total_steps', 10000))

    def _current_lambda(self) -> float:
        """Return the current GRL strength (fixed or Ganin schedule)."""
        if self.lambda_mode == 'schedule':
            p = min(1.0, self._step / max(1, self._total_steps_estimate))
            # GRL coefficient schedule from Ganin et al. 2016, eq. (10).
            return float(2.0 / (1.0 + torch.exp(torch.tensor(-10.0 * p))).item() - 1.0)
        else:  # 'fixed'
            return self.lambda_value

    def set_total_steps(self, total_steps: int):
        """Set the total-step estimate used by the lambda schedule."""
        self._total_steps_estimate = max(1, int(total_steps))

    def update(self, batch) -> dict:
        x, y, domain_labels = batch
        x = x.to(self.device)
        y = y.to(self.device)
        domain_labels = domain_labels.to(self.device)

        self.train()
        h = self.backbone(x)
        logits_cls = self.classifier(h)

        loss_cls = self.compute_per_domain_cls_loss(logits_cls, y, domain_labels)

        lambda_t = self._current_lambda()
        h_reversed = grad_reverse(h, lambda_t)
        logits_dom = self.discriminator(h_reversed)
        loss_dom = F.cross_entropy(logits_dom, domain_labels)

        loss_total = loss_cls + loss_dom

        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        self._step += 1

        with torch.no_grad():
            dom_pred = logits_dom.argmax(dim=1)
            dom_acc = (dom_pred == domain_labels).float().mean().item()

        return {
            'loss_cls': loss_cls.item(),
            'loss_dom': loss_dom.item(),
            'loss_total': loss_total.item(),
            'lambda_t': lambda_t,
            'dom_acc': dom_acc,
        }

