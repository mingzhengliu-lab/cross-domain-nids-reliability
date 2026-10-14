"""
Deep SVDD detector (Ruff et al., ICML 2018, "Deep One-Class Classification").

Three-step training:
  1. AE warmstart: short reconstruction-loss pretrain to initialize encoder.
  2. Center init: c = mean of phi(x_train) under warmstart encoder.
                 Components with |c_j| < eps_c are pushed to +/- eps_c
                 (Ruff trick to avoid trivial collapse).
  3. SVDD train: minimize sum ||phi(x) - c||^2 with weight decay.

Architecture choices (Ruff et al. 2018, §3 "Network architectures"):
  - All linear layers bias=False
  - All BatchNorms affine=False
  to prevent trivial constant mapping to c.

Score(x) = ||phi(x) - c||^2.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AnomalyDetector


class _SVDDEncoder(nn.Module):
    def __init__(self, in_dim: int = 39, hidden: int = 128, feat_dim: int = 32):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.BatchNorm1d(hidden, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden // 2, bias=False),
            nn.BatchNorm1d(hidden // 2, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, feat_dim, bias=False),
        )

    def forward(self, x):
        return self.layers(x)


class _SVDDDecoder(nn.Module):
    """For AE warmstart only. Symmetric to encoder, also bias-free."""
    def __init__(self, feat_dim: int = 32, hidden: int = 128, out_dim: int = 39):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(feat_dim, hidden // 2, bias=False),
            nn.BatchNorm1d(hidden // 2, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, hidden, bias=False),
            nn.BatchNorm1d(hidden, affine=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim, bias=False),
        )

    def forward(self, z):
        return self.layers(z)


class DeepSVDDDetector(AnomalyDetector):
    name = 'deep_svdd'

    def __init__(self, hidden: int = 128, feat_dim: int = 32,
                 lr: float = 1e-3, weight_decay: float = 1e-6,
                 warmstart_epochs: int = 10, svdd_epochs: int = 30,
                 batch_size: int = 4096, patience: int = 5, eps_c: float = 0.1,
                 device: str = 'cuda', random_state: int = 42, **kwargs):
        self.hidden = hidden
        self.feat_dim = feat_dim
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmstart_epochs = warmstart_epochs
        self.svdd_epochs = svdd_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.eps_c = eps_c
        self.device = device
        self.random_state = random_state
        self.encoder = None
        self.center = None

    # -----------------------------------------------------------------
    def _ae_warmstart(self, X_tr):
        D = X_tr.shape[1]
        encoder = _SVDDEncoder(D, self.hidden, self.feat_dim).to(self.device)
        decoder = _SVDDDecoder(self.feat_dim, self.hidden, D).to(self.device)
        opt = torch.optim.Adam(
            list(encoder.parameters()) + list(decoder.parameters()),
            lr=self.lr, weight_decay=self.weight_decay,
        )
        N = X_tr.shape[0]
        for _ in range(self.warmstart_epochs):
            encoder.train(); decoder.train()
            perm = torch.randperm(N, device=self.device)
            for i in range(0, N, self.batch_size):
                xb = X_tr[perm[i:i + self.batch_size]]
                if xb.shape[0] < 2:
                    continue
                z = encoder(xb)
                x_hat = decoder(z)
                loss = F.mse_loss(x_hat, xb)
                opt.zero_grad(); loss.backward(); opt.step()
        return encoder

    def _init_center(self, encoder, X):
        encoder.eval()
        zs = []
        with torch.no_grad():
            for i in range(0, X.shape[0], self.batch_size):
                xb = X[i:i + self.batch_size]
                if xb.shape[0] < 2:
                    continue
                zs.append(encoder(xb))
        z = torch.cat(zs, dim=0)
        c = z.mean(dim=0)
        # Ruff: avoid components too close to 0
        small = c.abs() < self.eps_c
        c[small & (c < 0)] = -self.eps_c
        c[small & (c >= 0)] = self.eps_c
        return c.detach()

    # -----------------------------------------------------------------
    def fit(self, X_benign):
        torch.manual_seed(self.random_state)
        if self.device == 'cuda' and torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.random_state)

        X_all = torch.from_numpy(np.asarray(X_benign, dtype=np.float32)).to(self.device)
        N = X_all.shape[0]
        n_val = max(1024, N // 10)
        perm = torch.randperm(N, device=self.device)
        X_val = X_all[perm[:n_val]]
        X_tr = X_all[perm[n_val:]]

        # Step 1: AE warmstart
        self.encoder = self._ae_warmstart(X_tr)

        # Step 2: center init
        self.center = self._init_center(self.encoder, X_tr)

        # Step 3: SVDD training
        opt = torch.optim.Adam(self.encoder.parameters(),
                               lr=self.lr, weight_decay=self.weight_decay)
        best_val = float('inf'); best_state = None; bad = 0
        N_tr = X_tr.shape[0]
        for epoch in range(self.svdd_epochs):
            self.encoder.train()
            perm_tr = torch.randperm(N_tr, device=self.device)
            for i in range(0, N_tr, self.batch_size):
                xb = X_tr[perm_tr[i:i + self.batch_size]]
                if xb.shape[0] < 2:
                    continue
                z = self.encoder(xb)
                d2 = ((z - self.center) ** 2).sum(dim=1)
                loss = d2.mean()
                opt.zero_grad(); loss.backward(); opt.step()

            self.encoder.eval()
            with torch.no_grad():
                vd = []
                for i in range(0, X_val.shape[0], self.batch_size):
                    xb = X_val[i:i + self.batch_size]
                    if xb.shape[0] < 2:
                        continue
                    z = self.encoder(xb)
                    vd.append(((z - self.center) ** 2).sum(dim=1))
                val_loss = torch.cat(vd).mean().item() if vd else float('inf')

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.encoder.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            self.encoder.load_state_dict(best_state)

    def score(self, X):
        self.encoder.eval()
        out = np.empty(len(X), dtype=np.float32)
        bs = self.batch_size
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = torch.from_numpy(np.asarray(X[i:i + bs], dtype=np.float32)).to(self.device)
                z = self.encoder(xb)
                d2 = ((z - self.center) ** 2).sum(dim=1).detach().cpu().numpy()
                out[i:i + bs] = d2
        return out
