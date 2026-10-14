"""
Autoencoder anomaly detector.

Trains a bottleneck AE with MSE on source benign (GPU-resident batching).
Score(x) = mean squared reconstruction error per sample.

Architecture: 39 -> 128 -> 64 -> 16 -> 64 -> 128 -> 39 with BN+ReLU+Dropout.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import AnomalyDetector


class _AENet(nn.Module):
    def __init__(self, in_dim: int = 39, hidden: int = 128, bottleneck: int = 16,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, bottleneck),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden // 2),
            nn.BatchNorm1d(hidden // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden // 2, hidden),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, in_dim),
        )

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class AEDetector(AnomalyDetector):
    name = 'autoencoder'

    def __init__(self, hidden: int = 128, bottleneck: int = 16, dropout: float = 0.1,
                 lr: float = 1e-3, weight_decay: float = 0.0,
                 max_epochs: int = 30, batch_size: int = 4096, patience: int = 5,
                 device: str = 'cuda', random_state: int = 42, **kwargs):
        self.hidden = hidden
        self.bottleneck = bottleneck
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.device = device
        self.random_state = random_state
        self.net = None
        self.history = []

    def fit(self, X_benign):
        torch.manual_seed(self.random_state)
        if torch.cuda.is_available() and self.device == 'cuda':
            torch.cuda.manual_seed_all(self.random_state)

        # Move all benign data to GPU once (matches main benchmark pattern).
        X_all = torch.from_numpy(np.asarray(X_benign, dtype=np.float32)).to(self.device)
        N, D = X_all.shape

        # Hold out 10% (or >=1024) for early stopping.
        n_val = max(1024, N // 10)
        perm = torch.randperm(N, device=self.device)
        X_val = X_all[perm[:n_val]]
        X_tr = X_all[perm[n_val:]]
        N_tr = X_tr.shape[0]

        self.net = _AENet(in_dim=D, hidden=self.hidden, bottleneck=self.bottleneck,
                          dropout=self.dropout).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr,
                               weight_decay=self.weight_decay)

        best_val = float('inf')
        best_state = None
        bad = 0

        for epoch in range(self.max_epochs):
            self.net.train()
            perm_tr = torch.randperm(N_tr, device=self.device)
            train_loss = 0.0
            n_batches = 0
            for i in range(0, N_tr, self.batch_size):
                idx = perm_tr[i:i + self.batch_size]
                xb = X_tr[idx]
                if xb.shape[0] < 2:  # BN needs >= 2
                    continue
                x_hat, _ = self.net(xb)
                loss = F.mse_loss(x_hat, xb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                train_loss += loss.item()
                n_batches += 1
            train_loss /= max(n_batches, 1)

            # Validation
            self.net.eval()
            with torch.no_grad():
                vl = 0.0
                nb = 0
                for i in range(0, X_val.shape[0], self.batch_size):
                    xb = X_val[i:i + self.batch_size]
                    if xb.shape[0] < 2:
                        continue
                    x_hat, _ = self.net(xb)
                    vl += F.mse_loss(x_hat, xb).item()
                    nb += 1
                val_loss = vl / max(nb, 1)

            self.history.append({'epoch': epoch + 1, 'train': train_loss, 'val': val_loss})

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {k: v.detach().clone() for k, v in self.net.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= self.patience:
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)

    def score(self, X):
        self.net.eval()
        out = np.empty(len(X), dtype=np.float32)
        bs = self.batch_size
        with torch.no_grad():
            for i in range(0, len(X), bs):
                xb = torch.from_numpy(np.asarray(X[i:i + bs], dtype=np.float32)).to(self.device)
                x_hat, _ = self.net(xb)
                err = ((x_hat - xb) ** 2).mean(dim=1).detach().cpu().numpy()
                out[i:i + bs] = err
        return out
