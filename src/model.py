"""DIE model components: semantic factorization and the source memory.

DIE (Dual-Intervention Exploration) decomposes a unit-normalized CLIP feature
z into a semantic part s and a domain-residual part r:

    s_i = p(z_i) C V^T,   r_i = a_i B_d^T,   z_i = s_i + r_i + e_i,

where V in R^{d x r} is an orthonormal semantic basis (thin-QR parameterized),
C = T V with the frozen text prototypes T, and B_d in R^{d x m} is a
domain-specific residual basis (d in {s, t}).  The adaptive posterior is

    p_theta(z) = softmax(tau_h (z V) (T V)^T)

with a learnable projection scale tau_h.  The frozen-text reference posterior
is q_0(z) = softmax(tau z T^T).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


def thin_qf(A: torch.Tensor) -> torch.Tensor:
    """Thin QR with fixed column signs: A -> Q in R^{d x r}."""
    Q, R = torch.linalg.qr(A, mode="reduced")
    d = torch.diag(R).sign()
    d = torch.where(d == 0, torch.ones_like(d), d)
    return Q * d


class SemanticFactorization(nn.Module):
    """V (semantic basis) + B_s/B_t (residual bases) + learnable scale tau_h."""

    def __init__(self, d: int, r: int, m: int, init_perturb: float = 1e-3):
        super().__init__()
        self.d = d
        self.r = r
        self.m = m
        # Vbar -> V = thin_qf(Vbar); initialized at fit time from V_T.
        self.Vbar = nn.Parameter(torch.randn(d, r) * init_perturb)
        self.B_s = nn.Parameter(torch.randn(d, m) * 0.01)
        self.B_t = nn.Parameter(torch.randn(d, m) * 0.01)
        # Learnable projection scale used by p_theta (log-parameterized).
        self.logit_scale_h = nn.Parameter(torch.tensor(float(4.6052)))  # exp ~ 100

    def V(self) -> torch.Tensor:
        return thin_qf(self.Vbar)

    def scale_h(self) -> torch.Tensor:
        return torch.exp(self.logit_scale_h).clamp(min=1.0, max=1e4)

    def basis(self, domain: str) -> torch.Tensor:
        return self.B_s if domain == "s" else self.B_t


class SourceMemory:
    """Class-stratified source donor memory + source class centroids.

    Centroids live in the shared semantic coordinates (h = z V) and are
    recomputed each audit round from the labeled source features.  The donor
    memory stores detached residual vectors r_j^s per class, FIFO-capped.
    """

    def __init__(self, K: int, donor_capacity: int = 4096, mu: float = 0.99):
        self.K = K
        self.capacity = donor_capacity
        self.mu = mu
        self._r: List[List[torch.Tensor]] = [[] for _ in range(K)]
        self.centroids: Optional[torch.Tensor] = None  # (K, r)

    # ------------------------------------------------------------ donors
    def push(self, r: torch.Tensor, y: torch.Tensor) -> None:
        r = r.detach()
        for i in range(r.shape[0]):
            c = int(y[i])
            self._r[c].append(r[i])
            if len(self._r[c]) > self.capacity:
                self._r[c] = self._r[c][-self.capacity:]

    def donor_counts(self) -> List[int]:
        return [len(d) for d in self._r]

    def sample_donors(self, cls: int, M: int, seed: int = 0,
                      device: torch.device = torch.device("cpu")) -> Optional[torch.Tensor]:
        """Draw up to M distinct source residuals of class cls (detached)."""
        pool = self._r[cls]
        if len(pool) < 1:
            return None
        import numpy as np
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pool), size=min(M, len(pool)), replace=False)
        return torch.stack([pool[i] for i in idx]).to(device)

    def donors_of(self, cls: int, device: torch.device) -> torch.Tensor:
        if len(self._r[cls]) == 0:
            return torch.zeros(0, 0, device=device)
        return torch.stack(self._r[cls]).to(device)

    # ---------------------------------------------------------- centroids
    def update_centroids(self, h_s: torch.Tensor, y_s: torch.Tensor) -> None:
        """Class means of source semantic coordinates (fresh each round)."""
        K = self.K
        device = h_s.device
        cent = torch.zeros(K, h_s.shape[1], device=device)
        cnt = torch.zeros(K, device=device)
        for i in range(h_s.shape[0]):
            c = int(y_s[i])
            cent[c] += h_s[i]
            cnt[c] += 1.0
        cent = cent / cnt.clamp(min=1.0).unsqueeze(1)
        self.centroids = F.normalize(cent, dim=-1)

    def nearest_centroid(self, h: torch.Tensor) -> torch.Tensor:
        """(n,) class of the nearest source centroid for each semantic coord."""
        if self.centroids is None:
            raise RuntimeError("SourceMemory.update_centroids must run first")
        return (h @ self.centroids.t()).argmax(dim=1)


class ClassDiscriminator(nn.Module):
    """Not used by DIE; kept for parity with the shared codebase if needed."""

    def __init__(self, d: int, K: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                 nn.Linear(hidden, K))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
