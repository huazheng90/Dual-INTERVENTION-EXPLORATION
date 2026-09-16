"""Selection-aware logit intervention and the risk-stratified split.

Exponential moving averages N_k (targets predicted as k by the unadjusted
posterior) and A_k (targets passing the raw audit) yield per-class admission
rates

    a_k = (A_k + alpha) / (N_k + 2 alpha),
    b_k = clip(log a_k - mean_{K_a} log a_l, -b_max, b_max),

with b_k = 0 for inactive classes and b = 0 when K_a is empty.  Classes with
relatively low admission rates receive larger relative logits (q_gamma uses
logits - gamma b), generating alternative hypotheses without estimating the
target prior.

Samples supported by both interventions form the conservative core C; samples
rejected by the raw audit but consistently re-classified by the adjustment
form the exploratory frontier F.
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch

from src.model import SourceMemory


class LogitIntervention:
    """EMA statistics and the class-offset vector b.

    With ``offset_stat="admission"`` (default, DIE) a_k is the audit-gated
    admission rate (A_k + alpha)/(N_k + 2 alpha); with ``offset_stat=
    "frequency"`` (the DLAE-style ablation) a_k is the raw prediction
    frequency, which inherits the bias-frequency confounding the paper
    argues against.
    """

    def __init__(self, K: int, alpha_smooth: float = 1.0, b_max: float = 2.0,
                 n_min: int = 10, ema: float = 0.9, device: torch.device = None,
                 offset_stat: str = "admission"):
        self.K = K
        self.alpha = alpha_smooth
        self.b_max = b_max
        self.n_min = n_min
        self.ema = ema
        self.offset_stat = offset_stat
        dev = device if device is not None else torch.device("cpu")
        self.N = torch.zeros(K, device=dev)
        self.A = torch.zeros(K, device=dev)
        self.b = torch.zeros(K, device=dev)
        self._active = False

    def update(self, yhat0: torch.Tensor, passed_raw: torch.Tensor) -> None:
        """Accumulate this round's statistics (EMA over rounds)."""
        n = yhat0.shape[0]
        for i in range(n):
            k = int(yhat0[i])
            self.N[k] = self.ema * self.N[k] + (1 - self.ema) * 1.0
            self.A[k] = self.ema * self.A[k] + (1 - self.ema) * float(passed_raw[i])

    def compute_offset(self) -> torch.Tensor:
        """b_k = clip(log a_k - mean, -b_max, b_max); zero for inactive."""
        b = torch.zeros_like(self.b)
        active = (self.N >= self.n_min)
        if active.sum() == 0:
            self.b = b
            return b
        if self.offset_stat == "frequency":
            a = (self.N + self.alpha) / (self.N.sum() + self.K * self.alpha)
        else:
            a = (self.A + self.alpha) / (self.N + 2 * self.alpha)
        loga = torch.log(a.clamp(min=1e-8))
        mean_log = loga[active].mean()
        b[active] = (loga[active] - mean_log).clamp(-self.b_max, self.b_max)
        self.b = b
        self._active = True
        return b


def core_split(c0: torch.Tensor, cg: torch.Tensor, k0: torch.Tensor,
               kg: torch.Tensor, delta: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """Core mask and per-sample core weight (Eq. core).

    c0/cg: certificates at gamma'=0 and gamma'=gamma for the predicted class
    (i.e., indexed at k_i^0); k0/kg: unadjusted/adjusted argmax classes.
    """
    agree = (k0 == kg)
    wC = torch.stack([c0, cg], dim=1).min(dim=1).values
    in_core = agree & (wC >= delta)
    return in_core, wC


def frontier_split(g0: torch.Tensor, kg: torch.Tensor, k0: torch.Tensor,
                   cg: torch.Tensor, v: torch.Tensor, H: torch.Tensor,
                   delta: float) -> torch.Tensor:
    """Frontier-candidate mask (Eq. frontier): raw audit rejected, adjusted
    class differs, the adjusted class passes both residual and centroid checks
    in two consecutive rounds (H), and the adjusted certificate >= delta."""
    return ((~g0) & (kg != k0) & (cg >= delta) & (v == kg) & H)


def frontier_quota(in_frontier: torch.Tensor, k0: torch.Tensor,
                   core_mask: torch.Tensor, score: torch.Tensor,
                   beta: float) -> torch.Tensor:
    """Per class keep at most ceil(beta * max(1, |C_k|)) highest-scoring
    frontier candidates."""
    out = torch.zeros_like(in_frontier)
    device = in_frontier.device
    n = in_frontier.shape[0]
    for c in torch.unique(k0).tolist():
        f_idx = (in_frontier & (k0 == c)).nonzero(as_tuple=True)[0]
        if f_idx.numel() == 0:
            continue
        n_core = int((core_mask & (k0 == c)).sum().item())
        quota = int(np.ceil(beta * max(1, n_core)))
        if quota <= 0:
            continue
        sc = score[f_idx]
        take = min(quota, f_idx.numel())
        _, top = sc.topk(take)
        out[f_idx[top]] = True
    return out
