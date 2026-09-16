"""Candidate-conditioned residual audit (the first DIE intervention).

For target sample i and candidate class k we transport class-validated source
residuals onto the target's semantic part and measure the frozen-text posterior
sensitivity.  With the class-offset vector b (from the selection-aware logit
intervention) and gamma' in {0, gamma}:

    J_i(k) = { j : y_j^s = k,  |log(||r_j^s||/||r_i^t||)| <= Delta,
               ||r_j^s|| / ||s_i^t|| <= kappa_r },
    u_i(k,gamma') = TailMean_rho_{j in J_i(k)} KL(q_{gamma'}(z_i) || q_{gamma'}(zhat_{i<-j})),
    m_i(k,gamma') = q_{gamma'}(z_i)_k - max_{l != k} q_{gamma'}(z_i)_l,
    c_i(k,gamma') = exp(-u_i / eta) * [m_i]_+,

where q_{gamma'}(z) = softmax(tau z T^T - gamma' b) and
zhat_{i<-j} = norm(s_i^t + r_j^s).  An audit is invalid (score 0) when fewer
than M valid donors are available.  The same s_i is used for every candidate,
preventing a candidate prototype from entering its own test.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from src.model import SourceMemory


def tail_mean(losses: torch.Tensor, rho: float) -> torch.Tensor:
    """Average the largest ceil(rho * n) entries along the last dimension."""
    k = max(1, int(np.ceil(rho * losses.shape[-1])))
    top, _ = losses.topk(k, dim=-1)
    return top.mean(dim=-1)


def candidate_classes(q0: torch.Tensor, n_cand: int, device: torch.device) -> torch.Tensor:
    """(n, n_cand) class indices per sample, from the frozen-text posterior."""
    if n_cand >= q0.shape[1]:
        return torch.arange(q0.shape[1], device=device).unsqueeze(0).expand(
            q0.shape[0], -1).contiguous()
    return q0.topk(n_cand, dim=-1).indices


def _shifted_posterior(z: torch.Tensor, T: torch.Tensor, tau: float,
                       b: torch.Tensor, gamma: float) -> torch.Tensor:
    logits = tau * z @ T.t()
    if gamma != 0 and b is not None and b.numel() > 0:
        logits = logits - gamma * b
    return torch.softmax(logits, dim=-1)


def audit_batch(
    z: torch.Tensor, s: torch.Tensor, r_t: torch.Tensor,
    T: torch.Tensor, tau: float, memory: SourceMemory,
    k_cand: torch.Tensor, cfg: dict, b: torch.Tensor, gamma: float,
    seed: int = 0,
) -> Dict[str, torch.Tensor]:
    """Audit every (i, k) candidate pair.

    Returns (all on device of z):
        u, m, c : (n, n_cand) score tensors for the shifted posterior gamma,
        valid   : (n, n_cand) bool (>= M valid donors), q0_rows: (n, n_cand)
                  q0 posterior values at candidate classes (unused here).
    """
    n, K_c = k_cand.shape
    device = z.device
    M = int(cfg["M"])
    rho = float(cfg["rho"])
    eta = float(cfg["eta"])
    delta = float(cfg["delta_energy"])     # Delta: log energy-difference bound
    kappa_r = float(cfg["kappa_r"])
    eps = float(cfg.get("eps_energy", 1e-6))

    u = torch.zeros(n, K_c, device=device)
    m = torch.zeros(n, K_c, device=device)
    c = torch.zeros(n, K_c, device=device)
    valid = torch.zeros(n, K_c, dtype=torch.bool, device=device)
    use_margin = bool(cfg.get("use_margin", True))
    tailmean = bool(cfg.get("tailmean", True))

    r_norm = r_t.norm(dim=1)                 # (n,)
    s_norm = s.norm(dim=1).clamp(min=eps)    # (n,)
    rng = np.random.default_rng(seed)

    for kk in range(K_c):
        cls = k_cand[:, kk]                  # (n,)
        for c_i in torch.unique(cls).tolist():
            mask = (cls == c_i)
            pool = memory.donors_of(c_i, device)   # (P, d) or empty
            if pool.shape[0] == 0:
                continue
            p_norm = pool.norm(dim=1).clamp(min=eps)   # (P,)
            # Valid-donor conditions (Eq. donors).
            logdiff = (p_norm.unsqueeze(0) / r_norm[mask].clamp(min=eps).unsqueeze(1)).log().abs()
            chi = p_norm.unsqueeze(0) / s_norm[mask].unsqueeze(1)
            ok = (logdiff <= delta) & (chi <= kappa_r)     # (n_m, P)
            n_m = int(mask.sum())
            idx_m = mask.nonzero(as_tuple=True)[0]
            for row in range(n_m):
                i = int(idx_m[row])
                oks = ok[row].nonzero(as_tuple=True)[0]
                if oks.numel() < M:
                    continue
                # Draw M distinct valid donors.
                sel = rng.choice(oks.numel(), size=M, replace=False)
                donors = pool[oks[sel]]                    # (M, d)
                zhat = F.normalize(s[i].unsqueeze(0) + donors, dim=-1)  # (M, d)
                q_i = _shifted_posterior(z[i].unsqueeze(0), T, tau, b, gamma)  # (1, K)
                q_ij = _shifted_posterior(zhat, T, tau, b, gamma)             # (M, K)
                kl = (q_i * torch.log(q_i.clamp(min=1e-12) / q_ij.clamp(min=1e-12))).sum(dim=-1)  # (M,)
                if tailmean:
                    u[i, kk] = tail_mean(kl.unsqueeze(0), rho)[0]
                else:
                    u[i, kk] = kl.mean()
                qi_k = q_i[0, c_i]
                top_other = q_i[0].clone()
                top_other[c_i] = -1.0
                m[i, kk] = qi_k - top_other.max()
                if use_margin:
                    c[i, kk] = torch.exp(-u[i, kk] / eta) * m[i, kk].clamp(min=0.0)
                else:
                    c[i, kk] = torch.exp(-u[i, kk] / eta)
                valid[i, kk] = True
    return {"u": u, "m": m, "c": c, "valid": valid}
