"""DIE losses: source supervision, decomposition, core CE, frontier KL.

The audit snapshot supplies detached masks, scores, and soft targets; the
optimization round back-propagates only to the trainable parameters.
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn.functional as F

from src.model import SemanticFactorization


def source_loss(p: torch.Tensor, q0: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Cross-entropy on both posteriors (Eq. posterior pair)."""
    return F.cross_entropy(p, y) + F.cross_entropy(q0, y)


def decomposition_loss(
    z: torch.Tensor, Pbar: torch.Tensor, C: torch.Tensor, V: torch.Tensor,
    A: torch.Tensor, B: torch.Tensor, E: torch.Tensor,
    V_T: torch.Tensor, cfg: dict, n_d: float,
) -> Tuple[torch.Tensor, dict]:
    """One domain's L_dec terms (Eq. decomp_loss).

    L_dec = ||R||_F^2/(2 n_d) + beta_a ||A||^2/n_d + xi ||E||_{2,1}/n_d
            + beta_b ||B||^2_F + lambda_T ||V V^T - V_T V_T^T||_F^2
            + lambda_perp ||V^T B||_F^2
    """
    R = z - Pbar @ (C @ V.t()) - A @ B.t() - E
    n = z.shape[0]
    l_recon = R.pow(2).sum() / (2.0 * n_d)
    l_a = float(cfg["beta_a"]) * A.pow(2).sum() / n_d
    l_e = float(cfg["xi"]) * E.norm(dim=1).sum() / n_d
    l_b = float(cfg["beta_b"]) * B.pow(2).sum()
    l_text = float(cfg["lambda_T"]) * (V @ V.t() - V_T @ V_T.t()).pow(2).sum()
    l_perp = float(cfg["lambda_perp"]) * (V.t() @ B).pow(2).sum()
    total = l_recon + l_a + l_e + l_b + l_text + l_perp
    return total, {"l_recon": l_recon.item(), "l_a": l_a.item(), "l_e": l_e.item(),
                   "l_b": l_b.item(), "l_text": l_text.item(), "l_perp": l_perp.item()}


def core_loss(phat: torch.Tensor, in_core: torch.Tensor, wC: torch.Tensor,
              k0: torch.Tensor) -> torch.Tensor:
    """L_C: stop-gradient-weighted CE on the predicted class (Eq. objective)."""
    if in_core.sum() == 0:
        return torch.zeros((), device=phat.device)
    idx = in_core.nonzero(as_tuple=True)[0]
    w = wC[idx].detach()
    ce = F.cross_entropy(phat[idx], k0[idx], reduction="none")
    return (w * ce).sum() / max(float(phat.shape[0]), 1.0)


def frontier_loss(phat: torch.Tensor, in_frontier: torch.Tensor,
                  weight: torch.Tensor, t_i: torch.Tensor) -> torch.Tensor:
    """L_F: stop-gradient-weighted KL to the detached soft target.

    The target t_i is the audit's shifted posterior q_gamma(z_i); weight is the
    adjusted certificate c_i(k_i^gamma, gamma)."""
    if in_frontier.sum() == 0:
        return torch.zeros((), device=phat.device)
    idx = in_frontier.nonzero(as_tuple=True)[0]
    w = weight[idx].detach()
    kl = (t_i[idx] * torch.log(t_i[idx].clamp(min=1e-12) / phat[idx].clamp(min=1e-12))).sum(dim=-1)
    return (w * kl).sum() / max(float(phat.shape[0]), 1.0)
