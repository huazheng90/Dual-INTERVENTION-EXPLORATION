"""Batch-local alternating solver for the DIE decomposition (Eq. decomp_loss).

For each paired mini-batch we alternate ``J_alt`` block updates:

1. ``A_d``: one gradient step on the batch decomposition objective
   (residual bases detached; no gradient is unrolled through this local
   solver).
2. ``E_d``: the closed-form row-wise proximal (soft-thresholding) update.

The returned ``(A, E)`` are detached and held fixed for the global backward
pass, matching the paper: the audit snapshot's decomposition is detached for
the following optimization round.
"""
from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F


def _proximal_rows(U: torch.Tensor, xi: float, eps: float) -> torch.Tensor:
    norms = U.norm(dim=1, keepdim=True)
    scale = torch.clamp(1.0 - xi / torch.clamp(norms, min=eps), min=0.0)
    return scale * U


def solve_decomposition(
    Z: torch.Tensor,
    Pbar: torch.Tensor,
    C: torch.Tensor,
    V: torch.Tensor,
    B: torch.Tensor,
    beta_a: float,
    xi: float,
    eps_prox: float,
    J_alt: int,
    lr_block: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Alternating solve of (A, E) for one domain given a fixed (Pbar, C, V, B).

    Minimizes 0.5||Z - Pbar C V^T - A B^T - E||^2 + beta_a ||A||^2 + xi||E||_{2,1}
    over A (gradient steps) and E (proximal rows).
    """
    n = Z.shape[0]
    device = Z.device
    A = torch.zeros(n, B.shape[1], device=device)
    E = torch.zeros_like(Z)
    CVt = C @ V.t()

    def _block_loss(Av: torch.Tensor) -> torch.Tensor:
        r = Av @ B.t()
        recon = Z - Pbar @ CVt - r - E
        return 0.5 * recon.pow(2).mean() + beta_a * Av.pow(2).mean()

    for _ in range(J_alt):
        with torch.enable_grad():
            A.requires_grad_(True)
            loss = _block_loss(A)
            grad = torch.autograd.grad(loss, A)[0]
        with torch.no_grad():
            A = A - lr_block * grad
        A = A.detach()
        U = Z - Pbar @ CVt - A @ B.t()
        E = _proximal_rows(U, xi, eps_prox)

    return A.detach(), E.detach()
