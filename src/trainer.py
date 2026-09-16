"""DIE trainer: warm-up then dual-intervention audit/optimization rounds.

Implements Algorithm 1 of the paper.  The vision tower is frozen and image
features are cached ahead of time (see data.py), so most computation operates
on feature arrays.  The audit round freezes a snapshot of the current model,
computes target decisions, source residuals, and source class centroids, and
detaches every mask, score, and soft target for the following optimization
round.

"""
from __future__ import annotations

import os
import time
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import losses as L
from src.audit import audit_batch, candidate_classes, tail_mean
from src.factorization import solve_decomposition
from src.intervention import (LogitIntervention, core_split, frontier_quota,
                              frontier_split)
from src.model import SemanticFactorization, SourceMemory
from src.utils import LOGGER


class FeatureSampler:
    """Shuffle-once epoch sampler over cached feature arrays (returns indices)."""

    def __init__(self, feats: torch.Tensor, labels: Optional[torch.Tensor],
                 batch_size: int, iters_per_epoch: int, seed: int):
        self.feats = feats
        self.labels = labels
        self.batch_size = batch_size
        self.iters = iters_per_epoch
        self.g = torch.Generator().manual_seed(seed)
        self.order = torch.randperm(feats.shape[0], generator=self.g)
        self.pos = 0

    def __iter__(self):
        return self

    def __next__(self) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        n = self.feats.shape[0]
        idx = []
        for _ in range(self.batch_size):
            if self.pos >= n:
                self.order = torch.randperm(n, generator=self.g)
                self.pos = 0
            idx.append(int(self.order[self.pos]))
            self.pos += 1
        idx = torch.tensor(idx, device=self.feats.device)
        z = self.feats[idx]
        y = self.labels[idx] if self.labels is not None else None
        return z, y, idx


class DIETrainer:
    def __init__(self, cfg: dict, backend: nn.Module, device: torch.device,
                 strong_aug_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None):
        self.cfg = cfg
        self.device = device
        self.backend = backend
        self.d = backend.d
        self.K = cfg["num_classes"]
        self.r = cfg["semantic_rank"]
        self.m = cfg["residual_rank"]
        self.strong_aug_fn = strong_aug_fn

        self.factorization = SemanticFactorization(self.d, self.r, self.m,
                                                   cfg.get("init_perturb", 1e-3)).to(device)
        self.memory = SourceMemory(self.K, cfg.get("donor_capacity", 4096))
        self.interv = LogitIntervention(self.K,
                                        alpha_smooth=cfg.get("alpha_smooth", 1.0),
                                        b_max=cfg.get("b_max", 2.0),
                                        n_min=cfg.get("n_min", 10),
                                        ema=cfg.get("ema_stats", 0.9),
                                        device=device,
                                        offset_stat=cfg.get("offset_stat", "admission"))
        self.V_T: Optional[torch.Tensor] = None
        self.tau = float(getattr(backend, "text_logit_scale", 100.0))
        if cfg.get("text_logit_scale", 0.0) > 0:
            self.tau = float(cfg["text_logit_scale"])
        self.H_prev: Optional[torch.Tensor] = None   # frontier candidates at round e-1

        self.opt_params = self._collect_trainable()
        self.optimizer = torch.optim.AdamW(self.opt_params, lr=cfg["lr"],
                                           weight_decay=cfg.get("weight_decay", 0.0))
        total_epochs = cfg["warmup_epochs"] + cfg["adapt_epochs"]
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs, eta_min=1e-6)
        self.step = 0

    # ------------------------------------------------------------------ utils
    def _collect_trainable(self) -> List[torch.nn.Parameter]:
        params = [p for p in self.factorization.parameters() if p.requires_grad]
        if self.cfg.get("train_prompt", False):
            params += [p for p in self.backend.parameters() if p.requires_grad]
        return params

    def _text_prototypes(self, class_names: Sequence[str]) -> torch.Tensor:
        return F.normalize(self.backend.text_prototypes(class_names), dim=-1)

    def _q0(self, z: torch.Tensor, T: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.tau * z @ T.t(), dim=-1)

    def _qg(self, z: torch.Tensor, T: torch.Tensor,
            b: torch.Tensor, gamma: float) -> torch.Tensor:
        logits = self.tau * z @ T.t()
        if gamma != 0 and b is not None and b.numel() > 0:
            logits = logits - gamma * b
        return torch.softmax(logits, dim=-1)

    def _p(self, z: torch.Tensor, V: torch.Tensor, C: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.factorization.scale_h() * (z @ V) @ C.t(), dim=-1)

    # ------------------------------------------------------------ checkpoint
    def save(self, path: str, extra: Optional[dict] = None) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {"factorization": self.factorization.state_dict(),
                 "optimizer": self.optimizer.state_dict(),
                 "scheduler": self.scheduler.state_dict(),
                 "step": self.step, "V_T": self.V_T, "cfg": self.cfg}
        if extra:
            state.update(extra)
        torch.save(state, path)
        LOGGER.info("checkpoint saved to %s", path)

    def load(self, path: str, load_optimizer: bool = True) -> None:
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.factorization.load_state_dict(state["factorization"])
        self.V_T = state.get("V_T")
        self.step = state.get("step", 0)
        if load_optimizer and "optimizer" in state:
            self.optimizer.load_state_dict(state["optimizer"])
            self.scheduler.load_state_dict(state["scheduler"])

    # ------------------------------------------------------------- features
    @torch.no_grad()
    def decompose_all(self, z: torch.Tensor, T: torch.Tensor,
                      y: Optional[torch.Tensor], domain: str,
                      batch: int = 256) -> Dict[str, torch.Tensor]:
        """Decompose a full feature matrix; returns concatenated tensors."""
        V = self.factorization.V()
        C = T @ V
        outs = {k: [] for k in ["z", "h", "p", "q0", "s", "r", "A", "E"]}
        n = z.shape[0]
        for i in range(0, n, batch):
            zb = z[i:i + batch].to(self.device)
            yb = y[i:i + batch] if y is not None else None
            f = self._project(zb, T, V, C, yb, domain)
            for k in outs:
                outs[k].append(f[k].cpu())
        return {k: torch.cat(v, dim=0) for k, v in outs.items()}

    @torch.no_grad()
    def _project(self, z: torch.Tensor, T: torch.Tensor, V: torch.Tensor,
                 C: torch.Tensor, y: Optional[torch.Tensor], domain: str) -> Dict[str, torch.Tensor]:
        q0 = self._q0(z, T)
        h = z @ V
        p = self._p(z, V, C)
        Pbar = F.one_hot(y, num_classes=self.K).float() \
            if (domain == "s" and y is not None) else p.detach()
        B = self.factorization.basis(domain)
        A, E = solve_decomposition(
            Z=z, Pbar=Pbar, C=C, V=V, B=B,
            beta_a=self.cfg.get("beta_a", 1e-3), xi=self.cfg.get("xi", 1e-3),
            eps_prox=self.cfg.get("eps_prox", 1e-8),
            J_alt=self.cfg.get("j_alt", 5), lr_block=self.cfg.get("lr_block", 0.1))
        s = Pbar @ (C @ V.t())
        r = A @ B.t()
        return {"z": z, "h": h, "p": p, "q0": q0, "s": s, "r": r, "A": A, "E": E,
                "Pbar": Pbar}

    def _phat(self, z: torch.Tensor, idx: torch.Tensor, V: torch.Tensor,
              C: torch.Tensor) -> torch.Tensor:
        """Strongly augmented posterior for core/frontier samples."""
        if self.strong_aug_fn is not None and idx is not None:
            z_aug = self.strong_aug_fn(idx)
        else:
            z_aug = z + self.cfg.get("feature_aug_std", 0.02) * torch.randn_like(z)
            z_aug = F.normalize(z_aug, dim=-1)
        return self._p(z_aug, V, C)

    # ---------------------------------------------------------------- audit
    @torch.no_grad()
    def _audit_round(self, class_names: Sequence[str], T: torch.Tensor,
                     z_s: torch.Tensor, y_s: torch.Tensor, z_t: torch.Tensor,
                     round_idx: int, seeds: List[int]) -> Dict[str, torch.Tensor]:
        """One audit pass over the full target set (and source for memory)."""
        cfg = self.cfg
        device = self.device
        fs = self.decompose_all(z_s, T, y_s, "s")
        ft = self.decompose_all(z_t, T, None, "t")
        z_t = z_t.to(device)
        T = T.to(device)

        # Source donor memories and centroids (round-fresh).
        self.memory.push(fs["r"].to(device), y_s.to(device))
        self.memory.update_centroids(fs["h"].to(device), y_s.to(device))

        q0 = self._q0(z_t, T)
        k0 = q0.argmax(dim=1)
        k_cand = candidate_classes(q0, cfg.get("n_candidates", 10), device)
        b_zero = torch.zeros(self.K, device=device)

        # Offsets use statistics from the PRECEDING round; then update.
        b = self.interv.compute_offset() if self.interv._active else b_zero

        use_audit = bool(cfg.get("use_audit", True))
        if not use_audit:
            # Passive margin+entropy filter (Table ablation: w/o audit).
            q0m = q0.max(dim=1).values - (q0 - torch.eye(self.K, device=device)[k0]).max(dim=1).values
            ent = -(q0 * torch.log(q0.clamp(min=1e-12))).sum(dim=1)
            ent = ent / float(np.log(self.K))
            c0 = q0m * (1.0 - ent).clamp(min=0.0)
            cg_k0 = c0.clone()
            cg_kg = c0.clone()
            g0 = c0 >= float(cfg["delta"])
            self.interv.update(k0, g0)
            b_next = self.interv.compute_offset()
            qg = self._qg(z_t, T, b, cfg.get("gamma", 1.0))
            kg = qg.argmax(dim=1)
            in_core, wC = core_split(c0, cg_k0, k0, kg, float(cfg["delta"]))
            v = self.memory.nearest_centroid(ft["h"].to(device))
            H = self.H_prev if self.H_prev is not None else torch.zeros_like(g0)
            f_cand = frontier_split(g0, kg, k0, cg_kg, v, H, float(cfg["delta"]))
            self.H_prev = f_cand.clone()
            in_frontier = frontier_quota(f_cand, k0, in_core, cg_kg, float(cfg["beta"]))
            LOGGER.info("audit round %d (no-audit filter): core=%d frontier=%d",
                        round_idx, int(in_core.sum()), int(in_frontier.sum()))
            return {"in_core": in_core, "wC": wC, "in_frontier": in_frontier,
                    "cg_kg": cg_kg, "t_i": qg, "k0": k0, "kg": kg, "b_next": b_next,
                    "q0": q0, "k_cand": k_cand, "c0": c0, "g0": g0}

        a0 = audit_batch(z_t, ft["s"].to(device), ft["r"].to(device), T, self.tau,
                         self.memory, k_cand, cfg, b_zero, 0.0, seed=seeds[0] + round_idx)
        ag = audit_batch(z_t, ft["s"].to(device), ft["r"].to(device), T, self.tau,
                         self.memory, k_cand, cfg, b, cfg.get("gamma", 1.0),
                         seed=seeds[0] + round_idx + 1)

        c0 = a0["c"].gather(1, (k_cand == k0.unsqueeze(1)).nonzero(as_tuple=True)[1].unsqueeze(1)).squeeze(1)
        g0 = c0 >= float(cfg["delta"])                          # raw admission
        self.interv.update(k0, g0)
        # Raw admission disparity -> offset for the NEXT round.
        b_next = self.interv.compute_offset()

        qg = self._qg(z_t, T, b, cfg.get("gamma", 1.0))
        kg = qg.argmax(dim=1)
        cg_k0 = ag["c"].gather(1, (k_cand == k0.unsqueeze(1)).nonzero(as_tuple=True)[1].unsqueeze(1)).squeeze(1)
        # k^gamma may fall outside the candidate window; its certificate is 0
        # there, so such samples cannot enter the frontier.
        cg_kg = torch.zeros(z_t.shape[0], device=device)
        in_cand = (k_cand == kg.unsqueeze(1)).any(dim=1)
        if in_cand.sum() > 0:
            col_kg = (k_cand[in_cand] == kg[in_cand].unsqueeze(1)).nonzero(as_tuple=True)[1]
            cg_kg[in_cand] = ag["c"][in_cand, col_kg]

        in_core, wC = core_split(c0, cg_k0, k0, kg, float(cfg["delta"]))
        v = self.memory.nearest_centroid(ft["h"].to(device))
        H = self.H_prev if self.H_prev is not None else torch.zeros_like(g0)
        f_cand = frontier_split(g0, kg, k0, cg_kg, v, H, float(cfg["delta"]))
        self.H_prev = f_cand.clone()
        in_frontier = frontier_quota(f_cand, k0, in_core, cg_kg, float(cfg["beta"]))

        LOGGER.info("audit round %d: core=%d (%.1f%%), frontier=%d (%.1f%%), "
                    "raw admission=%.1f%%, b_max=%.3f",
                    round_idx, int(in_core.sum()), 100 * float(in_core.float().mean()),
                    int(in_frontier.sum()), 100 * float(in_frontier.float().mean()),
                    100 * float(g0.float().mean()),
                    float(b_next.abs().max()) if b_next.abs().max() > 0 else 0.0)
        return {"in_core": in_core, "wC": wC, "in_frontier": in_frontier,
                "cg_kg": cg_kg, "t_i": qg, "k0": k0, "kg": kg, "b_next": b_next,
                "q0": q0, "k_cand": k_cand, "c0": c0, "g0": g0}

    # --------------------------------------------------------------- losses
    def _dec_loss(self, f: Dict[str, torch.Tensor], T: torch.Tensor,
                  V: torch.Tensor, C: torch.Tensor, domain: str,
                  n_d: float) -> torch.Tensor:
        B = self.factorization.basis(domain)
        l, _ = L.decomposition_loss(f["z"], f["Pbar"], C, V, f["A"], B, f["E"],
                                    self.V_T, self.cfg, n_d)
        return l

    # ------------------------------------------------------------ training
    def fit(self, class_names: Sequence[str], z_s: torch.Tensor, y_s: torch.Tensor,
            z_t: torch.Tensor, z_t_test: Optional[torch.Tensor] = None,
            y_t_test: Optional[torch.Tensor] = None,
            eval_every: int = 0, log_every: int = 10,
            save_path: Optional[str] = None) -> Dict:
        cfg = self.cfg
        device = self.device
        class_names = list(class_names)
        seeds = cfg["seeds"] if isinstance(cfg["seeds"], list) else [cfg["seeds"]]
        seed = seeds[0] if cfg.get("seed", 0) in seeds else cfg.get("seed", seeds[0])

        T0 = self._text_prototypes(class_names)
        U, S, Rh = torch.linalg.svd(T0, full_matrices=False)
        r_eff = min(int((S >= cfg.get("rank_eps", 1e-3) * S[0]).sum().item()), self.r)
        self.V_T = Rh[:r_eff].t().contiguous().detach()          # (d, r_eff)
        # (d, r) init: text subspace + orthogonal random completion.
        pert = cfg.get("init_perturb", 1e-3)
        extra = torch.randn(self.d, self.r - r_eff, device=device)
        extra = extra - self.V_T @ (self.V_T.t() @ extra)
        self.factorization.Vbar.data = torch.cat([self.V_T, pert * F.normalize(extra, dim=-1)], dim=1)

        z_s = z_s.to(device)
        y_s = y_s.to(device)
        z_t = z_t.to(device)
        iters = cfg["iterations_per_epoch"]
        warmup_epochs = cfg["warmup_epochs"]
        adapt_epochs = cfg["adapt_epochs"]
        rounds = cfg.get("rounds", 5)
        history = {"train_loss": [], "acc_test": []}

        total_epochs = warmup_epochs + adapt_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_epochs, eta_min=1e-6)

        # --- warm-up -----------------------------------------------------
        for epoch in range(warmup_epochs):
            sampler_s = FeatureSampler(z_s, y_s, cfg["batch_size"], iters, seed + epoch)
            sampler_t = FeatureSampler(z_t, None, cfg["batch_size"], iters, seed + epoch)
            for it in range(iters):
                # T is recomputed per iteration (prompt context is trainable);
                # detached when the prompt is frozen (paper: train only V/B).
                T = self._text_prototypes(class_names)
                if not cfg.get("train_prompt", False):
                    T = T.detach()
                V = self.factorization.V()
                C = T @ V
                z_b, y_b, _ = next(sampler_s)
                z_tb, _, _ = next(sampler_t)
                self.optimizer.zero_grad(set_to_none=True)
                fs = self._project(z_b, T, V, C, y_b, "s")
                ft = self._project(z_tb, T, V, C, None, "t")
                l_s = L.source_loss(fs["p"], fs["q0"], y_b)
                l_dec = self._dec_loss(fs, T, V, C, "s", float(z_s.shape[0])) + \
                    self._dec_loss(ft, T, V, C, "t", float(z_t.shape[0]))
                loss = l_s + cfg["lambda_dec"] * l_dec
                loss.backward()
                self.optimizer.step()
                self.step += 1
                if it % log_every == 0:
                    LOGGER.info("warmup epoch %d it %d loss=%.4f", epoch, it, loss.item())
                    history["train_loss"].append(loss.item())
            self.scheduler.step()
            if eval_every > 0 and z_t_test is not None and (epoch + 1) % eval_every == 0:
                history["acc_test"].append(self.evaluate(class_names, z_t_test, y_t_test))

        if save_path is not None:
            self.save(save_path, extra={"stage": "warmup"})
            self.save(os.path.join(os.path.dirname(save_path), "model_warmup.pt"),
                      extra={"stage": "warmup"})

        # --- audit rounds -------------------------------------------------
        round_steps = [iters * adapt_epochs // rounds] * rounds
        round_steps[-1] += iters * adapt_epochs - sum(round_steps)
        for e in range(rounds):
            T = self._text_prototypes(class_names)
            if not cfg.get("train_prompt", False):
                T = T.detach()
            audit = self._audit_round(class_names, T, z_s, y_s, z_t, e, seeds)
            if cfg.get("ramp_lambda_F", True):
                lamF = cfg.get("lambda_F", 1.0) * (e + 1) / rounds     # linear ramp
            else:
                lamF = cfg.get("lambda_F", 1.0)
            sampler_s = FeatureSampler(z_s, y_s, cfg["batch_size"], iters, seed + 1000 + e)
            sampler_t = FeatureSampler(z_t, None, cfg["batch_size"], iters, seed + 2000 + e)
            for it in range(round_steps[e]):
                V = self.factorization.V()
                C = T @ V
                z_b, y_b, _ = next(sampler_s)
                z_tb, _, idx_b = next(sampler_t)
                self.optimizer.zero_grad(set_to_none=True)
                fs = self._project(z_b, T, V, C, y_b, "s")
                ft = self._project(z_tb, T, V, C, None, "t")
                l_s = L.source_loss(fs["p"], fs["q0"], y_b)
                l_dec = self._dec_loss(fs, T, V, C, "s", float(z_s.shape[0])) + \
                    self._dec_loss(ft, T, V, C, "t", float(z_t.shape[0]))
                in_core_b = audit["in_core"][idx_b]
                in_f_b = audit["in_frontier"][idx_b]
                phat = self._phat(z_tb, idx_b, V, C)
                l_C = L.core_loss(phat, in_core_b, audit["wC"][idx_b],
                                  audit["k0"][idx_b])
                l_F = L.frontier_loss(phat, in_f_b, audit["cg_kg"][idx_b],
                                      audit["t_i"][idx_b])
                loss = (l_s + cfg["lambda_dec"] * l_dec
                        + cfg["lambda_C"] * l_C + lamF * l_F)
                loss.backward()
                self.optimizer.step()
                self.step += 1
                if it % log_every == 0:
                    LOGGER.info("round %d it %d loss=%.4f (l_C=%.4f l_F=%.4f lamF=%.3f)",
                                e, it, loss.item(), l_C.item(), l_F.item(), lamF)
                    history["train_loss"].append(loss.item())
            self.scheduler.step()
            if eval_every > 0 and z_t_test is not None and (e + 1) % eval_every == 0:
                history["acc_test"].append(self.evaluate(class_names, z_t_test, y_t_test))
            if save_path is not None:
                self.save(save_path, extra={"stage": "round", "round": e})
        return history

    # --------------------------------------------------------------- evaluate
    @torch.no_grad()
    def evaluate(self, class_names: Sequence[str],
                 z_test: torch.Tensor, y_test: Optional[torch.Tensor]) -> float:
        """Inference: argmax p_theta(f_theta(x)); no donor transport or offset."""
        T = self._text_prototypes(class_names)
        V = self.factorization.V()
        C = T @ V
        accs = []
        z_test = z_test.to(self.device)
        for i in range(0, z_test.shape[0], 256):
            zb = z_test[i:i + 256]
            p = self._p(zb, V, C)
            pred = p.argmax(dim=1)
            if y_test is not None:
                accs.append(float((pred == y_test[i:i + 256].to(self.device)).float().mean().item()))
        return float(np.mean(accs)) if accs else float("nan")
