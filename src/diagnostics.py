"""Post-hoc diagnostics for DIE.

All diagnostics load a frozen checkpoint and operate on cached features; none
retrains.  They produce the material for the paper's placeholder tables and
figures: per-class admission vs accuracy (Fig. admission), audit-round
statistics (Fig. curves), and the label-shift setup (Table shift).
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from src.audit import audit_batch, candidate_classes
from src.model import SourceMemory
from src.trainer import DIETrainer
from src.utils import LOGGER, set_seed


def load_trainer(cfg: dict, device: torch.device, seed: int = 0) -> DIETrainer:
    from data import DataModule
    from src.clip_backend import build_clip_backend
    backend = build_clip_backend(cfg, device)
    dm = DataModule(cfg, backend, device)
    src, tgt = cfg["transfers"][0]
    data = dm.prepare(src, tgt, inductive=cfg.get("inductive", False))
    trainer = DIETrainer(cfg, backend, device, strong_aug_fn=data.get("strong_aug_fn"))
    ckpt = os.path.join(cfg["output_dir"], f"{src}->{tgt}", f"seed{seed}", "model.pt")
    trainer.load(ckpt, load_optimizer=False)
    return trainer, data, backend, (src, tgt)


def admission_accuracy_diagnostic(cfg: dict, device: torch.device) -> Dict:
    """Per-class raw-admission rate vs per-class accuracy (Fig. admission)."""
    trainer, data, backend, (src, tgt) = load_trainer(cfg, device, seed=cfg["seeds"][0])
    T = trainer._text_prototypes(data["class_names"]).to(device)
    z_s = data["z_s"].to(device)
    y_s = data["y_s"].to(device)
    z_t = data["z_t"].to(device)
    y_t = data["y_t"].to(device)

    fs = trainer.decompose_all(z_s, T, y_s, "s")
    ft = trainer.decompose_all(z_t, T, None, "t")
    memory = SourceMemory(cfg["num_classes"], cfg.get("donor_capacity", 4096))
    memory.push(fs["r"].to(device), y_s.to(device))
    memory.update_centroids(fs["h"].to(device), y_s.to(device))

    q0 = trainer._q0(z_t, T)
    k0 = q0.argmax(dim=1)
    k_cand = candidate_classes(q0, cfg.get("n_candidates", 10), device)
    a0 = audit_batch(z_t, ft["s"].to(device), ft["r"].to(device), T, trainer.tau,
                     memory, k_cand, cfg, torch.zeros(cfg["num_classes"], device=device),
                     0.0, seed=7)
    c0 = a0["c"].gather(1, (k_cand == k0.unsqueeze(1)).nonzero(as_tuple=True)[1].unsqueeze(1)).squeeze(1)
    admitted = c0 >= float(cfg["delta"])

    per_class = []
    for k in range(cfg["num_classes"]):
        idx = (k0 == k).nonzero(as_tuple=True)[0]
        if idx.numel() == 0:
            continue
        acc = float((trainer.evaluate(data["class_names"],
                                      z_t[idx], y_t[idx]) / 100.0))
        adm = float(admitted[idx].float().mean().item())
        n = int(idx.numel())
        per_class.append({"class": k, "n_targets": n, "admission_rate": adm,
                          "accuracy": acc})
    corr = float(np.corrcoef([p["admission_rate"] for p in per_class],
                             [p["accuracy"] for p in per_class])[0, 1])
    return {"transfer": f"{src}->{tgt}", "seed": cfg["seeds"][0],
            "per_class": per_class, "admission_accuracy_corr": corr}


def round_statistics_diagnostic(cfg: dict, device: torch.device) -> Dict:
    """Replay the audit rounds from a trained model and report per-round
    core/frontier sizes and raw-admission rates (material for Fig. curves)."""
    trainer, data, backend, (src, tgt) = load_trainer(cfg, device, seed=cfg["seeds"][0])
    T = trainer._text_prototypes(data["class_names"]).to(device)
    z_s = data["z_s"].to(device)
    y_s = data["y_s"].to(device)
    z_t = data["z_t"].to(device)
    trainer.memory = SourceMemory(cfg["num_classes"], cfg.get("donor_capacity", 4096))
    trainer.interv = trainer.interv.__class__(
        cfg["num_classes"], alpha_smooth=cfg.get("alpha_smooth", 1.0),
        b_max=cfg.get("b_max", 2.0), n_min=cfg.get("n_min", 10),
        ema=cfg.get("ema_stats", 0.9), device=device,
        offset_stat=cfg.get("offset_stat", "admission"))
    trainer.H_prev = None
    rows = []
    for e in range(cfg.get("rounds", 5)):
        aud = trainer._audit_round(data["class_names"], T, z_s, y_s, z_t, e, cfg["seeds"])
        rows.append({"round": e,
                     "core_frac": float(aud["in_core"].float().mean().item()),
                     "frontier_frac": float(aud["in_frontier"].float().mean().item()),
                     "raw_admission": float(aud["g0"].float().mean().item()),
                     "b_max_abs": float(aud["b_next"].abs().max().item())})
    return {"transfer": f"{src}->{tgt}", "seed": cfg["seeds"][0], "rounds": rows}


def dirichlet_subset(cfg: dict, device: torch.device, alpha: float = 1.0) -> Dict:
    """Sample a target subset under Dirichlet(alpha) class weights (Table shift).

    Only the index mask is produced here; train the compared methods on the
    same mask with ``--set label_shift_alpha=...`` (the mask is seeded so that
    the confidence / frequency-offset / DIE runs share identical subsets).
    """
    trainer, data, backend, (src, tgt) = load_trainer(cfg, device, seed=cfg["seeds"][0])
    y_t = data["y_t"].numpy()
    K = cfg["num_classes"]
    rng = np.random.default_rng(cfg.get("label_shift_seed", 0))
    w = rng.dirichlet(np.full(K, alpha))
    keep = []
    for k in range(K):
        idx = np.where(y_t == k)[0]
        if idx.size == 0:
            continue  # class absent from the target split
        n_k = max(1, int(round(w[k] * idx.size)))
        n_k = min(n_k, idx.size)
        keep.append(rng.choice(idx, size=n_k, replace=False))
    keep = np.concatenate(keep)
    counts = {int(k): int((y_t[keep] == k).sum()) for k in range(K)}
    return {"transfer": f"{src}->{tgt}", "alpha": alpha, "n_selected": int(keep.size),
            "n_total": int(y_t.size), "per_class_counts": counts,
            "selection_seed": cfg.get("label_shift_seed", 0)}
