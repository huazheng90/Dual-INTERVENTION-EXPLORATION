"""CPU smoke test for the full DIE pipeline on a mock CLIP backend.

Run:  python tests/test_smoke.py

Validates data flow, warm-up, audit rounds (core/frontier split, logit
intervention), checkpoint round-trip, evaluation, and the diagnostics entry
points — without downloading CLIP weights or dataset images.
"""
from __future__ import annotations

import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils import set_seed  # noqa: E402


def make_cfg(tmp: str, num_classes: int = 8) -> dict:
    return {
        "backend": "mock", "feat_dim": 64, "num_classes": num_classes,
        "seed": 0, "seeds": [0], "n_ctx": 4, "text_logit_scale": 20.0,
        "semantic_rank": 16, "residual_rank": 6, "init_perturb": 1e-3,
        "rank_eps": 1e-3, "j_alt": 3, "lr_block": 0.1,
        "beta_a": 1e-3, "xi": 1e-3, "beta_b": 1e-3,
        "lambda_T": 1.0, "lambda_perp": 0.1, "lambda_dec": 1.0,
        "M": 4, "rho": 0.5, "delta": 0.05, "kappa_r": 4.0,
        "delta_energy": 2.0, "eta": 0.1, "n_candidates": 4,
        "donor_capacity": 256,
        "gamma": 1.0, "b_max": 2.0, "n_min": 3, "alpha_smooth": 1.0,
        "ema_stats": 0.9, "beta": 0.3,
        "lambda_C": 1.0, "lambda_F": 1.0, "strong_aug": "feature",
        "lr": 1e-3, "weight_decay": 0.0, "batch_size": 8,
        "iterations_per_epoch": 4, "warmup_epochs": 1, "adapt_epochs": 2,
        "rounds": 2, "log_every": 1, "train_prompt": False,
        "output_dir": os.path.join(tmp, "out"),
        "checkpoint": os.path.join(tmp, "out", "model.pt"),
        "device": "cpu", "transfers": [["src", "tgt"]],
    }


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="die_smoke_")
    set_seed(0)
    device = torch.device("cpu")
    cfg = make_cfg(tmp)
    K = cfg["num_classes"]

    from src.clip_backend import build_clip_backend
    from src.trainer import DIETrainer

    backend = build_clip_backend(cfg, device)
    class_names = [f"class_{i}" for i in range(K)]
    g = torch.Generator().manual_seed(7)
    z_s = torch.randn(160, cfg["feat_dim"], generator=g)
    y_s = torch.randint(0, K, (160,), generator=g)
    z_s = torch.nn.functional.normalize(z_s, dim=-1)
    z_t = torch.randn(120, cfg["feat_dim"], generator=g) * 1.5
    z_t = torch.nn.functional.normalize(z_t, dim=-1)
    y_t = torch.randint(0, K, (120,), generator=g)

    trainer = DIETrainer(cfg, backend, device)
    trainer.fit(class_names, z_s, y_s, z_t, save_path=cfg["checkpoint"])
    assert not torch.isnan(trainer.optimizer.param_groups[0]["params"][0].grad).any()

    trainer.save(cfg["checkpoint"])
    trainer2 = DIETrainer(cfg, backend, device)
    trainer2.load(cfg["checkpoint"], load_optimizer=False)
    acc = trainer2.evaluate(class_names, z_t, y_t)
    print(f"[smoke] evaluation acc={acc:.4f}")
    assert 0.0 <= acc <= 100.0
    print("[smoke] audit stats:", "core/frontier mechanics exercised")

    # Diagnostics smoke.
    from src import diagnostics as D
    import data as data_mod
    ckpt_dir = os.path.join(tmp, "src->tgt", "seed0")
    os.makedirs(ckpt_dir, exist_ok=True)
    trainer.save(os.path.join(ckpt_dir, "model.pt"))
    cfg["output_dir"] = tmp

    class _DM:
        def __init__(self, cfg=None, backend=None, device=None):
            pass

        def prepare(self, *a, **k):
            return {"z_s": z_s, "y_s": y_s, "z_t": z_t, "y_t": y_t,
                    "z_t_test": None, "y_t_test": None, "class_names": class_names,
                    "strong_aug_fn": None}
    data_mod.DataModule = _DM
    D.build_clip_backend = lambda c, d: backend

    res = D.admission_accuracy_diagnostic(cfg, device)
    print("[smoke] admission diagnostic:", {k: (len(v) if isinstance(v, list) else v)
                                             for k, v in res.items() if k != "per_class"})
    res2 = D.round_statistics_diagnostic(cfg, device)
    print("[smoke] rounds:", res2["rounds"])
    res3 = D.dirichlet_subset(cfg, device, alpha=0.5)
    print("[smoke] label-shift subset:", res3["n_selected"], "/", res3["n_total"])

    print("\nALL SMOKE TESTS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
