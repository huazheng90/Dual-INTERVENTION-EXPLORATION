"""Benchmark driver for DIE (mean accuracy over tasks, std over seeds)."""
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from src.trainer import DIETrainer
from src.utils import LOGGER, set_seed


def _mean_acc(accs: List[float]) -> float:
    return float(np.mean(accs))


def run_transfer(cfg: dict, backend, data: Dict, device: torch.device,
                 src: str, tgt: str) -> float:
    """Train one transfer; returns final target accuracy."""
    strong_aug_fn = data.get("strong_aug_fn")
    trainer = DIETrainer(cfg, backend, device, strong_aug_fn=strong_aug_fn)
    ckpt = os.path.join(cfg["output_dir"], f"{src}->{tgt}",
                        f"seed{cfg['seed']}", "model.pt")
    if os.path.exists(ckpt) and cfg.get("resume", False):
        trainer.load(ckpt)
        LOGGER.info("resumed %s", ckpt)
    trainer.fit(
        data["class_names"], data["z_s"], data["y_s"], data["z_t"],
        z_t_test=data.get("z_t_test"), y_t_test=data.get("y_t_test"),
        eval_every=0, log_every=cfg.get("log_every", 100),
        save_path=ckpt)
    acc = trainer.evaluate(data["class_names"], data["z_t_test"] or data["z_t"],
                           data["y_t_test"] or data["y_t"])
    LOGGER.info("[%s->%s] seed %d acc=%.4f", src, tgt, cfg["seed"], acc)
    return acc


def run_benchmark(cfg: dict, device: torch.device) -> Dict:
    """All transfers x seeds; archives per-transfer/per-seed JSON + summary."""
    from data import DataModule
    from src.clip_backend import build_clip_backend

    backend = build_clip_backend(cfg, device)
    dm = DataModule(cfg, backend, device)
    transfers = cfg["transfers"]
    seeds = cfg["seeds"] if isinstance(cfg["seeds"], list) else [cfg["seeds"]]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    per_transfer: Dict[str, List[float]] = {}
    per_seed_scores = []
    for src, tgt in transfers:
        data = dm.prepare(src, tgt, inductive=cfg.get("inductive", False))
        accs = []
        for seed in seeds:
            cfg["seed"] = seed
            set_seed(seed)
            acc = run_transfer(cfg, backend, data, device, src, tgt)
            accs.append(acc)
            rec = os.path.join(out_dir, f"{src}->{tgt}", f"seed{seed}")
            os.makedirs(rec, exist_ok=True)
            with open(os.path.join(rec, "results.json"), "w", encoding="utf-8") as f:
                json.dump({"src": src, "tgt": tgt, "seed": seed, "acc": acc}, f, indent=2)
        per_transfer[f"{src}->{tgt}"] = accs
        LOGGER.info("[%s->%s] mean=%.4f std=%.4f", src, tgt,
                    _mean_acc(accs), float(np.std(accs, ddof=1)))
        for i, seed in enumerate(seeds):
            if len(per_seed_scores) <= i:
                per_seed_scores.append([])
            per_seed_scores[i].append(accs[i])

    bench = [float(np.mean(v)) for v in per_seed_scores]
    summary = {"benchmark": cfg.get("benchmark", "unknown"),
               "transfers": list(per_transfer.keys()),
               "per_transfer": {k: {"mean": _mean_acc(v), "std": float(np.std(v, ddof=1))}
                                for k, v in per_transfer.items()},
               "per_seed_benchmark_scores": bench,
               "mean": _mean_acc(bench),
               "std": float(np.std(bench, ddof=1)),
               "seeds": seeds}
    with open(os.path.join(out_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\n=== benchmark summary ===")
    print(f"{summary['benchmark']}: {summary['mean']:.2f} +/- {summary['std']:.2f} "
          f"(seeds {seeds})")
    return summary
