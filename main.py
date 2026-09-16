"""DIE entry point.

Commands
--------
python main.py benchmark --config configs/die/office_home.yaml
    Run every directed transfer x every seed and archive results.

python main.py run --config ... --src Art --tgt Clipart
    Single transfer over the configured seeds.

python main.py diagnose --config ... --kind admission|rounds|label_shift
    Post-hoc mechanism analysis on saved checkpoints.

python main.py count_params --config ...
python main.py extract --config ...
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import torch
import yaml

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils import LOGGER, setup_logging  # noqa: E402


def load_config(path: str, overrides: Optional[List[str]] = None) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for kv in overrides:
            key, _, val = kv.partition("=")
            try:
                cfg[key] = yaml.safe_load(val)
            except Exception:
                cfg[key] = val
    if isinstance(cfg.get("seeds"), int):
        cfg["seeds"] = [cfg["seeds"]]
    if isinstance(cfg.get("transfers"), (list, tuple)) and \
            isinstance(cfg["transfers"][0], str):
        cfg["transfers"] = [cfg["transfers"]]
    cfg.setdefault("output_dir", os.path.join("outputs", cfg.get("experiment_name", "die")))
    return cfg


def _device(cfg: Dict) -> torch.device:
    want = cfg.get("device", "gpu")
    if want == "cpu":
        return torch.device("cpu")
    if want == "gpu":
        if torch.cuda.is_available():
            return torch.device("cuda")
        LOGGER.warning("device: gpu requested but CUDA unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cmd_benchmark(cfg: Dict, args) -> int:
    from src.eval import run_benchmark
    run_benchmark(cfg, _device(cfg))
    return 0


def cmd_run(cfg: Dict, args) -> int:
    from data import DataModule
    from src.clip_backend import build_clip_backend
    from src.eval import run_transfer
    from src.utils import set_seed
    device = _device(cfg)
    backend = build_clip_backend(cfg, device)
    dm = DataModule(cfg, backend, device)
    data = dm.prepare(args.src, args.tgt, inductive=cfg.get("inductive", False))
    for seed in cfg["seeds"]:
        cfg["seed"] = seed
        set_seed(seed)
        run_transfer(cfg, backend, data, device, args.src, args.tgt)
    return 0


def cmd_diagnose(cfg: Dict, args) -> int:
    from src import diagnostics as D
    device = _device(cfg)
    kinds = args.kind.split(",")
    out_dir = os.path.join(cfg["output_dir"], "diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    for kind in kinds:
        t0 = time.time()
        if kind == "admission":
            res = D.admission_accuracy_diagnostic(cfg, device)
        elif kind == "rounds":
            res = D.round_statistics_diagnostic(cfg, device)
        elif kind == "label_shift":
            res = D.dirichlet_subset(cfg, device, alpha=args.alpha)
        else:
            LOGGER.error("unknown diagnostic kind: %s", kind)
            return 1
        path = os.path.join(out_dir, f"{kind}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2, ensure_ascii=False)
        LOGGER.info("%s -> %s (%.1fs)", kind, path, time.time() - t0)
        print(yaml.safe_dump(res, sort_keys=False))
    return 0


def cmd_count_params(cfg: Dict, args) -> int:
    from src.clip_backend import build_clip_backend
    from src.trainer import DIETrainer
    trainer = DIETrainer(cfg, build_clip_backend(cfg, _device(cfg)), _device(cfg))
    n = sum(p.numel() for p in trainer.opt_params)
    print(f"trainable parameters: {n / 1e6:.4f} M")
    return 0


def cmd_extract(cfg: Dict, args) -> int:
    from data import DataModule
    from src.clip_backend import build_clip_backend
    device = _device(cfg)
    backend = build_clip_backend(cfg, device)
    dm = DataModule(cfg, backend, device)
    for src, tgt in cfg["transfers"]:
        dm.prepare(src, tgt, inductive=cfg.get("inductive", False))
    print("feature cache ready")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="DIE: Dual-Intervention Exploration")
    sub = p.add_subparsers(dest="command", required=True)

    for name, fn in [("benchmark", cmd_benchmark), ("run", cmd_run),
                     ("diagnose", cmd_diagnose), ("count_params", cmd_count_params),
                     ("extract", cmd_extract)]:
        sp = sub.add_parser(name)
        sp.add_argument("--config", required=True)
        sp.add_argument("--set", action="append", default=None,
                        help="override config key=value")
        if name == "run":
            sp.add_argument("--src", required=True)
            sp.add_argument("--tgt", required=True)
        if name == "diagnose":
            sp.add_argument("--kind", required=True,
                            help="admission,rounds,label_shift")
            sp.add_argument("--alpha", type=float, default=1.0)
        sp.set_defaults(func=fn)

    args = p.parse_args(argv)
    cfg = load_config(args.config, args.set)
    setup_logging(cfg.get("log_level", "INFO"),
                  os.path.join(cfg["output_dir"], "train.log"))
    LOGGER.info("config: %s", args.config)
    return args.func(cfg, args)


if __name__ == "__main__":
    raise SystemExit(main())
