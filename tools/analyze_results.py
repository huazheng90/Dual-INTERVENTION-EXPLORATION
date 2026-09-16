#!/usr/bin/env python3
"""Aggregate per-transfer results into a compact Markdown/JSON summary.

Usage:
    python3 tools/analyze_results.py --root outputs [--md results.md]

Scans every ``results.json`` written by main.py benchmark/run and produces
a per-experiment table of mean +/- std accuracy over seeds.
"""
from __future__ import annotations

import argparse
import json
import os


def collect(root: str) -> dict:
    table = {}
    for dirpath, _, files in os.walk(root):
        if "results.json" not in files:
            continue
        path = os.path.join(dirpath, "results.json")
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            print(f"[skip] {path}: {exc}")
            continue
        # benchmark summary
        if "benchmark" in data and "per_transfer" in data:
            name = data["benchmark"]
            table.setdefault(name, {})["overall"] = {
                "mean": data["mean"], "std": data["std"],
                "per_seed": data.get("per_seed_benchmark_scores"),
            }
            for k, v in data["per_transfer"].items():
                table[name][k] = v
        # single transfer summary (ablation runs produce one of these too)
        elif "mean" in data and "per_transfer" in data:
            name = os.path.basename(os.path.dirname(dirpath))
            table.setdefault(name, {})["overall"] = {
                "mean": data["mean"], "std": data["std"],
                "per_seed": data.get("per_seed_benchmark_scores"),
            }
            for k, v in data["per_transfer"].items():
                table[name][k] = v
    return table


def to_markdown(table: dict) -> str:
    lines = ["# DIE results summary", ""]
    for name in sorted(table):
        rows = table[name]
        lines.append(f"## {name}")
        lines.append("")
        lines.append("| task | mean | std |")
        lines.append("|---|---|---|")
        for task in sorted(rows):
            r = rows[task]
            if isinstance(r, dict) and "mean" in r:
                lines.append(f"| {task} | {r['mean']:.2f} | {r['std']:.2f} |")
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--md", default=None)
    args = ap.parse_args()
    table = collect(args.root)
    if not table:
        print("no results.json found under", args.root)
        return 1
    md = to_markdown(table)
    print(md)
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"\n[written] {args.md}")
    with open(os.path.join(args.root, "results_summary.json"), "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
