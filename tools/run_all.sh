#!/usr/bin/env bash
# Run all DIE experiments end-to-end (benchmarks + ablation + label shift).
# Assumes: python3, data prepared (see docs/REPRODUCE.md).
set -euo pipefail
cd "$(dirname "$0")/.."

PY=${PY:-python3}
CFG_DIR=configs/die

run_cfg() {
  local cfg="$1"
  echo "===== $cfg ====="
  "$PY" main.py benchmark --config "$cfg"
}

# 1. Main benchmarks
run_cfg "$CFG_DIR/office_home.yaml"
run_cfg "$CFG_DIR/visda.yaml"
run_cfg "$CFG_DIR/domainnet.yaml"

# 2. Ablation table (Table ablation) on Office-Home Art -> Clipart
for f in "$CFG_DIR"/ablation/*.yaml; do
  run_cfg "$f"
done

# 3. Aggregate all results into outputs/results_summary.json
"$PY" tools/analyze_results.py --root outputs
echo "All experiments finished."
