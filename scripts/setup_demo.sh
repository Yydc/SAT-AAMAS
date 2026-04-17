#!/usr/bin/env bash
# One-shot installation + demo run.
#
# Creates a clean Python environment, installs the SAT package, generates
# a tiny synthetic dataset, runs two short SAT stages on three built-in
# sat:tiny agents, and evaluates the team. No model checkpoints are
# downloaded for this demo path.
#
# Usage:
#     bash scripts/setup_demo.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"

echo "==> Installing dependencies"
if [[ "${SAT_SKIP_INSTALL:-0}" == "1" ]]; then
  echo "==> SAT_SKIP_INSTALL=1, using the current Python environment"
else
  $PYTHON -m pip install -r requirements.txt
  $PYTHON -m pip install -e .
fi

echo "==> Preparing synthetic demo dataset"
$PYTHON scripts/prepare_data.py --dataset demo

echo "==> Training (2 stages, 3x local sat:tiny agents)"
$PYTHON scripts/train.py --config configs/sat_demo.yaml

echo "==> Evaluating"
$PYTHON scripts/evaluate.py --config configs/sat_demo.yaml --ckpt_dir outputs/sat_demo

echo "==> Done. Predictions in outputs/sat_demo/predictions.jsonl"
