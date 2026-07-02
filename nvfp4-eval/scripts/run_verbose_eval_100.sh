#!/usr/bin/env bash
# Run 100-clip eval with --print_every 1 and export per-clip CSVs.
set -euo pipefail

YOUR_HOME="${YOUR_HOME:-/home/shadeform}"
QUANT_DIR="$YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_quant"
ARTIFACT_DIR="$YOUR_HOME/alpamayo-e2e-challenge/nvfp4-eval"
LOG_DIR="$ARTIFACT_DIR/logs"

source "$QUANT_DIR/run_env.sh"
mkdir -p "$LOG_DIR" "$ARTIFACT_DIR/results"

BF16_LOG="$LOG_DIR/bf16_verbose_100.log"
NVFP4_LOG="$LOG_DIR/nvfp4_verbose_100.log"

echo "=== bf16 baseline 100 clips (print_every=1) ==="
python eval.py --limit 100 --print_every 1 2>&1 | tee "$BF16_LOG"

python "$ARTIFACT_DIR/scripts/parse_eval_log.py" "$BF16_LOG" \
  -o "$ARTIFACT_DIR/results/per_clip_bf16_100.csv"

echo "=== NVFP4 100 clips (print_every=1) ==="
python eval.py --ckpt ./outputs/alpamayo1.5_nvfp4_calib100 \
  --limit 100 --print_every 1 2>&1 | tee "$NVFP4_LOG"

python "$ARTIFACT_DIR/scripts/parse_eval_log.py" "$NVFP4_LOG" \
  -o "$ARTIFACT_DIR/results/per_clip_nvfp4_100.csv"

python "$ARTIFACT_DIR/scripts/merge_per_clip.py"

echo "=== DONE ==="
