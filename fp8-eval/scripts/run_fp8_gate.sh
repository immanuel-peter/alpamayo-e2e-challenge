#!/usr/bin/env bash
# FP8 gate: quantize (100 calib clips) + eval (100 clips) on H100.
set -euo pipefail

YOUR_HOME="${YOUR_HOME:-/home/shadeform}"
QUANT_DIR="$YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_quant"
ARTIFACT_DIR="$YOUR_HOME/alpamayo-e2e-challenge/fp8-eval"
LOG_DIR="$ARTIFACT_DIR/logs"
OUTPUT_DIR="$QUANT_DIR/outputs"

source "$QUANT_DIR/run_env.sh"
mkdir -p "$LOG_DIR" "$ARTIFACT_DIR/results" "$OUTPUT_DIR"

CALIB_LOG="$LOG_DIR/quantize_fp8_calib100.log"
BF16_LOG="$LOG_DIR/bf16_eval_100.log"
FP8_LOG="$LOG_DIR/fp8_eval_100.log"

echo "=== Step 1: FP8 quantization (100 calib clips) ==="
python quantize.py \
  --ckpt "$ALPAMAYO_MODEL_CKPT" \
  --quant_format=fp8 \
  --num_of_calib_clips=100 \
  --save_model_dir="$OUTPUT_DIR" \
  2>&1 | tee "$CALIB_LOG"

echo "=== Step 2: bf16 baseline eval (100 clips) ==="
python eval.py --ckpt "$ALPAMAYO_MODEL_CKPT" --limit 100 --print_every 25 2>&1 | tee "$BF16_LOG"

echo "=== Step 3: FP8 eval (100 clips) ==="
python eval.py \
  --ckpt "$OUTPUT_DIR/alpamayo1.5_fp8_calib100" \
  --limit 100 \
  --print_every 25 \
  2>&1 | tee "$FP8_LOG"

echo "=== DONE ==="
echo "Logs: $LOG_DIR"
echo "Quantized checkpoint: $OUTPUT_DIR/alpamayo1.5_fp8_calib100"
