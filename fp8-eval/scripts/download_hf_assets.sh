#!/usr/bin/env bash
# Download gated HF assets required for FP8 gate eval.
set -euo pipefail

YOUR_HOME="${YOUR_HOME:-/home/shadeform}"
source "$YOUR_HOME/alpamayo-recipes/recipes/alpamayo1_5_quant/run_env.sh"

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: HF_TOKEN is not set."
  echo "Request access and export your token:"
  echo "  https://huggingface.co/nvidia/Cosmos-Reason2-8B"
  echo "  https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles"
  echo "  export HF_TOKEN=hf_..."
  echo "  hf auth login --token \"\$HF_TOKEN\""
  exit 1
fi

hf auth login --token "$HF_TOKEN"

MODEL_LOCAL="$YOUR_HOME/alpamayo_model_hf"
if [[ ! -f "$MODEL_LOCAL/config.json" ]]; then
  echo "=== Download Alpamayo-1.5-10B ==="
  hf download nvidia/Alpamayo-1.5-10B --local-dir "$MODEL_LOCAL"
fi

echo "=== Download Cosmos-Reason2-8B processor/tokenizer (model init) ==="
hf download nvidia/Cosmos-Reason2-8B \
  config.json tokenizer.json tokenizer_config.json \
  preprocessor_config.json processor_config.json \
  special_tokens_map.json vocab.json merges.txt 2>/dev/null || \
  hf download nvidia/Cosmos-Reason2-8B

echo "=== Download Qwen3-VL-2B-Instruct processor (eval inference) ==="
hf download Qwen/Qwen3-VL-2B-Instruct \
  config.json processor_config.json preprocessor_config.json \
  tokenizer.json tokenizer_config.json vocab.json merges.txt

echo "=== Verify PhysicalAI-AV dataset access ==="
python - <<'PY'
import physical_ai_av
avdi = physical_ai_av.PhysicalAIAVDatasetInterface()
print("PhysicalAI-AV dataset interface OK")
PY

echo "=== HF assets ready ==="
