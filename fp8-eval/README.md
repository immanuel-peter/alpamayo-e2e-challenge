# FP8 gate evaluation (H100)

Per `PLAN.md` Phase 0 and `JOURNAL.md` next steps: quantize Alpamayo 1.5-10B to
**FP8** (Hopper-native deployment format), then eval 100 PAI gold clips and compare
minADE vs bf16 baseline.

## Prerequisites

- 1× H100 (this box)
- HF access: `nvidia/Alpamayo-1.5-10B`, `nvidia/Cosmos-Reason2-8B`, `nvidia/PhysicalAI-Autonomous-Vehicles`
- `HF_TOKEN` exported (e.g. in `~/.bashrc`); `run_env.sh` loads it automatically

## Scripts

| Script | Purpose |
|--------|---------|
| `scripts/download_hf_assets.sh` | Auth + download model/processor; verify PAI dataset |
| `scripts/run_fp8_gate.sh` | Full gate: FP8 quant (100 calib) → bf16 eval (100) → FP8 eval (100) |

## Run

```bash
# one-time asset check
./fp8-eval/scripts/download_hf_assets.sh

# full gate (hours)
nohup ./fp8-eval/scripts/run_fp8_gate.sh > fp8-eval/logs/run_fp8_gate.nohup.log 2>&1 &
```

## Logs & outputs

- `logs/quantize_fp8_calib100.log`
- `logs/bf16_eval_100.log`
- `logs/fp8_eval_100.log`
- Quantized checkpoint: `alpamayo-recipes/recipes/alpamayo1_5_quant/outputs/alpamayo1.5_fp8_calib100`

## Gate criteria

- FP8 mean minADE **< 1.0 m** on 100 held-out eval clips (same parquet as NVFP4 gate)
- Record avg time/clip on H100 (real FP8 path vs B300 fake-quant)

NVFP4 reference (100 clips, B300): bf16 **0.817 m**, NVFP4 **0.848 m**.
