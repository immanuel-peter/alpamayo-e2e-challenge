# Journal

---

## 2026-07-01 — Phase 0 decision gate: NVFP4 perception quality test

### Goal

Answer the question: **does NVFP4-quantized Alpamayo 1.5-10B retain enough
perception quality to skip distillation and run RL directly on the quantized
10B?**

If quantized minADE < 1.0m and trajectories look reasonable, the entire
distillation pipeline (Phases 1-2, ~$220, ~36+ hours) can be skipped.

### Key findings from research

1. **No "INT4" exists** — the PLAN's "INT4" maps to NVFP4 (NVIDIA 4-bit
   float) or AutoQuant (~4.8-6.5 effective bits). Supported formats:
   - `fp8` — ~11 GB, 2x compression (H100-compatible)
   - `nvfp4` — ~5.5 GB, ~4x compression (Blackwell-only)
   - `auto` — mixed NVFP4 + FP8, ~9 GB @ 6.5 bits (Blackwell-only)
   - `w4a8_nvfp4_fp8` — 4-bit weights, 8-bit activations (Blackwell-only)

2. **NVFP4 requires Blackwell**, not Hopper. H100 only supports FP8. The
   PLAN's budget of "Brev 1xH100 $2.28/hr" for the INT4 gate is wrong.

3. **Run with the quant recipe directly**, not AlpaSim or Alpagym:
   - `alpamayo-recipes/recipes/alpamayo1_5_quant/` has `quantize.py` and
     `eval.py`
   - `eval.py` runs standalone open-loop inference
     (`sample_trajectories_from_data_with_vlm_rollout`) on PAI dataset clips
     and reports avg minADE — exactly the gate metric
   - No simulator, no RL framework, no gRPC needed

4. **RTX 5090 on Brev at $0.78/hr** is the best option — 3x cheaper than
   H100, supports all quantization formats, 32GB VRAM (tight but workable
   under `no_grad` + `autocast`), 900GB disk.

### Plan

#### Step 1: Provision RTX 5090 on Brev

```bash
brev create alpamayo-quant-gate --type excesssupply_RTX5090
```

- 1x RTX 5090, 32GB VRAM, $0.78/hr, 900GB disk, 2min boot
- Estimated runtime: 4-8 hours → ~$3-6

#### Step 2: Environment setup on the instance

The quant recipe's `pyproject.toml` declares three git-sourced Python
dependencies that `uv sync` will pull automatically:

| Package | Source | What it provides |
|---------|--------|-----------------|
| `alpamayo1_5` | `git+https://github.com/NVlabs/alpamayo1.5.git` | Model class, inference, helper, dataset loader |
| `alpamayo_r1` | `git+https://github.com/NVlabs/alpamayo.git` | `common.logging` used by quantize.py / eval.py |
| `alpamayo-recipes` | `../../src` (editable, same repo) | Shared utilities, chat template, metrics |

So only the `alpamayo-recipes` repo needs to be cloned — `uv sync` fetches
the other two via git.

```bash
git clone https://github.com/NVlabs/alpamayo-recipes.git
cd alpamayo-recipes/recipes/alpamayo1_5_quant

uv venv am15_quant
source am15_quant/bin/activate
MAX_JOBS=4 uv sync --active --no-install-package flash-attn
MAX_JOBS=4 uv sync --active

hf auth login   # need access to gated model + dataset
```

HF cache contents needed:

| HF repo | Type | Gated? | Why |
|---------|------|--------|-----|
| `nvidia/Alpamayo-1.5-10B` | Model | **Yes** | The model being quantized/evaluated. Self-contained — does NOT pull Cosmos-Reason2-8B. |
| `Qwen/Qwen3-VL-2B-Instruct` | Processor | No | VLM backbone processor. Auto-downloaded by `helper.get_processor()`. |
| `nvidia/PhysicalAI-Autonomous-Vehicles` | Dataset | **Yes** | Calibration + eval clips loaded by `load_physical_aiavdataset()`. |

> **Cosmos-Reason2-8B is NOT needed for the quant recipe.** The PLAN's disk
> budget lists it (~16 GB) but that's for the AlpaSim/Alpagym path. The
> released `Alpamayo-1.5-10B` checkpoint is self-contained — its VLM backbone
> is Qwen3-VL-2B-Instruct, loaded via the processor, not a separate Cosmos
> model download.

Env vars:
```bash
export YOUR_HOME="$HOME"
export ALPAMAYO_WORKSPACE="$YOUR_HOME/alpamayo-recipes"
export ALPAMAYO_MODEL_DIR="$YOUR_HOME/alpamayo_model_converted_from_hf"
export ALPAMAYO_PAI_LOCAL_DIR="$YOUR_HOME/PAI_mini"
export ALPAMAYO_LOG_DIR="$YOUR_HOME/alpamayo_logs"
export HF_HOME="$YOUR_HOME/.cache/huggingface"
```

> The `ALPAMAYO_MODEL_DIR` checkpoint conversion step (`scripts/convert_checkpoint.py to-a1`)
> is for the SFT/RL recipes, not the quant recipe. The quant recipe loads
> directly from `nvidia/Alpamayo-1.5-10B` via `Alpamayo1_5.from_pretrained()`.

#### Step 3: Baseline eval (bf16, ~10 clips)

Run `eval.py` on the unquantized model to get a baseline minADE. Use a small
clip count for speed.

```bash
uv run --active eval.py --limit 10 --print_every 1
```

Record: avg minADE (bf16), avg time/clip.

#### Step 4: Quantize to NVFP4

```bash
uv run --active quantize.py \
  --quant_format=nvfp4 \
  --num_of_calib_clips=100 \
  --save_model_dir=./outputs
```

This loads the 10B in FP16 (~22GB), calibrates on 100 PAI clips, then
quantizes. The calibration runs `sample_trajectories_from_data_with_vlm_rollout`
under `no_grad` + `autocast(fp16)`.

VRAM concern: 32GB total. Model FP16 ~22GB + KV cache for 256-token
generation ~5GB + activations ~2GB = ~29GB. Should fit but will be tight.
Watch for OOM.

Output: `./outputs/alpamayo1.5_nvfp4_calib100/`

#### Step 5: Eval quantized model

```bash
uv run --active eval.py --ckpt ./outputs/alpamayo1.5_nvfp4_calib100 --limit 10 --print_every 1
```

Record: avg minADE (NVFP4), avg time/clip.

#### Step 6: Compare and decide

| Metric | bf16 baseline | NVFP4 | Pass? |
|--------|--------------|-------|-------|
| avg minADE | TBD | TBD | < 1.0m? |
| avg time/clip | TBD | TBD | — |

- **If NVFP4 minADE < 1.0m**: consider skipping distillation. RL directly
  on the quantized 10B preserves full perception capacity and simplifies the
  pipeline.
- **If NVFP4 minADE >= 1.0m**: proceed with distillation to 2B as planned.

Also run FP8 as a middle-ground data point:
```bash
uv run --active quantize.py --quant_format=fp8 --num_of_calib_clips=100 --save_model_dir=./outputs
uv run --active eval.py --ckpt ./outputs/alpamayo1.5_fp8_calib100 --limit 10 --print_every 1
```

### Resource analysis: will it fit on the RTX 5090?

#### Disk (900 GB available)

| Item | Est. size | Notes |
|------|-----------|-------|
| Python venv (torch 2.8, flash-attn, modelopt, etc.) | ~5-10 GB | CUDA toolkit is on the Brev base image, not in venv |
| `alpamayo-recipes` repo (source + parquets) | ~0.5 GB | Includes 200KB calib parquet + 73KB eval parquet |
| `alpamayo1.5` + `alpamayo` git deps (uv-installed) | ~0.2 GB | Source only, pulled by `uv sync` |
| HF model cache: `nvidia/Alpamayo-1.5-10B` | ~22 GB | FP16, 5 safetensors shards. Self-contained — no Cosmos-Reason2 download. |
| HF processor cache: `Qwen/Qwen3-VL-2B-Instruct` | ~0.5 GB | Processor/tokenizer only, not full model weights |
| HF dataset cache: 110 clips (100 calib + 10 eval) | ~11-55 GB | `physical_ai_av` streams clips on demand via `get_clip_feature(maybe_stream=True)`. Each clip loads 4 cameras × 4 frames decoded from video + egomotion. Not full NuRec scenes (~1.5 GB each) — just camera frames + poses. |
| Quantized model outputs (NVFP4 + FP8) | ~16.5 GB | ~5.5 GB (NVFP4) + ~11 GB (FP8) |
| System overhead / tmp | ~5 GB | |
| **Total** | **~60-110 GB** | **Fits in 900 GB with massive headroom** |

The big variable is dataset clip caching. `physical_ai_av` streams on demand
and caches what it downloads. Per-clip data is camera video frames (4 cams ×
4 frames) + egomotion — not the full ~1.5 GB NuRec scene (those include .usdz
3D reconstructions we don't need). Even in the worst case where full camera
videos are cached (~500 MB/clip), 110 clips = ~55 GB, still well under 900 GB.

#### VRAM (32 GB available)

The alpamayo1.5 README states:

> Single-sample inference (num_traj_samples=1): ~24 GB
> Measured on an NVIDIA H100 80GB GPU.

The quant recipe README says it's tested on:

> NVIDIA 5090 GPU with CUDA 12

So NVIDIA has verified this works on 32 GB. Breakdown during calibration:

| Component | Est. VRAM | Notes |
|-----------|-----------|-------|
| Model weights (FP16) | ~22 GB | `Alpamayo1_5.from_pretrained(dtype=torch.float16)` |
| KV cache (256-token CoC generation) | ~2-3 GB | Accumulates during autoregressive generation |
| Diffusion expert activations | ~1-2 GB | Under `no_grad` + `autocast(fp16)` |
| Camera frames in VRAM | ~0.5 GB | 4 cams × 4 frames × 3 × H × W in fp16 |
| PyTorch overhead / fragmentation | ~1-2 GB | |
| **Total** | **~27-30 GB** | **Fits in 32 GB, but tight (~85-94%)** |

Risk: `num_traj_samples=6` (default) generates 6 trajectory samples per clip
during calibration, which adds diffusion sampling memory. If OOM occurs:
- Reduce to `--num_traj_samples=1` in quantize.py
- Or fall back to B300 (288 GB VRAM at $9.49/hr)

The quant recipe also has explicit memory management in quantize.py:
`gc.collect()` + `torch.cuda.empty_cache()` after each calibration clip.

#### Compute (12 vCPUs, 120 GB RAM)

- 12 vCPUs: sufficient for data loading, video decoding, preprocessing
- 120 GB RAM: more than enough — model is 22 GB, clip data is small
- No bottleneck here

#### Time estimate

| Step | Est. time | Notes |
|------|-----------|-------|
| Instance boot + SSH | ~2 min | Brev 5090 boot time |
| Environment setup + flash-attn build | ~30-60 min | `MAX_JOBS=4 uv sync` — flash-attn compilation is slow |
| Model download (22 GB from HF) | ~10-20 min | Depends on network speed |
| Baseline eval (10 clips) | ~10-20 min | ~1-2 min/clip including HF streaming |
| NVFP4 quantization (100 calib clips) | ~2-3 hr | ~1-2 min/clip for load + VLM rollout + calibration |
| NVFP4 eval (10 clips) | ~10-20 min | |
| FP8 quantization (100 calib clips) | ~2-3 hr | Can skip if NVFP4 result is clear |
| FP8 eval (10 clips) | ~10-20 min | |
| **Total** | **~5-8 hr** | Matches PLAN estimate of 4-8 hr |

#### Fallback: B300 if RTX 5090 OOMs

If the 32GB VRAM on the 5090 is insufficient during calibration, fall back
to B300:

```bash
brev delete alpamayo-quant-gate  # if created
brev create alpamayo-quant-gate --type verda_B300
```

- 1x B300, 288GB VRAM, $9.49/hr, 2TB disk
- 9x more VRAM, no OOM concerns
- At ~5-8 hours: ~$47-76 (vs ~$4-6 on 5090)

### Estimated cost

| Scenario | Instance | Hours | Cost |
|----------|----------|-------|------|
| Best case (5090 works) | RTX 5090 @ $0.78 | 5-8 | $4-6 |
| Fallback (B300 needed) | B300 @ $9.49 | 5-8 | $47-76 |

### Prerequisites before launching

- [ ] HF access approved for `nvidia/Alpamayo-1.5-10B` (gated model)
- [ ] HF access approved for `nvidia/PhysicalAI-AV` (gated dataset)
- [ ] Brev CLI installed and working locally
