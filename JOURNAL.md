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
| avg minADE (100 clips) | **0.817m** | **0.848m** | **Yes** (< 1.0m) |
| avg minADE (10 clips, smoke) | 0.618m | 0.646m | Yes |
| avg time/clip (100 clips) | 1,399 ms | 4,378 ms | — (3.1× slower) |

- **If NVFP4 minADE < 1.0m**: consider skipping distillation. RL directly
  on the quantized 10B preserves full perception capacity and simplifies the
  pipeline.
- **If NVFP4 minADE >= 1.0m**: proceed with distillation to 2B as planned.

Also run FP8 as a middle-ground data point (skipped — NVFP4 result was clear):
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

- [x] HF access approved for `nvidia/Alpamayo-1.5-10B` (gated model)
- [x] HF access approved for `nvidia/PhysicalAI-AV` (gated dataset)
- [x] Brev CLI installed and working locally

---

## 2026-07-02 — Phase 0 results & decision

### Execution summary

Ran on **1× NVIDIA B300** (275 GB VRAM, compute cap 10.3) via Brev — not the
planned RTX 5090, but the quant recipe README lists B300 as tested hardware.
Total wall time ~2 hours (much faster than the 5–8 hr estimate, thanks to B300
throughput and cached HF assets after first download).

| Step | Status | Notes |
|------|--------|-------|
| Environment setup | Done | `alpamayo-recipes/recipes/alpamayo1_5_quant`, `uv sync`, flash-attn |
| bf16 baseline eval (10 clips) | Done | 10/10 succeeded — fast smoke test |
| bf16 baseline eval (100 clips) | Done | 10/10 succeeded — **writeup numbers** |
| NVFP4 quantization (100 calib clips) | Done | ~24 min; output `outputs/alpamayo1.5_nvfp4_calib100` (~7.6 GB) |
| NVFP4 eval (10 clips) | Done | 10/10 succeeded (after B300 runtime fixes below) |
| NVFP4 eval (100 clips) | Done | 100/100 succeeded — **writeup numbers** |
| FP8 quant + eval | Skipped | NVFP4 passed gate decisively |

Eval clips: `1005_7cam_gold_eval_metadb_public.parquet` (644 total; used first
100 for writeup, first 10 for fast gate).
Calibration clips: `0417_5k_train_set_for_calibration_25.10.parquet` (100).
Default eval settings: `num_traj_samples=6`, `seed=42`.

### Results (100 clips — primary)

| Metric | bf16 (`nvidia/Alpamayo-1.5-10B`) | NVFP4 (`alpamayo1.5_nvfp4_calib100`) | Δ |
|--------|----------------------------------|--------------------------------------|---|
| **mean minADE** | **0.817m** | **0.848m** | +0.030m (+3.7%) |
| **median minADE** | 0.480m | 0.526m | +0.046m |
| **p95 minADE** | 2.617m | 2.561m | — |
| **avg time/clip** | 1,400 ms | 4,380 ms | 3.1× slower |
| clips succeeded | 100/100 | 100/100 | — |
| clips < 1.0m gate | 78/100 | 79/100 | — |

Full per-clip data: `nvfp4-eval/results/per_clip_100.csv` (verbose re-run,
`--print_every 1`). Figures: `nvfp4-eval/figures/`.

**Distribution & statistics (100 clips):**
- Heavy-tailed: max minADE ~9.5m on outlier clips; median (~0.5m) much lower
  than mean (~0.8m).
- Per-clip delta: NVFP4 better on **42**, worse on **58**; mean |Δ| = 0.28m.
- Pearson **r = 0.90** — errors correlate strongly; quant preserves ranking.
- Paired t-test **p = 0.57**; bootstrap 95% CI for mean Δ: **[-0.29, +0.35] m**
  (spans zero). The +3.7% mean shift is not statistically significant at N=100.
- Gate breakdown: **75** both pass, **18** both fail, 3 bf16-only pass, 4
  NVFP4-only pass.

Logs: `nvfp4-eval/logs/bf16_verbose_100.log`, `nvfp4_verbose_100.log`.

### Results (10 clips — smoke test only)

Fast gate run; **do not use as headline writeup numbers** — the first 10 clips
were an easier subset (~32% lower absolute minADE than the 100-clip mean).

| Metric | bf16 | NVFP4 | Δ |
|--------|------|-------|---|
| avg minADE | 0.618m | 0.646m | +0.029m (+4.7%) |
| avg time/clip | 1,547 ms | 4,205 ms | 2.7× slower |

The **relative** NVFP4 penalty is stable across 10 vs 100 clips (~4%); the
**absolute** error was underestimated by the smoke test.

Per-clip minADE for smoke test (meters):

| # | clip_id (prefix) | bf16 | NVFP4 |
|---|------------------|------|-------|
| 1 | 55b172f7 | 1.452 | 1.540 |
| 2 | b874a6b1 | 0.652 | 0.525 |
| 3 | f4433f2a | 0.573 | 0.515 |
| 4 | 10bad3f8 | 0.113 | 0.173 |
| 5 | 30352e9e | 0.264 | 0.542 |
| 6 | be1d9f78 | 0.911 | 0.345 |
| 7 | 752561f7 | 0.117 | 0.432 |
| 8 | a8dd5ceb | 0.161 | 0.125 |
| 9 | 5a1ea30e | 0.637 | 0.938 |
| 10 | f289708f | 1.297 | 1.330 |

NVFP4 adds ~3.1× inference latency on B300 at 100 clips (ModelOpt fake-quant
GEMM via Triton; logs show `RealQuantLinear: No real-quant GEMM found` on
every layer). Acceptable for offline RL training; may need kernel tuning for
the 0.1s AlpaSim driver budget later. Latency is fake-quant path — not
optimized Blackwell NVFP4 kernels.

### B300 runtime fixes (required)

Stock torch 2.8.0 + triton 3.4.0 (bundled with torch) does **not** work out of
the box on B300 (`sm_103a`). These are infrastructure issues, not model-quality
problems — the quantized checkpoint loads and runs fine once fixed.

1. **NVRTC (bf16 eval):** bundled `libnvrtc.so.12` (CUDA 12.8) doesn't know
   `sm_103`. Fix: install `nvidia-cuda-nvrtc` pip package (CUDA 13) and symlink
   over the bundled library; set `LD_LIBRARY_PATH` to cu13 lib dir.

2. **Triton ptxas (NVFP4 eval):** bundled ptxas only supports up to `sm_101a`.
   Fix: symlink `nvidia/cu13/bin/ptxas` into
   `triton/backends/nvidia/bin/ptxas`.

3. **Triton version (NVFP4 eval):** triton 3.4.0 fails CUDA version check when
   using cu13 ptxas (`got CUDA version: 13.3`). Fix: pin **`triton>=3.7.1`**
   in `pyproject.toml` with `[tool.uv] override-dependencies` (torch 2.8.0
   pulls 3.4.0 otherwise; `uv run` re-syncs it back).

All three fixes are captured in
`alpamayo-recipes/recipes/alpamayo1_5_quant/run_env.sh` and `pyproject.toml`.

### Decision (revised 2026-07-02)

**Skip distillation. RL on bf16 10B in Alpagym, then FP8 quantize for
deployment.**

The initial decision ("RL directly on the NVFP4-quantized 10B") was revised
after discovering two constraints:

1. **Alpagym only loads bf16 checkpoints.** The `alpamayo_r1` policy bundle's
   `load_inference_model` enforces `dtype=bfloat16` via
   `ExpertModelRL.from_pretrained`. There is no quantized-model RL path.
   Alpagym runs RL with an in-process trainable model, not an external Docker
   driver — it can't RL-tune a black-box container.

2. **Challenge eval runs on Hopper (H100), not Blackwell.** The README says
   "EC2 configs start 16 replicas across GPUs 4-7" on 8-GPU instances
   (`p5.48xlarge` = 8×H100). NVFP4 is Blackwell-only. The submitted Docker
   image must use **FP8** (Hopper-native) or INT8, not NVFP4.

**Corrected pipeline:**
```
nvidia/Alpamayo-1.5-10B (bf16, HF)
  → convert to Alpagym format (convert_release_config_to_training.py)
  → RL training in Alpagym (bf16, traj_future path, GRPO)
  → export RL checkpoint (convert_cosmos_rl_checkpoint.py)
  → quantize to FP8 (quantize.py --quant_format=fp8)
  → bake FP8 weights into Docker image
```

This is **train-then-compress**, not compress-then-train. The Phase 0 gate
is still valid: it proves PTQ preserves perception (+3.7% at NVFP4, so FP8
will be even better), giving confidence that post-RL FP8 quantization will
preserve RL-tuned behavior. PTQ rounds weights; it doesn't retrain, so
self-correcting behavior learned through RL should survive much better than
distillation would have.

**What the NVFP4 result tells us (still valid):**
- NVFP4 avg minADE **0.848m** is under the **1.0m gate** (~15cm margin).
- NVFP4 adds only **+3.7%** mean error vs bf16 (0.817m → 0.848m) — consistent
  with the ~4.7% seen on the smoke test; relative quant penalty is stable.
- Per-clip behavior is reasonable — no systematic blow-up; Pearson r=0.90.
- Skipping Phases 1–2 saves ~$220 and ~36+ hours of distillation work.
- Full 10B perception capacity is preserved vs. a 2B student.
- Since FP8 (8-bit) is less aggressive than NVFP4 (4-bit), FP8 quality loss
  will be strictly less than the +3.7% we measured.

**Writeup headline (suggested):**
> On 100 held-out PAI gold eval clips, NVFP4 increases mean minADE by 3.7%
> (0.817 → 0.848 m), remaining under our 1.0 m gate. The shift is not
> statistically significant (paired t-test p=0.57); per-clip errors correlate
> strongly (r=0.90). This confirms PTQ preserves perception quality, enabling
> a train-then-compress pipeline: RL on bf16 10B, then FP8 quantize for
> Hopper-compatible deployment.

**Next steps (revised):**
1. ~~Update PLAN.md: drop 2B distillation phases, reflect train-then-compress.~~ Done.
2. ~~Run **FP8 gate eval** on H100~~ — **Done.** FP8 0.836 m < 1.0 m gate.
3. Set up Alpagym RL pipeline with bf16 10B checkpoint (converted to Alpagym
   format via `convert_release_config_to_training.py`).
4. RL training in Alpagym (bf16, `traj_future` path, GRPO with AlpaSim).
5. Export RL checkpoint, re-quantize to FP8, bake into Docker image.
6. Profile FP8 inference latency for 0.1s driver budget on H100.

**Cost revision:** RL is now on bf16 10B (not 2B), ~5× more expensive per
rollout than the PLAN assumed. Moderate RL (~$320 vs ~$64), aggressive RL
(~$1,640 vs ~$328). The ~$220 distillation savings partially offset this.

**FP8 efficiency levers (if 16 GiB VRAM is tight with 2 concurrent rollouts):**
- `--quant_weight_only` FP8: weights ~11 GB, activations stay FP16
- `num_traj_samples=1`: single trajectory, no selection
- Reduce diffusion `inference_step` count
- `traj_future` path (no CoC generation) — already planned
- torch.compile on forward pass
- TensorRT export for expert + vision encoder
- INT8 weight-only as fallback (~6-7 GB, H100-compatible)

**Empirical artifacts:** `nvfp4-eval/` in this repo — summary JSON, CSVs,
methodology config, log manifest, figure checklist, log parser script.

### Expanded eval for writeup (done)

Verbose re-run with `--print_every 1` for full per-clip export + figures:

```bash
nvfp4-eval/scripts/run_verbose_eval_100.sh
nvfp4-eval/scripts/generate_figures.py
```

Outputs: `results/per_clip_100.csv`, `figures/*.png`. Wall time ~33 min on B300.

---

## 2026-07-02 — FP8 gate evaluation (H100)

### Execution summary

Ran on **1× NVIDIA H100 PCIe** via `fp8-eval/scripts/run_fp8_gate.sh`. Total wall
time ~1 hr 7 min (quant ~27 min, bf16 eval ~18 min, FP8 eval ~21 min).

| Step | Status | Notes |
|------|--------|-------|
| FP8 quantization (100 calib clips) | Done | output `outputs/alpamayo1.5_fp8_calib100` (~11 GB) |
| bf16 baseline eval (100 clips) | Done | 100/100 succeeded |
| FP8 eval (100 clips) | Done | 100/100 succeeded |

Same parquets and settings as NVFP4 gate (`num_traj_samples=6`, `seed=42`).

### Results (100 clips — H100)

| Metric | bf16 | FP8 | Δ |
|--------|------|-----|---|
| **mean minADE** | **0.854 m** | **0.836 m** | −0.018 m (−2.1%) |
| **avg time/clip** | 2,582 ms | 5,329 ms | 2.1× slower |
| clips succeeded | 100/100 | 100/100 | — |

Full summary: `fp8-eval/results/summary.json`. Logs: `fp8-eval/logs/`.

**Gate: PASSED** — FP8 mean minADE **0.836 m** < **1.0 m** threshold (164 m margin).

### Comparison to NVFP4 gate (B300, 100 clips)

| Format | mean minADE | avg time/clip | Hardware |
|--------|-------------|---------------|----------|
| bf16 | 0.817 m | 1,400 ms | B300 |
| NVFP4 | 0.848 m | 4,380 ms | B300 |
| bf16 | 0.854 m | 2,582 ms | H100 |
| **FP8** | **0.836 m** | **5,329 ms** | **H100** |

H100 bf16 baseline (0.854 m) is ~4.5% higher than B300 bf16 (0.817 m) — likely
run variance, not a hardware quality difference. FP8 (0.836 m) sits between the
two bf16 baselines and is slightly better than NVFP4 (0.848 m), consistent with
8-bit being less aggressive than 4-bit.

On this paired H100 run, FP8 is marginally *better* than bf16 (−2.1%); treat as
noise (same caveat as NVFP4 paired t-test p=0.57 at N=100).

### Latency caveat

Logs show `RealQuantLinear: No real-quant GEMM found` on every layer during FP8
eval — ModelOpt fake-quant path, same as NVFP4 on B300. The 5.3 s/clip FP8
latency and 2.1× slowdown vs bf16 are **not** representative of optimized
Hopper FP8 kernels. Still far above the 0.1 s `drive()` budget; kernel tuning /
TensorRT / `traj_future` path required before deployment.

### Decision

**Both quantization gates passed.** Train-then-compress pipeline confirmed:
RL on bf16 10B in Alpagym → FP8 PTQ for Hopper deployment. Proceed to Phase 1
(Alpagym RL setup).

---

## Writeup requirements (NVFP4 gate evaluation section)

Checklist of what the technical writeup / blog must include beyond this journal.
Use the journal as source material; expand each item for external readers.

### Narrative & motivation

- [ ] **Problem framing** — E2E challenge needs a deployable driver; full bf16
  10B is too heavy for the container budget; distillation (2B student) was the
  fallback plan (~$220, ~36+ hr).
- [ ] **Gating hypothesis** — NVFP4 PTQ retains enough open-loop driving
  quality to skip distillation and RL-tune the quantized 10B directly.
- [ ] **Decision criteria** — avg minADE < 1.0m on PAI gold eval clips;
  explain *why* this threshold and *why* open-loop minADE (not closed-loop
  collision rate) is the right fast gate before sim RL investment.

### Model & task context (assume reader is unfamiliar)

- [ ] **Alpamayo 1.5 in one paragraph** — 10B VLA: Qwen3-VL-2B backbone +
  diffusion trajectory expert; VLM chain-of-causation rollout then trajectory
  sampling.
- [ ] **What NVFP4 is** — NVIDIA 4-bit floating-point quant format;
  Blackwell-only, ~4× weight compression (~22 GB → ~5.5–7.6 GB).
- [ ] **What minADE measures** — minimum average displacement error over
  `num_traj_samples=6` predicted trajectories vs. GT future ego path at
  `t0_us=5.1s`; units in meters; lower is better.

### Methodology (reproducibility)

- [ ] **Recipe & scripts** — `alpamayo-recipes/recipes/alpamayo1_5_quant/`
  (`quantize.py`, `eval.py`); link to NVIDIA recipe README.
- [ ] **Checkpoints** — bf16: `nvidia/Alpamayo-1.5-10B`; NVFP4:
  `outputs/alpamayo1.5_nvfp4_calib100` (100 calib clips, `max` calibrator,
  ModelOpt 0.43.0).
- [ ] **Data** — eval: `1005_7cam_gold_eval_metadb_public.parquet`; calib:
  `0417_5k_train_set_for_calibration_25.10.parquet`; gated HF dataset access.
- [ ] **Eval settings** — `num_traj_samples=6`, `seed=42`, `max_generation_length=256`,
  `top_p=0.98`, `temperature=0.6`.
- [ ] **Hardware** — 1× NVIDIA B300 (sm_103a), torch 2.8.0, triton 3.7.1;
  document B300 runtime fixes (`run_env.sh`).
- [ ] **Exact commands / commit** — pin git SHA of `alpamayo-recipes` and log
  file paths for reproducibility.

### Results to report

- [x] **Headline table** — bf16 vs NVFP4: avg minADE, avg time/clip, clip
  success rate, checkpoint size. **100-clip numbers:** 0.817m / 0.848m / 3.1×.
- [x] **Per-clip analysis** — scatter, delta histogram, gate breakdown in
  `figures/`; full CSV in `results/per_clip_100.csv`.
- [x] **Statistical honesty** — N=100 of 644; high per-clip variance documented
  (running avg swings 0.70–0.86m); 10-clip smoke test was optimistic (~32%
  lower absolute error).
- [ ] **Trajectory figures** — 1–2 overlaid pred vs GT trajectory plots for
  representative clips (best case, worst regression).

### Limitations & caveats (must not skip)

- [ ] **Open-loop only** — minADE on held-out PAI clips does not predict
  closed-loop collision-free duration in AlpaSim; sim RL still required.
- [ ] **Fake-quant inference on B300** — logs show `RealQuantLinear: No
  real-quant GEMM found`; accuracy reflects quantized weights but latency
  (3.1× slower at 100 clips) is fake-quant Triton path, not optimized
  Blackwell NVFP4 kernels. FP8 on H100 will use real Hopper FP8 GEMM —
  latency will differ and should be measured separately.
- [ ] **NVFP4 not deployable** — challenge eval runs on H100 (Hopper);
  NVFP4 is Blackwell-only. Deployment must use FP8 or INT8. The NVFP4 result
  serves as a worst-case bound (4-bit > 8-bit degradation).
- [x] **Sample size** — 100-clip eval done; 10-clip gate was directional only.
  Remaining gap vs full 644-clip benchmark noted.
- [x] **FP8 evaluated on H100** — mean minADE 0.836 m (< 1.0 m gate). See
  2026-07-02 FP8 gate section. Fake-quant latency still needs optimization.

### Decision & downstream implications

- [x] **Gate outcome** — PASSED: skip distillation. RL on bf16 10B in Alpagym,
  then FP8 quantize post-RL for Hopper-compatible deployment (0.848m < 1.0m
  on 100 clips at NVFP4; FP8 will be strictly better).
- [ ] **Economics** — ~$220 and ~36+ hr saved vs 2B distillation pipeline.
  RL cost increases ~5× (bf16 10B vs 2B): moderate ~$320, aggressive ~$1,640.
- [ ] **Tradeoffs** — keep full 10B capacity for RL; FP8 deployment (~11 GB
  weights) is tighter in 16 GiB VRAM than NVFP4 would have been (~7.6 GB);
  latency profiling on H100 real-quant still needed for 0.1s budget.
- [ ] **Train-then-compress rationale** — Alpagym only loads bf16; challenge
  eval is H100 (Hopper, no NVFP4); PTQ preserves RL behavior better than
  distillation would have.
- [ ] **Alternative path rejected** — brief description of what distillation
  would have entailed (SFT 2B student from 10B teacher, then RL).

### Optional for paper-grade depth

- [ ] Full 644-clip eval with confidence intervals.
- [ ] FP8 / AutoQuant comparison at 6.5 effective bits.
- [ ] Closed-loop AlpaSim metrics post-RL as validation that open-loop gate
  was sufficient.
- [ ] Comparison to NVIDIA-published Alpamayo 1.5 benchmarks if available.
