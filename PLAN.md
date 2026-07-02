# Comprehensive Plan: AlpaSim E2E Challenge Driver

## Objective

Maximize **collision-free driving duration** in closed-loop simulation across held-out
real-world scenes. The submission is a Docker container implementing the
`egodriver.EgodriverService` gRPC interface.

## Core Metric

This is **not a trajectory prediction competition**. It is a **closed-loop survival
competition**. The model that drives the longest without crashing wins. What matters is:

- **Self-correcting behavior** — small trajectory errors must not compound into crashes
- **Perception quality** — the model must understand obstacles, road geometry, and routes
- **Constraint compliance** — ≤0.1s per `drive()` call, ≤16 GiB VRAM, ≤40 GiB image, no network

Open-loop metrics (minADE/minFDE) are irrelevant if the model diverges in closed loop.

## Strategy: RL on bf16 10B, Then FP8 Quantize for Deployment

### Why this order

- **RL directly optimizes the deployed model.** GRPO shapes self-correcting
  behavior in the 10B's own parameter space — the exact model that runs at
  inference time (after quantization).
- **Alpagym only loads bf16 checkpoints.** The `alpamayo_r1` policy bundle
  enforces `dtype=bfloat16` via `ExpertModelRL.from_pretrained`. There is no
  quantized-model RL path. RL must run on bf16 weights.
- **Challenge eval runs on Hopper (H100), not Blackwell.** AWS `p5.48xlarge`
  (8×H100) is the eval hardware. NVFP4 is Blackwell-only. The submitted Docker
  image must use **FP8** (Hopper-native) for deployment.
- **PTQ preserves RL behavior.** Post-training quantization rounds weights; it
  doesn't retrain. Self-correcting behavior learned through RL survives
  compression much better than distillation (which transfers input→output
  mappings, not closed-loop robustness).
- **Phase 0 gate proved PTQ is safe.** NVFP4 (4-bit) added only +3.7% minADE.
  FP8 (8-bit) is strictly less aggressive — degradation will be less.

### Corrected checkpoint flow

```
nvidia/Alpamayo-1.5-10B (bf16, HF)
  → convert to Alpagym format (convert_release_config_to_training.py)
  → RL training in Alpagym (bf16, traj_future path, GRPO with AlpaSim)
  → export RL checkpoint (convert_cosmos_rl_checkpoint.py)
  → quantize to FP8 (quantize.py --quant_format=fp8)
  → bake FP8 weights into Docker image (~11 GB)
```

### Why no CoC at inference

Chain-of-Causation (CoC) reasoning requires autoregressive VLM generation
(~256 tokens, ~50-80ms) before the diffusion expert runs. On a 10B model at
0.1s budget, this leaves no room for vision encoding, diffusion sampling, or
multi-session contention.

The `last_component="traj_future"` path (used by Alpagym's RL recipe) skips
VLM generation entirely — the expert conditions on prefill-only KV cache.
This is the deploy path.

**CoC is not generated at RL or deploy time.** The 10B was RL post-trained by
NVIDIA with CoC reasoning, so the reasoning quality is already internalized
into the model's latent representations. Our RL further tunes for closed-loop
robustness on the `traj_future` path directly.

---

## Architecture

### RL model (10B Alpamayo 1.5, bf16)

```
Camera images → Vision encoder → VLM prefill → Diffusion expert → trajectory
                                               ↑
                                     prefill-only KV cache (no CoC generation)
```

- Model: `nvidia/Alpamayo-1.5-10B` (gated, needs HF access approval)
- Dataset: `nvidia/PhysicalAI-Autonomous-Vehicles` (gated)
- Inference: `sample_trajectories_from_data` with `last_component="traj_future"`
- RL framework: Alpagym (`alpamayo_r1` policy bundle, `ExpertModelRL`, bf16)
- VRAM: ~24 GB single-sample (H100 80GB)

### Deployed model (10B + FP8 quantization, RL post-trained)

```
Camera images → Vision encoder → VLM prefill → Diffusion expert → trajectory
                                               ↑
                                     prefill-only KV cache, FP8 weights
```

- Same architecture and inference path as RL model
- FP8 quantized weights baked into Docker image (no network at runtime)
- RL post-trained for closed-loop robustness, then PTQ to FP8
- ~11 GB weights (FP8), fits in 16 GiB VRAM with 2 concurrent rollouts
- Eval hardware: 8×H100 (Hopper) — FP8 is native, NVFP4 is not supported
- 2 concurrent rollouts per replica, 16 replicas across 4 GPUs

---

## Phases

### Phase 0: Baseline & Setup (DONE)

**Goal:** Get the starter kit running and submitted to establish a floor score.
Quantization gate: confirm PTQ preserves perception quality.

- [x] Register at the [challenge HF Space](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026), get approved
- [x] Get HF access to `nvidia/Alpamayo-1.5-10B` and `nvidia/PhysicalAI-Autonomous-Vehicles`
- [ ] Build the starter driver container:
  ```bash
  docker build -f e2e_challenge/starter_kit/Dockerfile -t alpasim-e2e-starter-driver:latest .
  ```
- [ ] Run local PAI smoke test against AlpaSim
- [ ] Submit the straight-line baseline to get a leaderboard score
- [ ] Authenticate with the challenge CLI (`auth-url` → `configure-token` → `me`)
- [ ] Verify ECR login works (`ecr-login`)

**Decision gate (DONE — PASSED):** NVFP4-quantized 10B Alpamayo has sufficient
perception quality. On 100 PAI gold eval clips, NVFP4 avg minADE = 0.848m
(< 1.0m gate, +3.7% vs bf16 0.817m, not statistically significant p=0.57).
Since FP8 (8-bit) is less aggressive than NVFP4 (4-bit), FP8 will degrade
perception even less. Distillation is skipped.

- [x] Quantize 10B Alpamayo to NVFP4 (100 calib clips, ModelOpt 0.43.0)
- [x] Run eval on 100 clips: bf16 vs NVFP4 minADE comparison
- [x] Result: NVFP4 minADE 0.848m < 1.0m gate → skip distillation
- [x] Run FP8 gate eval on H100 (deployment format, Hopper-native):
  ```bash
  # ran on 1× H100 PCIe via fp8-eval/scripts/run_fp8_gate.sh
  # quantize ~27 min; bf16 eval ~18 min; FP8 eval ~21 min
  ```
  **Result (100 clips):** bf16 0.854 m, FP8 **0.836 m** (< 1.0 m gate, PASSED).
  FP8 checkpoint ~11 GB. Latency: bf16 2.6 s/clip, FP8 5.3 s/clip (2.1×).
  Note: ModelOpt still reports `RealQuantLinear: No real-quant GEMM found`
  (fake-quant path on H100 too); latency is not yet optimized Hopper FP8 kernels.
  Artifacts: `fp8-eval/results/summary.json`, `fp8-eval/logs/`.

### Phase 1: RL Post-Training (GRPO via Alpagym)

**Goal:** Directly optimize the bf16 10B for collision-free duration in closed-loop
simulation.

Reference: [Alpamayo 1.x RL recipe](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo1_x_rl)
and Alpagym's `AlpamayoR1InferenceModel` / `AlpamayoPolicy` infrastructure.

- [ ] Request access to `nvidia/Alpamayo-1.5-10B` and `nvidia/PhysicalAI-Autonomous-Vehicles` on HF
- [ ] Set up Alpagym environment with AlpaSim as the simulator backend
- [ ] Convert checkpoint: `scripts/convert_release_config_to_training.py`
      (HF → Alpagym format, sets `ALPAMAYO_MODEL_DIR`)
- [ ] Configure the 10B as the policy model:
  - `load_inference_model`: load 10B with `attn_implementation="sdpa"`, `dtype=bfloat16`
  - `last_component="traj_future"` (match deploy path)
  - `num_context_frames`, `num_historical_waypoints`, `num_future_waypoints` matching driver config
- [ ] Design reward function:
  - Primary: collision-free duration (directly the competition metric)
  - Secondary: progress along route (encourage forward motion, not just standing still)
  - Optional: risk-horizon discounting — penalize near-future collisions more than far-future
  - The reward system supports `metric` and `distance_to_gt` term kinds
- [ ] Configure GRPO:
  - Use `force_gt_duration_us` warmup so episodes start with ground-truth trajectory
    (ensures the model has enough ego history and a stable starting state for exploration)
  - Start with conservative scenes (straight roads), curriculum toward harder scenes
  - Multiple rollout workers (8-GPU topology, 32-64 concurrent rollouts)
- [ ] Train and monitor:
  - Track collision-free duration across training
  - Watch for reward collapse / KL spikes (use trackio alerts)
  - Check that the model doesn't learn to game the reward (e.g., driving in circles)
- [ ] Evaluate RL-tuned model in AlpaSim closed loop
- [ ] Compare: baseline 10B vs RL-tuned 10B on collision-free duration
- [ ] Export RL checkpoint: `scripts/convert_cosmos_rl_checkpoint.py`

> **Cost note:** RL on bf16 10B is ~5× more expensive per rollout than the
> original 2B plan. Moderate: ~$320, aggressive: ~$1,640. The ~$220
> distillation savings partially offset this.

### Phase 2: gRPC Driver Integration

**Goal:** Wire the RL-tuned 10B model into the gRPC driver container, replacing the
starter kit's straight-line driver.

Reference implementation: `AlpamayoPolicy` in
`alpagym/packages/runtime/src/alpagym_runtime/policies/alpamayo/policy.py` — this class
does exactly the buffering + preprocessing + inference + postprocessing pipeline needed.

- [ ] Implement `EgodriverServiceServicer` with per-session state:
  - `SessionState`: frame buffer (ring buffer per camera), ego history, route waypoints
  - `start_session` / `close_session`: create/destroy session state
  - `submit_image_observation`: decode `image_bytes` → uint8 tensor, append to camera ring buffer
  - `submit_egomotion_observation`: extract poses + dynamic states, append to ego history
  - `submit_route`: store route waypoints + timestamp
  - `drive`: run full inference pipeline (see below)
- [ ] Implement the `drive()` inference pipeline:
  1. Extract camera context: latest N frames per camera as `[N_total, 1, 3, H, W]` uint8
  2. Extract ego history: normalize world-frame poses into rig-t0 frame (xyz + rotation matrices)
  3. Convert route: transform waypoints into rig-t0 frame as `[NUM_ROUTE_WAYPOINTS, 2]`
  4. Run model: `sample_trajectories_from_data` with `last_component="traj_future"`
  5. Convert output: trajectory (xyz + rotation) → `PoseAtTime` protos (Vec3 + Quat + timestamp)
- [ ] Handle multi-session concurrency (2 concurrent rollouts per replica):
  - Thread-safe session dict
  - Shared model instance (inference is stateless between sessions)
  - `ThreadPoolExecutor` with enough workers for concurrent `drive()` calls
- [ ] Write the Dockerfile:
  - Base: CUDA runtime image
  - Install: alpasim_grpc, torch, transformers, alpamayo1_5, dependencies
  - Copy: model weights (baked in), driver.py
  - Run as non-root user, read-only root filesystem, tmpfs at /tmp and /run
  - Expose port 6789
- [ ] Test locally: build container, run smoke test against AlpaSim
- [ ] Measure: latency per `drive()` call, peak VRAM, image size

### Phase 3: Inference Optimization & FP8 Quantization

**Goal:** Hit ≤0.1s latency and ≤16 GiB VRAM with the FP8-quantized 10B model.

- [ ] **Quantize RL checkpoint to FP8:**
  ```bash
  uv run --active quantize.py --quant_format=fp8 --num_of_calib_clips=100 --save_model_dir=./outputs
  ```
  - Target: ~11 GB weights, <14 GB total VRAM with KV cache + vision encoder + 2 sessions
  - Test: does FP8 degrade perception enough to affect closed-loop driving?
  - If VRAM is tight with 2 concurrent rollouts, try `--quant_weight_only` (weights FP8, activations FP16)
- [ ] **Diffusion optimization:** Reduce inference steps
  - The `diffusion_kwargs` in the inference adapter supports `inference_step` and `int_method`
  - Find the minimum steps that maintain trajectory quality
- [ ] **Batch size:** Set `num_traj_samples=1`, `num_traj_sets=1` (single trajectory, no selection)
- [ ] **KV cache:** Enable `use_cache=True` for prefill (already default in Alpamayo)
- [ ] **Optional:** torch.compile on the model forward pass
- [ ] **Optional:** TensorRT export for the expert + vision encoder
- [ ] **Fallback if FP8 too tight:** INT8 weight-only (~6-7 GB, H100-compatible)
- [ ] **Verify:** End-to-end `drive()` latency ≤0.1s with 2 concurrent sessions on H100
- [ ] **Verify:** Peak VRAM ≤16 GiB with 2 concurrent sessions on H100

### Phase 4: Containerization & Submission

**Goal:** Package the FP8-quantized RL-tuned 10B model into a compliant Docker container
and submit.

- [ ] Bake FP8-quantized RL-tuned weights into the Docker image
- [ ] Verify image size ≤40 GiB
- [ ] Verify read-only root filesystem works (model loads from baked-in weights, no writes)
- [ ] Verify no outbound network calls at runtime
- [ ] Run final local smoke test (PAI track)
- [ ] Run local smoke test (nuPlan track, if attempting)
- [ ] Push to ECR:
  ```bash
  uv run e2e_challenge/competitor_cli/alpasim_challenge.py ecr-login
  docker tag <image>:<tag> 696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/<team_id>:<tag>
  docker push 696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/<team_id>:<tag>
  ```
- [ ] Submit:
  ```bash
  uv run e2e_challenge/competitor_cli/alpasim_challenge.py submit --track pai \
    696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/<team_id>:<tag>
  ```
- [ ] Check status and leaderboard

### Phase 5: Iterate

- [ ] Analyze failure modes from leaderboard results (which scenes does it crash on?)
- [ ] If latency is the bottleneck: reduce diffusion steps, torch.compile, TensorRT, or INT8
- [ ] If perception is the bottleneck: try higher-res input, more context frames, or FP8 weight-only
- [ ] If closed-loop robustness is the bottleneck: more RL training, adjust reward design
- [ ] If VRAM is the bottleneck: reduce context frames, reduce image resolution, or INT8 weight-only
- [ ] Try both tracks (PAI and nuPlan) — they may favor different configurations

---

## Key Design Decisions

### 1. Train-then-compress, not compress-then-train

RL on bf16 10B in Alpagym (the only supported path), then FP8 PTQ for
deployment. PTQ rounds weights; it doesn't retrain, so self-correcting
behavior learned through RL survives compression. Distillation would transfer
input→output mappings but lose closed-loop robustness.

### 2. Remove CoC from RL and deployment

CoC generation is too slow for 0.1s latency. The 10B uses the `traj_future`
path (prefill → expert, no VLM generation). The model was already RL
post-trained by NVIDIA with CoC reasoning, so reasoning quality is
internalized. Our RL further tunes for closed-loop robustness on the
`traj_future` path.

### 3. `traj_future` as the single inference path

Used for RL and deployment. No train/deploy gap. Already the default in
Alpagym's RL recipe (`bundle.py`).

### 4. 10B instead of 2B student

The Phase 0 gate proved NVFP4 (4-bit) preserves perception (+3.7% minADE).
FP8 (8-bit) will be even better. The 10B retains full perception capacity
for RL — no distillation bottleneck. Trade-off: RL is ~5× more expensive
per rollout than a 2B would have been, and FP8 weights (~11 GB) are tighter
in 16 GiB VRAM than a 2B would have been (~1-2 GB).

### 5. FP8 for deployment, not NVFP4

Challenge eval runs on 8×H100 (Hopper). NVFP4 is Blackwell-only. FP8
(E4M3) is Hopper-native and supported by ModelOpt's quant recipe.

### 6. RL reward = collision-free duration

Directly optimizes the competition metric. Progress-along-route as secondary reward
prevents the degenerate strategy of standing still to avoid collisions. Risk-horizon
discounting encourages conservative early behavior (survive long enough to accumulate
reward).

### 7. `force_gt_duration_us` warmup for RL exploration

The 10B may crash immediately in some scenes, giving GRPO no signal. The AlpaSim
warmup feeds ground-truth trajectories for the first N seconds, giving the model a stable
starting state before its own policy takes over. This ensures RL rollouts last long enough
to produce meaningful reward gradients.

---

## Compute Budget

All estimates are grounded in measured smoke-run data (`s0_45_smoke` on a
2×H100 launchpad box: 45 scenes, ~746s/scene, ~2.4 hr wall-clock) and in the
official Alpamayo 1.5 SFT/RL recipe configurations.

### AlpaSim topology constraint

AlpaSim only supports three topology configs — there is **no 4-GPU option**:

| Topology | GPUs | Concurrent rollouts | Use case |
|----------|------|---------------------|----------|
| `1gpu` | 1 | ~1–2 | Minimal testing |
| `2gpu` | 4 workers, 12 slots | 12 | Evaluation (proven config) |
| `8gpu_12rollouts` | 8 | 12+ (up to 64 with `8gpu_64rollouts`) | RL training at scale |

This means the jump from 2 GPUs to 8 GPUs is the only way to increase
parallelism — there is no middle ground.

### Per-phase cost estimates (revised — 10B bf16, no distillation)

| Phase | What burns GPU | Hardware | Est. wall time | Est. cost |
|-------|---------------|----------|---------------|-----------|
| **0: Baseline & gate** | NVFP4 gate (done on B300), FP8 gate (H100), container build | 1×H100 ($2.28/hr) | 4–8 hr | $10–18 |
| **1: RL (moderate)** | GRPO closed-loop rollouts on bf16 10B (1K samples, 15 epochs) | 2gpu: 2×H100 ($4.56/hr) | ~70 hr | ~$320 |
| **1: RL (aggressive)** | GRPO at larger scale on bf16 10B (5K samples) | 8gpu: 8×H100 ($18.24/hr) | ~90 hr | ~$1,640 |
| **2–3: Integration & FP8 optimization** | gRPC driver wiring, FP8 quantization, latency tuning on H100 | 1×H100 ($2.28/hr) | 10–20 hr | $23–46 |
| **4–5: Eval & iteration** | Full 916-scene closed-loop eval (measured: ~49 hr/run) | 2×H100 launchpad ($5.69/hr) | 49 hr × 3 full + 4 hr × 5 subset | ~$1,060 |
| **Overhead** | Setup/teardown, debugging, failed runs (15–25%) | — | — | $200–400 |

### Total compute budget

| Level | Total cost | What you get |
|-------|-----------|--------------|
| **Scraping by** | ~$1,500–2,000 | Baseline + moderate RL (bf16 10B) + 2 eval runs |
| **Serious attempt** | ~$2,500–3,500 | Moderate RL + 3 full evals + 5 subset evals |
| **Go all in** | ~$4,000–6,000 | Aggressive RL (5K+ scenes) + extensive iteration + 5–7 full eval runs |

**Planning number: ~$2,500–3,500.** The biggest lever is RL training scale —
start with the `2gpu` local-test config (~$320/run) and scale to `8gpu` only
if it shows promise. The second biggest lever is eval iteration — use 100-scene
subsets (~$30 each, ~4 hours) for tuning and full 916-scene runs (~$279 each)
only for milestone validation.

### Cheapest available GPU instances (Brev, June 2026)

For phases that don't need the full 56 TB disk or 96 GB VRAM of the launchpad
box, cheaper alternatives exist:

| Instance | GPUs | VRAM/GPU | $/hr | Disk | Best for |
|----------|------|----------|------|------|----------|
| hyperstack_A100_80G | 1×A100 | 80 GB | $1.62 | 850 GB | Teacher data generation |
| hyperstack_H100 | 1×H100 | 80 GB | $2.28 | 850 GB | SFT, integration, quantization |
| a100-80gb.2x (crusoe) | 2×A100 | 80 GB | $3.96 | 128 GB | SFT (2B fits on one GPU) |
| hyperstack_H100x2 | 2×H100 | 80 GB | $4.56 | 2 TB | SFT, moderate RL |
| **dmz.h100x2.pcie (launchpad)** | **2×H100** | **96 GB** | **$5.69** | **56 TB** | **Evaluation (proven config)** |
| denvr_A100_sxm4_80Gx8 | 8×A100 | 80 GB | $14.64 | 17 TB | Aggressive RL |
| hyperstack_H100x8 | 8×H100 | 80 GB | $18.24 | 7 TB | Aggressive RL |

The launchpad 2×H100 at $5.69/hr is uniquely good for evaluation — 96 GB
VRAM per GPU (vs 80 GB on all alternatives), 56 TB SSD (vs 2–17 TB elsewhere),
and a proven `2gpu` topology config with measured timing data.

### Eval cost is the dominant line item

Each full 916-scene evaluation run costs ~$279 on the launchpad box (measured:
~746s/scene, ~49 hr wall-clock at $5.69/hr). With 3 milestone evals plus 5
100-scene subset runs for iteration, eval alone costs ~$1,060 — more than
SFT and RL combined.

| Eval strategy | Runs | Cost | Notes |
|---------------|------|------|-------|
| Milestones only (baseline → distill → RL) | 3 full | ~$837 | Minimal feedback for tuning |
| Milestones + subset iteration | 3 full + 5×100-scene | ~$1,060 | **Recommended** — subsets ~$30 each |
| Full iteration | 5–7 full | ~$1,395–1,953 | More data, expensive |

### Splitting work across AWS credits and Brev

~$9K of AWS credits is available. AWS only sells H100/A100-80GB in **8-GPU boxes**
(`p5.48xlarge`, 8×H100 ~$55/hr; `p4de.24xlarge`, 8×A100-80GB ~$27–41/hr) — there is
no single-GPU H100/A100 SKU. So AWS is only economical for phases that genuinely
keep all 8 GPUs busy; anything 1–2 GPU wastes 6–7 GPUs at the box rate. Rule of
thumb: **fills 8 GPUs → AWS; needs 1–2 GPUs or rapid spin-up/teardown → Brev.**

| Phase | Where | Why |
|-------|-------|-----|
| **1: RL (aggressive, `8gpu`)** | **AWS** (`p5`, 8×H100) | You'd rent 8 GPUs anyway; credits absorb the dominant RL cost. |
| **4–5: Milestone full evals** | **AWS** (`p5`, or `p5e`/H200 for >80 GB VRAM) | High parallelism + big VRAM. Stage the 1.5 TB scene suite from S3/FSx for Lustre (no 56 TB local mount, but `p5` has ~30 TB local NVMe). |
| **0: FP8 quantization gate** | **Brev** (1×H100) | Single-GPU, quick iteration. FP8 is Hopper-native. |
| **1: RL (moderate, `2gpu`)** | **Brev** (2×H100) | Below the 8-GPU threshold. |
| **2–3: gRPC integration + FP8 latency/VRAM tuning** | **Brev** (1×H100) | Constant launch-and-kill; AWS Capacity Blocks make bursty work painful. Must test on H100 (eval hardware). |
| **Subset (100-scene) evals** | **Brev** (2×H100 launchpad) | Frequent, small iteration runs. |

**AWS caveats:**
- **Capacity Blocks for ML** — `p5` on-demand is frequently capacity-constrained.
  Reserve blocks 1–182 days ahead, paid upfront (credits apply). Plan AWS phases
  around reserved windows, not impulse launches.
- **Effective value** — AWS per-GPU (~$6.88/hr H100 on-demand, ~$3.9 via Capacity
  Block) is ~2–4× Brev's $2.28. Kept fully utilized on 8-GPU boxes, $9K of credits
  ≈ **$2,500–4,500 of Brev-equivalent compute** — enough to cover the RL + eval
  phases (the dominant line items) while Brev handles the agile single-GPU loop.

---

## Disk Budget

### Per-item breakdown

| Category | Item | Size | Notes |
|----------|------|------|-------|
| **Model weights** | Alpamayo 1.5-10B (bf16, for RL) | ~20.6 GB | 5 safetensors shards |
| | AlpaGym-converted checkpoint | ~20 GB | HF → AlpaGym format |
| | RL checkpoints (keep 3–5 steps) | ~60–100 GB | ~20 GB per checkpoint step |
| | FP8 quantized checkpoint (for Docker) | ~11 GB | Baked into image |
| | NVFP4 quantized checkpoint (gate artifact) | ~7.6 GB | Already generated on B300 |
| **Scene data** | Per NuRec scene (.usdz + camera MP4) | ~1.5–2.0 GB | From `PhysicalAI-Autonomous-Vehicles-NuRec` |
| | Full `public_2601` suite (916 scenes) | **~1.5 TB** | **Dominant disk cost** |
| **Rollout output** | Per scene, `render_video=true` | ~100–300 MB | Videos + telemetry + metrics |
| | Per scene, `render_video=false` | ~15–40 MB | Telemetry + metrics only |
| | Full 916-scene eval run (no video) | ~15–35 GB | Keep `render_video=false` for production |
| | Full 916-scene eval run (with video) | ~110–270 GB | Only for visual debugging |
| **Environment** | `uv` Python environment | ~15–25 GB | PyTorch, CUDA libs |
| | Docker images (NRE, base, driver) | ~10–15 GB | Built by AlpaSim wizard |
| | AlpaSim + AlpaGym + Alpamayo1.5 repos | ~20 MB | Source code only |
| **NuPlan track** (optional) | MTGS assets (15 shards) | ~456 GB | Only if running NuPlan track locally |

### Total disk by scenario

| Scenario | Disk needed | Key drivers |
|----------|-------------|-------------|
| **Getting started** (100-scene eval + dev) | ~300 GB | Models (45 GB) + env (50 GB) + 100 scenes (200 GB) + outputs (5 GB) |
| **Moderate** (full pipeline, moderate RL, 3 eval runs) | ~1.8–1.9 TB | Full 916 scenes (1.5 TB) + models (120 GB) + SFT data (80 GB) + env (50 GB) + outputs (50 GB) |
| **Serious** (aggressive RL, 7 eval runs, both tracks) | ~2.6–2.7 TB | Above + NuPlan assets (460 GB) + more checkpoints (200 GB) |

### Where to put things

| Disk | Mount | Best use | Notes |
|------|-------|----------|-------|
| System SSD | `/` (~107 GB free) | Environment, model weights, code | Fits ~50 GB env + ~45 GB models with room |
| Data SSD | `/data` (~934 GB free) | Scene data, rollout outputs, SFT data | Fits ~450 scenes; does NOT fit full 916-scene suite |
| Launchpad 56 TB | (remote box) | Full 916-scene eval runs | No disk concerns at all |

### Disk is the reason the launchpad box wins for evaluation

| Machine | Disk | Fits full 916-scene eval? | $/hr |
|---------|------|--------------------------|------|
| Current dev box (`/data` 934 GB) | ~1 TB | ❌ needs ~1.5 TB scenes | — |
| hyperstack_H100x2 | 2 TB | ❌ (scenes + outputs + env > 2 TB) | $4.56 |
| hyperstack_H100x8 | 7 TB | ✅ | $18.24 |
| **dmz.h100x2.pcie (launchpad)** | **56 TB** | ✅ effortlessly | $5.69 |

The launchpad box is more expensive per hour than hyperstack alternatives, but
its 56 TB SSD eliminates all storage concerns. For SFT and RL phases that
don't need all 916 scenes locally, cheaper instances with 850 GB–2 TB disks
are sufficient.

---

## Risk Register

| Risk | Impact | Mitigation |
|---|---|---|
| FP8 quantization degrades RL-tuned behavior | Self-correcting behavior lost in compression | Phase 0 gate proved PTQ is safe (NVFP4 +3.7%, FP8 will be less). Re-eval post-RL quantized model in AlpaSim before submitting. |
| FP8 too tight for 16 GiB VRAM with 2 concurrent rollouts | OOM during eval | Use `--quant_weight_only` (weights FP8, activations FP16). Fall back to INT8 weight-only (~6-7 GB). Reduce `num_context_frames`. |
| RL on bf16 10B is ~5× more expensive than 2B plan | Budget overrun | Start with `2gpu` moderate config (~$320). Scale to `8gpu` only if showing promise. Use AWS credits for 8-GPU phases. |
| RL doesn't converge or reward collapses | No improvement over baseline | Start with conservative scenes, use `force_gt` warmup, monitor KL/reward with trackio alerts. |
| Latency still over 0.1s after FP8 optimization | Submission disqualified | Reduce diffusion steps, use TensorRT, reduce image resolution, torch.compile. INT8 fallback. |
| Docker image over 40 GiB | Submission rejected | FP8 weights ~11 GB, minimal dependencies, multi-stage build. Well under 40 GiB. |
| AlpaSim local setup is complex | Can't smoke test locally | Follow `e2e_challenge/starter_kit/README.md` carefully, use `+e2e_challenge=dev` for minimal test. |

---

## References

- [AlpaSim GitHub](https://github.com/NVlabs/alpasim)
- [AlpaGym GitHub](https://github.com/NVlabs/alpagym)
- [Challenge HF Space](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026)
- [Alpamayo 1.5 Model Card](https://huggingface.co/nvidia/Alpamayo-1.5-10B)
- [Alpamayo 1.5 GitHub](https://github.com/NVlabs/alpamayo1.5)
- [Alpamayo Recipes (SFT + RL + Quantization)](https://github.com/NVlabs/alpamayo-recipes)
- [Alpamayo 1.5 Quantization Recipe](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo1_5_quant)
- [Alpagym Post-Training Blog](https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/)
- [Alpamayo Paper (arXiv:2511.00088)](https://arxiv.org/abs/2511.00088)
