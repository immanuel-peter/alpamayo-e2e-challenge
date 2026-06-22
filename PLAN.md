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

## Strategy: Distill 10B → 2B, Then RL on the 2B

### Why this order

- **RL directly optimizes the deployed model.** No train/deploy gap. The self-correcting
  behavior that RL produces is shaped in the student's own parameter space — the exact
  model that runs at inference time.
- **Compute efficiency.** GRPO needs many closed-loop rollouts per update. A 2B model is
  ~5x cheaper per rollout than 10B, yielding 5x more rollouts for the same budget.
- **Distillation gives a strong starting point.** The 2B student inherits perception and
  basic driving behavior from the 10B teacher via imitation, so RL starts from a model
  that can already drive — not from scratch.
- **RL-then-distill loses the key property.** Distillation transfers input→output
  mappings, not closed-loop robustness. The self-correcting behavior learned through RL
  would be lost during compression.

### Why no CoC at inference

Chain-of-Causation (CoC) reasoning requires autoregressive VLM generation (~256 tokens,
~50-80ms) before the diffusion expert runs. On a 2B model at 0.1s budget, this leaves no
room for vision encoding, diffusion sampling, or multi-session contention.

The `last_component="traj_future"` path (used by Alpagym's RL recipe) skips VLM generation
entirely — the expert conditions on prefill-only KV cache. This is the deploy path.

**CoC is removed from the student but used as teacher signal during distillation.** The
student never generates CoC tokens, but is trained against teacher trajectories that
benefited from CoC reasoning. The reasoning quality is internalized into the student's
latent representations rather than explicitly generated.

Keeping CoC in the student would also:
- Create a train/deploy mismatch (distill with CoC, RL/deploy without)
- Split limited 2B capacity between a reasoning language head and trajectory expert
- Slow RL rollouts, reducing gradient quality

---

## Architecture

### Teacher (10B Alpamayo 1.5)

```
Camera images → Vision encoder → VLM prefill → CoC generation → Diffusion expert → trajectory*
                                                              ↑
                                                     reasoning-enriched KV cache
```

- Model: `nvidia/Alpamayo-1.5-10B` (gated, needs HF access approval)
- Dataset: `nvidia/PhysicalAI-Autonomous-Vehicles` (gated)
- Inference: `sample_trajectories_from_data_with_vlm_rollout` (full pipeline with CoC)
- VRAM: ~24 GB single-sample (H100 80GB)

### Student (2B distilled model)

```
Camera images → Vision encoder → VLM prefill → Diffusion expert → trajectory
                                              ↑
                                    prefill-only KV cache (no CoC generation)
```

- Architecture: 2B VLA (Cosmos-Reason backbone + diffusion expert, scaled down)
- Inference: `sample_trajectories_from_data` with `last_component="traj_future"`
- VRAM target: <8 GB (INT4 quantized) to leave headroom in 16 GiB budget
- Latency target: <50ms per `drive()` call

### Deployed model (2B + INT4 quantization, RL post-trained)

```
Camera images → Vision encoder → VLM prefill → Diffusion expert → trajectory
                                              ↑
                                    prefill-only KV cache, INT4 weights
```

- Same architecture and inference path as student
- INT4 quantized weights baked into Docker image (no network at runtime)
- RL post-trained for closed-loop robustness
- 2 concurrent rollouts per replica, 16 replicas across 4 GPUs

---

## Phases

### Phase 0: Baseline & Setup

**Goal:** Get the starter kit running and submitted to establish a floor score.

- [ ] Register at the [challenge HF Space](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026), get approved
- [ ] Build the starter driver container:
  ```bash
  docker build -f docs/starter_kit/Dockerfile -t alpasim-e2e-starter-driver:latest .
  ```
- [ ] Run local PAI smoke test against AlpaSim
- [ ] Submit the straight-line baseline to get a leaderboard score
- [ ] Authenticate with the challenge CLI (`auth-url` → `configure-token` → `me`)
- [ ] Verify ECR login works (`ecr-login`)

**Decision gate:** Before committing to distillation, smoke-test whether INT4 quantized
10B Alpamayo has sufficient perception quality to explore useful behaviors during RL. If
INT4-10B perception is good enough, the distillation step may be skippable — RL directly
on the quantized 10B would be simpler and preserve more capacity. The choice depends on
whether quantization degrades perception enough to prevent meaningful RL exploration.

- [ ] Load 10B Alpamayo with INT4 quantization
- [ ] Run inference on example clips from `test_inference.py`
- [ ] Compare trajectory quality (minADE) between bf16 and INT4
- [ ] If INT4 minADE < 1.0m and trajectories look reasonable → consider skipping distillation
- [ ] If INT4 perception is degraded → proceed with distillation to 2B

### Phase 1: Teacher Data Generation

**Goal:** Generate high-quality trajectory data from the 10B teacher (with CoC) to use as
distillation targets.

- [ ] Request access to `nvidia/Alpamayo-1.5-10B` and `nvidia/PhysicalAI-Autonomous-Vehicles` on HF
- [ ] Set up Alpamayo 1.5 environment (`uv sync --active`)
- [ ] Load 10B teacher in bf16 on H100/A100
- [ ] Run teacher inference on PhysicalAI-AV dataset clips using
      `sample_trajectories_from_data_with_vlm_rollout` (with CoC, `return_extra=True`)
- [ ] Generate paired data: `(images, ego_history, route) → trajectory`
- [ ] Store teacher outputs as a dataset for SFT distillation
- [ ] Also generate teacher outputs on the `traj_future` path (no CoC) for comparison —
      this tells us how much CoC reasoning actually contributes to trajectory quality

### Phase 2: Distillation (SFT 10B → 2B)

**Goal:** Train a 2B student that inherits the teacher's perception and driving behavior,
using the `traj_future` path (no CoC generation).

- [ ] Select a 2B-scale VLA architecture (Cosmos-Reason backbone + diffusion expert)
- [ ] Adapt the [Alpamayo 1.5 SFT recipe](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo1_5_sft)
      for the smaller architecture
- [ ] Train student to match teacher trajectories:
  - Input: camera images + ego history + route (same format as teacher)
  - Target: teacher's CoC-enhanced trajectory output
  - Inference path: `last_component="traj_future"` (prefill → expert, no generation)
- [ ] Evaluate student open-loop (minADE vs teacher)
- [ ] Evaluate student closed-loop (smoke test in AlpaSim) — does it survive at all?
- [ ] Verify student fits in 16 GiB VRAM at bf16
- [ ] Measure student inference latency (target: <50ms without quantization)

### Phase 3: gRPC Driver Integration

**Goal:** Wire the distilled 2B model into the gRPC driver container, replacing the
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

### Phase 4: Inference Optimization

**Goal:** Hit ≤0.1s latency and ≤16 GiB VRAM with the 2B model.

- [ ] **Quantization:** Apply INT4 quantization to the 2B model
  - Target: <5 GB weights, <8 GB total VRAM with KV cache + vision encoder
  - Test: does INT4 degrade perception enough to affect closed-loop driving?
- [ ] **Diffusion optimization:** Reduce inference steps
  - The `diffusion_kwargs` in the inference adapter supports `inference_step` and `int_method`
  - Find the minimum steps that maintain trajectory quality
- [ ] **Batch size:** Set `num_traj_samples=1`, `num_traj_sets=1` (single trajectory, no selection)
- [ ] **KV cache:** Enable `use_cache=True` for prefill (already default in Alpamayo)
- [ ] **Optional:** torch.compile on the model forward pass
- [ ] **Optional:** TensorRT export for the expert + vision encoder
- [ ] **Verify:** End-to-end `drive()` latency ≤0.1s with 2 concurrent sessions
- [ ] **Verify:** Peak VRAM ≤16 GiB with 2 concurrent sessions

### Phase 5: RL Post-Training (GRPO via Alpagym)

**Goal:** Directly optimize the 2B student for collision-free duration in closed-loop
simulation.

Reference: [Alpamayo 1.x RL recipe](https://github.com/NVlabs/alpamayo-recipes/tree/main/recipes/alpamayo1_x_rl)
and Alpagym's `AlpamayoR1InferenceModel` / `AlpamayoPolicy` infrastructure.

- [ ] Set up Alpagym environment with AlpaSim as the simulator backend
- [ ] Configure the 2B student as the policy model:
  - `load_inference_model`: load 2B with `attn_implementation="sdpa"`, `dtype=bfloat16`
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
- [ ] Compare: distilled-only vs distilled+RL on collision-free duration

### Phase 6: Containerization & Submission

**Goal:** Package the RL-tuned 2B model into a compliant Docker container and submit.

- [ ] Bake INT4-quantized RL-tuned weights into the Docker image
- [ ] Verify image size ≤40 GiB
- [ ] Verify read-only root filesystem works (model loads from baked-in weights, no writes)
- [ ] Verify no outbound network calls at runtime
- [ ] Run final local smoke test (PAI track)
- [ ] Run local smoke test (nuPlan track, if attempting)
- [ ] Push to ECR:
  ```bash
  uv run docs/competitor_cli/alpasim_challenge.py ecr-login
  docker tag <image>:<tag> 696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/<team_id>:<tag>
  docker push 696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/<team_id>:<tag>
  ```
- [ ] Submit:
  ```bash
  uv run docs/competitor_cli/alpasim_challenge.py submit --track pai \
    696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/<team_id>:<tag>
  ```
- [ ] Check status and leaderboard

### Phase 7: Iterate

- [ ] Analyze failure modes from leaderboard results (which scenes does it crash on?)
- [ ] If latency is the bottleneck: reduce diffusion steps, try smaller model, optimize inference
- [ ] If perception is the bottleneck: try higher-res input, more context frames, or INT8 instead of INT4
- [ ] If closed-loop robustness is the bottleneck: more RL training, adjust reward design
- [ ] If VRAM is the bottleneck: reduce context frames, reduce image resolution, or smaller model
- [ ] Try both tracks (PAI and nuPlan) — they may favor different configurations

---

## Key Design Decisions

### 1. Distill before RL, not after

RL on the 2B directly optimizes the deployed model for closed-loop survival. Distilling
after RL would lose the self-correcting behavior that RL produces.

### 2. Remove CoC from the student

CoC generation is too slow for 0.1s latency. The student uses the `traj_future` path
(prefill → expert, no VLM generation). Teacher trajectories produced with CoC serve as
distillation targets — the reasoning benefit is internalized into latent representations.

### 3. `traj_future` as the single inference path

Used for distillation, RL, and deployment. No train/deploy gap. Already the default in
Alpagym's RL recipe (`bundle.py`).

### 4. 2B instead of quantized 10B

A 2B model at INT4 (~1 GB weights) has ample VRAM headroom for 2 concurrent sessions.
The 10B at INT4 (~5 GB weights) is feasible but tighter, and a 2B is 5x cheaper per RL
rollout. The trade-off is perception capacity — validate with the Phase 0 decision gate.

### 5. RL reward = collision-free duration

Directly optimizes the competition metric. Progress-along-route as secondary reward
prevents the degenerate strategy of standing still to avoid collisions. Risk-horizon
discounting encourages conservative early behavior (survive long enough to accumulate
reward).

### 6. `force_gt_duration_us` warmup for RL exploration

The distilled 2B may crash immediately in some scenes, giving GRPO no signal. The AlpaSim
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

### Per-phase cost estimates

| Phase | What burns GPU | Hardware | Est. wall time | Est. cost |
|-------|---------------|----------|---------------|-----------|
| **0: Baseline & setup** | Container build, INT4 quantization gate | 1×H100 ($2.28/hr) | 4–8 hr | $10–18 |
| **1: Teacher data gen** | 10B open-loop inference (CoC + diffusion) on PAI clips | 1×A100 80GB ($1.62/hr) | 60–100 hr (50K clips) | $100–160 |
| **2: SFT distillation** | Training 2B student on teacher trajectories (100K samples, 3 epochs) | 2×H100 ($4.56/hr) | ~26 hr | ~$118 |
| **3–4: Integration & optimization** | gRPC driver wiring, quantization tuning, latency measurement | 1×H100 ($2.28/hr) | 10–20 hr | $23–46 |
| **5: RL (moderate)** | GRPO closed-loop rollouts (1K samples, 15 epochs, 180K rollouts) | 2gpu: 2×H100 ($4.56/hr) | ~14 hr | ~$64 |
| **5: RL (aggressive)** | GRPO at larger scale (5K samples, 900K rollouts) | 8gpu: 8×H100 ($18.24/hr) | ~18 hr | ~$328 |
| **6–7: Eval & iteration** | Full 916-scene closed-loop eval (measured: ~49 hr/run) | 2×H100 launchpad ($5.69/hr) | 49 hr × 3 full + 4 hr × 5 subset | ~$1,060 |
| **Overhead** | Setup/teardown, debugging, failed runs (15–25%) | — | — | $200–400 |

### Total compute budget

| Level | Total cost | What you get |
|-------|-----------|--------------|
| **Scraping by** | ~$1,000–1,500 | Baseline + minimal distillation + local-scale RL + 2 eval runs |
| **Serious attempt** | ~$1,800–2,500 | Full distillation + moderate RL + 3 full evals + 5 subset evals |
| **Go all in** | ~$3,500–5,500 | Aggressive RL (5K+ scenes) + extensive iteration + 5–7 full eval runs |

**Planning number: ~$2,000–3,000.** The biggest lever is RL training scale —
start with the `2gpu` local-test config (~$64/run) and scale to `8gpu` only
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

---

## Disk Budget

### Per-item breakdown

| Category | Item | Size | Notes |
|----------|------|------|-------|
| **Model weights** | Alpamayo 1.5-10B (teacher) | ~20.6 GB | 5 safetensors shards |
| | Cosmos-Reason2-8B (backbone, gated) | ~16 GB | Required by Alpamayo 1.5 |
| | Distilled 2B student (bf16) | ~4–8 GB | Your model |
| | 2B student INT4 (for Docker) | ~1–2 GB | Baked into image |
| | AlpaGym-converted checkpoint | ~20 GB | HF → AlpaGym format |
| | RL checkpoints (keep 3–5 steps) | ~60–100 GB | ~20 GB per checkpoint step |
| | Lingo-Judge model (optional) | ~1–2 GB | For reasoning reward only |
| **Scene data** | Per NuRec scene (.usdz + camera MP4) | ~1.5–2.0 GB | From `PhysicalAI-Autonomous-Vehicles-NuRec` |
| | Full `public_2601` suite (916 scenes) | **~1.5 TB** | **Dominant disk cost** |
| **Rollout output** | Per scene, `render_video=true` | ~100–300 MB | Videos + telemetry + metrics |
| | Per scene, `render_video=false` | ~15–40 MB | Telemetry + metrics only |
| | Full 916-scene eval run (no video) | ~15–35 GB | Keep `render_video=false` for production |
| | Full 916-scene eval run (with video) | ~110–270 GB | Only for visual debugging |
| **SFT training data** | PAI dataset chunks (per 20 chunks) | ~3–5 GB | 4 cameras × ~40 MB/chunk |
| | Teacher trajectories (50K clips) | ~50–100 GB | Trajectories + metadata, no raw video |
| | LingoQA scenery data | ~5–10 GB | 148K QA pairs + 17.5K images |
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
| INT4 quantization degrades perception too much | Model can't perceive obstacles, crashes immediately | Test early (Phase 0 gate). Fall back to INT8 or larger model. |
| 2B model lacks capacity for good perception | Distilled student is too weak for RL exploration | Use 3B or 4B instead. Or RL on quantized 10B. |
| RL doesn't converge or reward collapses | No improvement over distilled baseline | Start with conservative scenes, use `force_gt` warmup, monitor KL/reward with trackio alerts. |
| Latency still over 0.1s after optimization | Submission disqualified | Reduce diffusion steps, use TensorRT, reduce image resolution, or smaller model. |
| Docker image over 40 GiB | Submission rejected | Use INT4 weights (~1-5 GB), minimal dependencies, multi-stage build. |
| AlpaSim local setup is complex | Can't smoke test locally | Follow `docs/starter_kit/README.md` carefully, use `+e2e_challenge=dev` for minimal test. |

---

## References

- [AlpaSim GitHub](https://github.com/NVlabs/alpasim)
- [AlpaGym GitHub](https://github.com/NVlabs/alpagym)
- [Challenge HF Space](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026)
- [Alpamayo 1.5 Model Card](https://huggingface.co/nvidia/Alpamayo-1.5-10B)
- [Alpamayo 1.5 GitHub](https://github.com/NVlabs/alpamayo1.5)
- [Alpamayo Recipes (SFT + RL)](https://github.com/NVlabs/alpamayo-recipes)
- [Alpagym Post-Training Blog](https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/)
- [Alpamayo Paper (arXiv:2511.00088)](https://arxiv.org/abs/2511.00088)
