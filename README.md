# AlpaSim E2E Closed-Loop Challenge

Personal entry for the [NVIDIA AlpaSim E2E Closed-Loop Challenge 2026](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026), launched at CVPR 2026.

## What this is

AlpaSim runs your driving policy in a closed-loop simulation — every trajectory your model outputs actually moves the car, updates the environment, and feeds back as the next observation. The evaluation metric is **collision-free driving duration** across a set of held-out real-world reconstructed scenes. Small errors compound; you can't cheat by replaying ground truth.

The submission contract is a **Docker container** that implements a gRPC server (`egodriver.EgodriverService`). NVIDIA's evaluator connects to your container and calls it on every timestep:

| RPC | What it sends you |
|-----|-------------------|
| `submit_image_observation` | Camera frames from the simulator |
| `submit_egomotion_observation` | Ego pose history (where the car has been) |
| `submit_route` | Route / destination |
| `drive` | Trigger — return the next trajectory |

Your `drive()` response is a list of future `(x, y, z, quaternion, timestamp)` poses. The simulator executes that trajectory, renders the next frame, and calls `drive()` again.

## Two tracks

- **PAI (Physical AI AV)**: NuRec-compatible scenes reconstructed from real-world driving data
- **nuPlan**: Structured scenarios with MTGS neural rendering

Both use the same container interface. Use `--track pai` or `--track nuplan` when submitting.

## Hard constraints

| Constraint | Limit |
|------------|-------|
| `drive()` average response time | ≤ 0.1 seconds |
| VRAM per replica | ≤ 16 GiB |
| Image size | ≤ 40 GiB |
| Outbound network at runtime | blocked |
| Root filesystem | read-only |
| Writable scratch | `/tmp` (2 GiB), `/run` (64 MiB) |

Evaluation runs 16 replicas of the submitted image across 4 GPUs, with 2 concurrent rollouts per replica.

## Approach

The plan is to wire **Alpamayo 1.5** (NVIDIA's 10B VLA model) into the gRPC driver interface, then use **Alpagym** (GRPO-based closed-loop RL) to post-train the model specifically to maximize collision-free duration in simulation before containerizing.

The baseline starter kit drives straight at 5 m/s — anything that actually perceives the scene and follows a route should beat it substantially.

Key engineering problems to solve:
- **Latency**: Alpamayo 1.5 at full precision is too slow for 0.1s. Need INT4/INT8 quantization and careful inference optimization.
- **VRAM**: 10B params at 16 GiB is tight with quantization; need to measure carefully.
- **Session state**: The gRPC interface is stateful per session — need to buffer observations correctly across calls.
- **Reward design** (Alpagym): default rewards are progress + safety; worth experimenting with risk-horizon discounting.

## Repo layout

```
e2e_challenge/               # Official challenge artifacts from NVlabs/alpasim@e2e_challenge
  README.md                  # Challenge overview, submission contract, constraints
  starter_kit/               # Minimal working driver container (straight-line baseline)
    driver.py                # gRPC servicer implementation to copy and modify
    Dockerfile
    run_local_container.sh
  competitor_cli/            # CLI for auth, ECR push, submit, leaderboard
    alpasim_challenge.py
    README.md
src/grpc/                    # Official alpasim_grpc package used by starter images
```

## Quickstart (local smoke test)

```bash
# Build the starter driver
docker build -f e2e_challenge/starter_kit/Dockerfile -t alpasim-e2e-starter-driver:latest .

# Start the driver container
e2e_challenge/starter_kit/run_local_container.sh

# In another terminal — run the PAI smoke test (requires AlpaSim installed)
source setup_local_env.sh
ALPASIM_DRIVER_HOST=localhost ALPASIM_DRIVER_PORT=6789 \
uv run alpasim_wizard +e2e_challenge=dev \
  wizard.log_dir=./runs/e2e_challenge_smoke

# Results at:
# ./runs/e2e_challenge_smoke/aggregate/results-summary.json
```

## Submission

```bash
# Authenticate (tokens expire after 12 hours)
uv run e2e_challenge/competitor_cli/alpasim_challenge.py auth-url
uv run e2e_challenge/competitor_cli/alpasim_challenge.py configure-token

# Log in to ECR and push your image
uv run e2e_challenge/competitor_cli/alpasim_challenge.py ecr-login
docker tag <your-image>:<tag> 696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/<team_id>:<tag>
docker push 696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/<team_id>:<tag>

# Submit (use pai or nuplan)
uv run e2e_challenge/competitor_cli/alpasim_challenge.py submit --track pai \
  696254625193.dkr.ecr.us-east-1.amazonaws.com/teams/<team_id>:<tag>

# Check status / leaderboard
uv run e2e_challenge/competitor_cli/alpasim_challenge.py status <submission_id>
uv run e2e_challenge/competitor_cli/alpasim_challenge.py leaderboard --track pai
```

## References

- [AlpaSim GitHub](https://github.com/NVlabs/alpasim)
- [AlpaGym GitHub](https://github.com/NVlabs/alpagym)
- [Challenge HuggingFace Space](https://huggingface.co/spaces/nvidia/AlpasimE2EClosedLoopChallenge2026)
- [Alpagym post-training blog](https://developer.nvidia.com/blog/how-to-post-train-autonomous-vehicle-models-in-closed-loop-with-nvidia-alpamayo/)
- [Alpamayo 2 Super announcement](https://huggingface.co/blog/drmapavone/nvidia-alpamayo-2)
- [nvidia/Alpamayo-1.5-10B on HuggingFace](https://huggingface.co/nvidia/Alpamayo-1.5-10B)
- [Alpamayo Recipes GitHub](https://github.com/NVlabs/alpamayo-recipes)
