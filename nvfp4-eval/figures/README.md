# Figures for NVFP4 gate writeup

| File | Description | Status |
|------|-------------|--------|
| `minade_scatter_100.png` | bf16 vs NVFP4 minADE (axes clipped 0–4m) | **Done** |
| `minade_scatter_100_full.png` | Same, full range incl. ~9.5m outliers | **Done** |
| `latency_comparison.png` | Grouped bar: bf16 vs NVFP4 mean ms/clip | **Done** |
| `gate_breakdown_100.png` | Both pass / both fail / asymmetric gate | **Done** |
| `delta_histogram_100.png` | NVFP4 − bf16 per-clip delta distribution | **Done** |
| `trajectory_best_improvement.png` | Pred vs GT overlay | **Done** |
| `trajectory_worst_regression.png` | Pred vs GT overlay | **Done** |

Regenerate stats figures: `python scripts/generate_figures.py` from repo root `nvfp4-eval/`.

Regenerate trajectory overlays:
```bash
source /path/to/alpamayo-recipes/recipes/alpamayo1_5_quant/run_env.sh
python scripts/plot_trajectory_figures.py --all-notable
```

Notable clips for trajectory figures (from `summary.json`):
- Best NVFP4 improvement: `fc70ddaa-ea3b-47d5-b51d-fe1a3049216c` (bf16 3.51m → NVFP4 0.51m)
- Worst NVFP4 regression: `7b026836-2af6-42ce-8485-8a1b52ba1fb0` (bf16 1.02m → NVFP4 3.72m)
