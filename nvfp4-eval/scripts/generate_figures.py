#!/usr/bin/env python3
"""Generate writeup figures from per_clip_100.csv and summary.json."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
CSV_PATH = RESULTS / "per_clip_100.csv"
GATE_M = 1.0


def load_rows() -> list[dict]:
    with CSV_PATH.open() as f:
        return list(csv.DictReader(f))


def plot_scatter(rows: list[dict]) -> None:
    b = np.array([float(r["bf16_minade_m"]) for r in rows])
    n = np.array([float(r["nvfp4_minade_m"]) for r in rows])
    lim = max(b.max(), n.max()) * 1.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(b, n, alpha=0.65, s=36, edgecolors="none", c="#2563eb")
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5, label="y = x")
    ax.axhline(GATE_M, color="#dc2626", ls=":", lw=1, alpha=0.7, label=f"gate {GATE_M}m")
    ax.axvline(GATE_M, color="#dc2626", ls=":", lw=1, alpha=0.7)
    ax.set_xlim(0, min(lim, 4.0))
    ax.set_ylim(0, min(lim, 4.0))
    ax.set_xlabel("bf16 minADE (m)")
    ax.set_ylabel("NVFP4 minADE (m)")
    ax.set_title("Per-clip minADE: bf16 vs NVFP4 (N=100)")
    r = np.corrcoef(b, n)[0, 1]
    ax.text(0.04, 0.96, f"Pearson r = {r:.2f}\nmean Δ = {(n - b).mean():+.3f} m",
            transform=ax.transAxes, va="top", fontsize=10,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    ax.legend(loc="lower right", fontsize=9)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIGURES / "minade_scatter_100.png", dpi=150)
    plt.close(fig)


def plot_scatter_full(rows: list[dict]) -> None:
    """Scatter without axis clip — shows heavy tail outliers."""
    b = np.array([float(r["bf16_minade_m"]) for r in rows])
    n = np.array([float(r["nvfp4_minade_m"]) for r in rows])
    lim = max(b.max(), n.max()) * 1.05

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(b, n, alpha=0.65, s=36, edgecolors="none", c="#2563eb")
    ax.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.5)
    ax.set_xlabel("bf16 minADE (m)")
    ax.set_ylabel("NVFP4 minADE (m)")
    ax.set_title("Per-clip minADE (full range, N=100)")
    fig.tight_layout()
    fig.savefig(FIGURES / "minade_scatter_100_full.png", dpi=150)
    plt.close(fig)


def plot_latency(summary: dict) -> None:
    p = summary["primary_eval"]
    labels = ["bf16", "NVFP4"]
    means = [p["bf16"]["avg_time_ms"], p["nvfp4"]["avg_time_ms"]]
    colors = ["#64748b", "#2563eb"]

    fig, ax = plt.subplots(figsize=(4, 4))
    bars = ax.bar(labels, means, color=colors, width=0.5)
    ax.set_ylabel("Mean eval time (ms / clip)")
    ax.set_title("Inference latency (100 clips, B300)")
    for bar, val in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 80,
                f"{val:.0f} ms", ha="center", fontsize=10)
    ratio = p["delta"]["latency_ratio"]
    ax.text(0.5, 0.95, f"NVFP4 / bf16 = {ratio:.2f}×", transform=ax.transAxes,
            ha="center", va="top", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIGURES / "latency_comparison.png", dpi=150)
    plt.close(fig)


def plot_gate_breakdown(rows: list[dict]) -> None:
    b = np.array([float(r["bf16_minade_m"]) for r in rows])
    n = np.array([float(r["nvfp4_minade_m"]) for r in rows])
    both_pass = ((b < GATE_M) & (n < GATE_M)).sum()
    both_fail = ((b >= GATE_M) & (n >= GATE_M)).sum()
    bf16_only = ((b < GATE_M) & (n >= GATE_M)).sum()
    nvfp4_only = ((b >= GATE_M) & (n < GATE_M)).sum()

    labels = ["Both pass", "Both fail", "bf16 pass\nNVFP4 fail", "bf16 fail\nNVFP4 pass"]
    counts = [both_pass, both_fail, bf16_only, nvfp4_only]
    colors = ["#16a34a", "#dc2626", "#f59e0b", "#8b5cf6"]

    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(labels, counts, color=colors, width=0.55)
    ax.set_ylabel("Clips (of 100)")
    ax.set_title(f"Per-clip gate breakdown (< {GATE_M}m minADE)")
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                str(val), ha="center", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGURES / "gate_breakdown_100.png", dpi=150)
    plt.close(fig)


def plot_delta_hist(rows: list[dict]) -> None:
    d = np.array([float(r["delta_minade_m"]) for r in rows])
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.hist(d, bins=20, color="#2563eb", alpha=0.75, edgecolor="white")
    ax.axvline(0, color="k", lw=1)
    ax.axvline(d.mean(), color="#dc2626", ls="--", lw=1, label=f"mean = {d.mean():+.3f} m")
    ax.set_xlabel("NVFP4 − bf16 minADE (m)")
    ax.set_ylabel("Clip count")
    ax.set_title("Per-clip quantization delta (N=100)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES / "delta_histogram_100.png", dpi=150)
    plt.close(fig)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = load_rows()
    summary = json.loads((RESULTS / "summary.json").read_text())

    plot_scatter(rows)
    plot_scatter_full(rows)
    plot_latency(summary)
    plot_gate_breakdown(rows)
    plot_delta_hist(rows)
    print(f"Wrote figures to {FIGURES}/")


if __name__ == "__main__":
    main()
