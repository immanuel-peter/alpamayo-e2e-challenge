#!/usr/bin/env python3
"""Render pred vs GT trajectory overlays for writeup figures."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import matplotlib.pyplot as plt
import modelopt.torch.opt as mto
import numpy as np
import torch
from alpamayo1_5 import helper
from alpamayo1_5.load_physical_aiavdataset import load_physical_aiavdataset
from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "figures"
SUMMARY = ROOT / "results" / "summary.json"

T0_US = 5_100_000
NUM_TRAJ_SAMPLES = 6
MAX_GEN = 256
TOP_P = 0.98
TEMP = 0.6
SEED = 42
BF16_CKPT = "nvidia/Alpamayo-1.5-10B"
NVFP4_CKPT = "/home/shadeform/alpamayo-recipes/recipes/alpamayo1_5_quant/outputs/alpamayo1.5_nvfp4_calib100"


@torch.inference_mode()
def infer_clip(ckpt: str, clip_id: str, device: str = "cuda") -> dict:
    mto.enable_huggingface_checkpointing()
    model = Alpamayo1_5.from_pretrained(ckpt, dtype=torch.float16).to(
        device=device, dtype=torch.float16
    )
    model.eval()
    processor = helper.get_processor(model.tokenizer)

    data = load_physical_aiavdataset(clip_id, t0_us=T0_US)
    messages = helper.create_message(
        data["image_frames"].flatten(0, 1), camera_indices=data["camera_indices"]
    )
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {
        "tokenized_data": inputs,
        "ego_history_xyz": data["ego_history_xyz"],
        "ego_history_rot": data["ego_history_rot"],
    }
    model_inputs = helper.to_device(model_inputs, device)

    gt_xy = data["ego_future_xyz"].cpu()[0, 0, :, :2].numpy()
    hist_xy = data["ego_history_xyz"].cpu()[0, 0, :, :2].numpy()

    del data, messages, inputs
    gc.collect()
    torch.cuda.synchronize()

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    with torch.autocast("cuda", dtype=torch.float16):
        pred_xyz, _ = model.sample_trajectories_from_data_with_vlm_rollout(
            data=model_inputs,
            top_p=TOP_P,
            temperature=TEMP,
            num_traj_samples=NUM_TRAJ_SAMPLES,
            max_generation_length=MAX_GEN,
        )

    pred_xy = pred_xyz.detach().cpu().numpy()[0, 0, :, :, :2]  # (S,T,2)
    d = np.linalg.norm(pred_xy - gt_xy[None, :, :], axis=-1)
    ade = d.mean(axis=-1)
    best_idx = int(ade.argmin())

    del model, processor, model_inputs, pred_xyz
    gc.collect()
    torch.cuda.empty_cache()

    return {
        "gt_xy": gt_xy,
        "hist_xy": hist_xy,
        "pred_xy": pred_xy,
        "best_idx": best_idx,
        "minade": float(ade.min()),
    }


def plot_overlay(
    *,
    clip_id: str,
    label: str,
    bf16: dict,
    nvfp4: dict,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))

    gt = bf16["gt_xy"]
    hist = bf16["hist_xy"]

    ax.plot(hist[:, 0], hist[:, 1], color="#94a3b8", lw=1.5, ls="--", label="history")
    ax.plot(gt[:, 0], gt[:, 1], color="black", lw=2.5, label="ground truth")

    for i, (res, color, name) in enumerate(
        [
            (bf16, "#64748b", f"bf16 best (minADE={bf16['minade']:.2f}m)"),
            (nvfp4, "#2563eb", f"NVFP4 best (minADE={nvfp4['minade']:.2f}m)"),
        ]
    ):
        pred = res["pred_xy"]
        # faint alternate samples
        for s in range(pred.shape[0]):
            if s == res["best_idx"]:
                continue
            ax.plot(pred[s, :, 0], pred[s, :, 1], color=color, alpha=0.15, lw=0.8)
        best = pred[res["best_idx"]]
        ax.plot(best[:, 0], best[:, 1], color=color, lw=2, label=name)

    ax.scatter([gt[0, 0]], [gt[0, 1]], c="black", s=40, zorder=5)
    ax.scatter([gt[-1, 0]], [gt[-1, 1]], c="black", s=60, marker="*", zorder=5)

    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(f"{label}\n{clip_id[:8]}…")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip-id", action="append", dest="clip_ids")
    ap.add_argument("--all-notable", action="store_true", help="Plot clips from summary.json")
    args = ap.parse_args()

    FIGURES.mkdir(parents=True, exist_ok=True)
    clips: list[tuple[str, str, str]] = []

    if args.all_notable or not args.clip_ids:
        summary = json.loads(SUMMARY.read_text())
        n = summary["primary_eval"]["notable_clips"]
        clips = [
            (n["best_nvfp4_improvement"]["clip_id"], "best_improvement", "trajectory_best_improvement.png"),
            (n["worst_nvfp4_regression"]["clip_id"], "worst_regression", "trajectory_worst_regression.png"),
        ]
    else:
        for cid in args.clip_ids:
            clips.append((cid, cid[:8], f"trajectory_{cid[:8]}.png"))

    for clip_id, label, filename in clips:
        print(f"=== {label}: {clip_id} ===")
        print("  bf16 inference...")
        bf16 = infer_clip(BF16_CKPT, clip_id)
        print(f"    minADE={bf16['minade']:.3f}m")
        print("  NVFP4 inference...")
        nvfp4 = infer_clip(NVFP4_CKPT, clip_id)
        print(f"    minADE={nvfp4['minade']:.3f}m")
        out = FIGURES / filename
        plot_overlay(clip_id=clip_id, label=label.replace("_", " ").title(), bf16=bf16, nvfp4=nvfp4, out_path=out)
        print(f"  wrote {out}")


if __name__ == "__main__":
    main()
