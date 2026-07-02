#!/usr/bin/env python3
"""Merge bf16 and NVFP4 per-clip CSVs into per_clip_100.csv."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(path: Path) -> dict[str, dict[str, str | float | int]]:
    with path.open() as f:
        return {row["clip_id"]: row for row in csv.DictReader(f)}


def main() -> None:
    bf16_path = RESULTS / "per_clip_bf16_100.csv"
    nvfp4_path = RESULTS / "per_clip_nvfp4_100.csv"
    out_path = RESULTS / "per_clip_100.csv"

    bf16 = load(bf16_path)
    nvfp4 = load(nvfp4_path)
    common = sorted(set(bf16) & set(nvfp4), key=lambda c: int(bf16[c]["clip_index"]))

    if len(common) != 100:
        raise SystemExit(f"Expected 100 matched clips, got {len(common)}")

    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "clip_index",
                "clip_id",
                "bf16_minade_m",
                "bf16_time_ms",
                "nvfp4_minade_m",
                "nvfp4_time_ms",
                "delta_minade_m",
            ],
        )
        writer.writeheader()
        for clip_id in common:
            b = bf16[clip_id]
            n = nvfp4[clip_id]
            b_min = float(b["minade_m"])
            n_min = float(n["minade_m"])
            writer.writerow(
                {
                    "clip_index": b["clip_index"],
                    "clip_id": clip_id,
                    "bf16_minade_m": f"{b_min:.4f}",
                    "bf16_time_ms": b["time_ms"],
                    "nvfp4_minade_m": f"{n_min:.4f}",
                    "nvfp4_time_ms": n["time_ms"],
                    "delta_minade_m": f"{n_min - b_min:.4f}",
                }
            )

    print(f"Wrote {len(common)} rows to {out_path}")


if __name__ == "__main__":
    main()
