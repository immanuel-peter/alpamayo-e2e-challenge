#!/usr/bin/env python3
"""Extract per-clip minADE rows from alpamayo1_5_quant eval.py logs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

ROW_RE = re.compile(
    r"\[(?P<idx>\d+)/(?P<total>\d+)\] clip_id=(?P<clip_id>\S+) "
    r"minADE=(?P<minade>[0-9.]+)m time=(?P<time_ms>[0-9.]+)ms"
)


def parse_log(path: Path) -> list[dict[str, str | float | int]]:
    rows: list[dict[str, str | float | int]] = []
    for line in path.read_text().splitlines():
        m = ROW_RE.search(line)
        if not m:
            continue
        rows.append(
            {
                "clip_index": int(m.group("idx")),
                "total_clips": int(m.group("total")),
                "clip_id": m.group("clip_id"),
                "minade_m": float(m.group("minade")),
                "time_ms": float(m.group("time_ms")),
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("log", type=Path, help="eval.py log file")
    ap.add_argument("-o", "--output", type=Path, required=True, help="output CSV path")
    args = ap.parse_args()

    rows = parse_log(args.log)
    if not rows:
        raise SystemExit(f"No per-clip rows found in {args.log} (need --print_every 1?)")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["clip_index", "total_clips", "clip_id", "minade_m", "time_ms"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
