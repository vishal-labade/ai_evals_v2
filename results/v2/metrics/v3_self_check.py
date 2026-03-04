#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", type=Path, required=True)
    args = p.parse_args()

    outdir = args.outdir
    turn = outdir / "metrics_turn_v3.csv"
    run = outdir / "metrics_run_v3.csv"
    lb = outdir / "product_leaderboard_v3.csv"
    sc = outdir / "product_scorecard_v3.json"

    for f in [turn, run, lb, sc]:
        if not f.exists():
            raise SystemExit(f"[error] missing: {f}")

    df = pd.read_csv(lb)
    required = ["pri_mean", "ttft_ms_mean", "prompt_tps_mean", "decode_tps_mean", "prefill_time_share_mean"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise SystemExit(f"[error] leaderboard missing columns: {missing}")

    with sc.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if "leaderboard_top10" not in obj:
        raise SystemExit("[error] scorecard missing leaderboard_top10")

    print("[ok] V3 artifacts look good.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())