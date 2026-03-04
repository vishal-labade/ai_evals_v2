#!/usr/bin/env python3
"""
Plot PRI vs latency (V3)

Reads:  artifacts/v3/product_leaderboard_v3.csv
Writes: artifacts/v3/pri_vs_latency.png

Notes:
- Uses matplotlib (no seaborn).
- Does NOT set colors (defaults only).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--leaderboard", type=Path, default=Path("artifacts/v3/product_leaderboard_v3.csv"))
    p.add_argument("--out", type=Path, default=Path("artifacts/v3/pri_vs_latency.png"))
    p.add_argument("--label", choices=["profile", "model", "profile_model"], default="profile_model")
    p.add_argument("--top_k_labels", type=int, default=12, help="Annotate the top-K by PRI (to avoid clutter).")
    args = p.parse_args()

    df = pd.read_csv(args.leaderboard)

    required = {"latency_ms_mean", "pri_mean", "profile", "model", "temperature", "num_ctx"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns in leaderboard: {sorted(missing)}")

    # Keep valid rows
    df = df.dropna(subset=["latency_ms_mean", "pri_mean"]).copy()

    # X/Y
    x = df["latency_ms_mean"].astype(float)
    y = df["pri_mean"].astype(float)

    # Label text
    def mk_label(r) -> str:
        if args.label == "profile":
            return f"{r['profile']} (ctx={int(r['num_ctx'])})"
        if args.label == "model":
            return f"{r['model']} (ctx={int(r['num_ctx'])})"
        return f"{r['profile']} | {r['model']} (ctx={int(r['num_ctx'])})"

    df["_label"] = df.apply(mk_label, axis=1)

    # Plot
    plt.figure()
    plt.scatter(x, y)
    plt.xlabel("Mean latency (ms)")
    plt.ylabel("PRI (mean)")
    plt.title("Product Reliability Index (PRI) vs Latency")

    # Annotate top-K by PRI to reduce clutter
    top = df.sort_values("pri_mean", ascending=False).head(args.top_k_labels)
    for _, r in top.iterrows():
        plt.annotate(
            r["_label"],
            (float(r["latency_ms_mean"]), float(r["pri_mean"])),
            textcoords="offset points",
            xytext=(6, 4),
            ha="left",
            fontsize=8,
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(args.out, dpi=150)
    print(f"[ok] wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())