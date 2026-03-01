#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


def _ensure_cols(df: pd.DataFrame, cols: list[str], name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} missing required columns: {missing}. Have: {list(df.columns)}")


def plot_context_cliff(ctx_df: pd.DataFrame, out_dir: Path) -> None:
    # Expect qwen14b_ctx_{1k,2k,4k,6k,8k} cell_ids and num_ctx, mcs_auc, drop_turn_mean
    _ensure_cols(ctx_df, ["cell_id", "num_ctx", "mcs_auc"], "ctx_cliff_run_csv")

    df = ctx_df.copy()
    df = df[df["cell_id"].astype(str).str.contains("qwen14b_ctx_", na=False)]
    if df.empty:
        raise ValueError("No qwen14b_ctx_* rows found in ctx cliff CSV.")

    df["num_ctx"] = pd.to_numeric(df["num_ctx"], errors="coerce")
    df["mcs_auc"] = pd.to_numeric(df["mcs_auc"], errors="coerce")
    if "drop_turn_mean" in df.columns:
        df["drop_turn_mean"] = pd.to_numeric(df["drop_turn_mean"], errors="coerce")

    df = df.sort_values("num_ctx")

    plt.figure()
    plt.plot(df["num_ctx"], df["mcs_auc"], marker="o")
    plt.xlabel("num_ctx")
    plt.ylabel("MCS AUC")
    plt.title("Context Cliff (qwen14b): MCS AUC vs num_ctx")
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    out_path = out_dir / "context_cliff_qwen14b_mcs_auc.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()

    if "drop_turn_mean" in df.columns and df["drop_turn_mean"].notna().any():
        plt.figure()
        plt.plot(df["num_ctx"], df["drop_turn_mean"], marker="o")
        plt.xlabel("num_ctx")
        plt.ylabel("drop_turn_mean")
        plt.title("Context Cliff (qwen14b): drop_turn_mean vs num_ctx")
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        out_path = out_dir / "context_cliff_qwen14b_drop_turn_mean.png"
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()


def plot_fixed_ctx_model_size(beh_df: pd.DataFrame, out_dir: Path) -> None:
    _ensure_cols(beh_df, ["profile", "num_ctx", "mcs_auc"], "behavioral_run_csv")
    df = beh_df.copy()

    df["num_ctx"] = pd.to_numeric(df["num_ctx"], errors="coerce")
    df["mcs_auc"] = pd.to_numeric(df["mcs_auc"], errors="coerce")

    # fixed ctx=2048
    df = df[df["num_ctx"] == 2048]
    if df.empty:
        raise ValueError("No rows with num_ctx==2048 found in behavioral run CSV.")

    # pick a canonical set if present
    wanted = ["phi_mini_t0", "qwen_3b_t0", "qwen_14b_t0", "gpt_20b_t0", "mixtral_moe_7b_t0", "mistral_12b_t0", "gemma_9b_t0", "llama_8b_t0"]
    df["profile"] = df["profile"].astype(str)
    sub = df[df["profile"].isin(wanted)]
    if sub.empty:
        sub = df

    # collapse duplicates per profile by mean
    sub = sub.groupby("profile", as_index=False).agg(mcs_auc=("mcs_auc", "mean"))

    plt.figure()
    plt.bar(sub["profile"], sub["mcs_auc"])
    plt.xlabel("model profile")
    plt.ylabel("MCS AUC @ 2K")
    plt.title("Fixed Context (2K): MCS AUC by Model")
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
    out_path = out_dir / "model_size_fixed_ctx_mcs_auc_2k.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def plot_tradeoff(beh_df: pd.DataFrame, out_dir: Path) -> None:
    # This plot needs a behavioral metric + memory metric.
    # We will use mcs_auc for memory. For behavior, if "bri" exists in enriched CSV, use it; else skip.
    if "bri" not in beh_df.columns:
        # still produce a simple comparison figure using only mcs_auc, so pipeline never breaks
        df = beh_df.copy()
        _ensure_cols(df, ["cell_id", "mcs_auc"], "behavioral_run_csv")
        df["mcs_auc"] = pd.to_numeric(df["mcs_auc"], errors="coerce")
        df = df[df["cell_id"].astype(str).str.contains("tradeoff", na=False)]
        if df.empty:
            return
        df = df.groupby("cell_id", as_index=False).agg(mcs_auc=("mcs_auc", "mean"))
        plt.figure()
        plt.bar(df["cell_id"], df["mcs_auc"])
        plt.xlabel("tradeoff cell_id")
        plt.ylabel("MCS AUC")
        plt.title("Tradeoff Cells: MCS AUC")
        plt.xticks(rotation=45, ha="right")
        plt.ylim(0, 1.05)
        plt.grid(True, axis="y", linestyle="--", linewidth=0.5)
        out_path = out_dir / "tradeoff_mcs_only.png"
        plt.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close()
        return

    df = beh_df.copy()
    _ensure_cols(df, ["cell_id", "mcs_auc", "bri"], "behavioral_run_csv")
    df["mcs_auc"] = pd.to_numeric(df["mcs_auc"], errors="coerce")
    df["bri"] = pd.to_numeric(df["bri"], errors="coerce")

    # tradeoff cells are often: qwen14b_ctx_6k and gpt20b_ctx_2k (or similar)
    df = df[df["cell_id"].astype(str).isin(["qwen14b_ctx_6k", "gpt20b_ctx_2k"]) | df["cell_id"].astype(str).str.contains("tradeoff", na=False)]
    if df.empty:
        return

    df = df.groupby("cell_id", as_index=False).agg(bri=("bri", "mean"), mcs_auc=("mcs_auc", "mean"))

    plt.figure()
    plt.scatter(df["bri"], df["mcs_auc"])
    for _, r in df.iterrows():
        plt.annotate(str(r["cell_id"]), (r["bri"], r["mcs_auc"]))
    plt.xlabel("Behavior (BRI)")
    plt.ylabel("Memory (MCS AUC)")
    plt.title("Tradeoff: Behavior vs Memory")
    plt.ylim(0, 1.05)
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)
    out_path = out_dir / "tradeoff_behavior_vs_memory.png"
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--behavioral_run_csv", required=True)
    ap.add_argument("--ctx_cliff_run_csv", required=True)
    ap.add_argument("--out_dir", default="artifacts/figures")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    beh = pd.read_csv(args.behavioral_run_csv)
    ctx = pd.read_csv(args.ctx_cliff_run_csv)

    # Context cliff plots
    plot_context_cliff(ctx, out_dir)

    # Fixed ctx model size plot (uses behavioral enriched file)
    plot_fixed_ctx_model_size(beh, out_dir)

    # Tradeoff plot (only if bri present)
    plot_tradeoff(beh, out_dir)

    print(f"[plots] wrote figures to: {out_dir}")


if __name__ == "__main__":
    main()