#!/usr/bin/env python3
"""
Build a leaderboard with confidence intervals from scored CSVs.

Usage:
  python scripts/add_confidence_intervals.py \
    --inputs "results/scored/prod_base_full_*.csv" \
    --out_csv results/leaderboards/base_overall_ci_20260222.csv \
    --out_md  results/leaderboards/base_overall_ci_20260222.md

Notes:
- Reports NOT_IN_CONTEXT in two ways:
  1) NIC_global  = mean(not_in_context_violation) over ALL prompts (violations / total prompts)
  2) NIC_cond    = mean(not_in_context_violation) over NIC prompts only (violations / NIC prompts)
- Also reports nic_k / nic_n to make the denominator obvious.
"""

from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    # try partial matches
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None


def wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """Wilson score interval for binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = k / n
    denom = 1.0 + (z * z) / n
    center = (phat + (z * z) / (2 * n)) / denom
    half = (z * math.sqrt((phat * (1 - phat) + (z * z) / (4 * n)) / n)) / denom
    lo = max(0.0, center - half)
    hi = min(1.0, center + half)
    return lo, hi


def bootstrap_ci_mean(
    x: np.ndarray, iters: int = 2000, alpha: float = 0.05, seed: int = 0
) -> Tuple[float, float]:
    """Bootstrap percentile CI for mean."""
    rng = np.random.default_rng(seed)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return (float("nan"), float("nan"))
    n = len(x)
    means = np.empty(iters, dtype=float)
    for i in range(iters):
        samp = rng.choice(x, size=n, replace=True)
        means[i] = float(np.mean(samp))
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    return lo, hi


def _pct(x: float) -> float:
    return 100.0 * x if pd.notnull(x) else float("nan")


def _to_num(arr_like) -> np.ndarray:
    return pd.to_numeric(arr_like, errors="coerce").to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--inputs",
        required=True,
        help='Glob for scored CSVs, e.g. "results/scored/prod_base_full_*.csv"',
    )
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_md", default=None)
    ap.add_argument("--bootstrap_iters", type=int, default=2000)

    # NEW: how to detect NIC prompts (used for conditional NIC rate)
    ap.add_argument(
        "--nic_mode",
        choices=["gt", "task_or_category", "auto"],
        default="auto",
        help=(
            "How to identify NOT_IN_CONTEXT prompts for conditional rate. "
            "'gt' uses ground_truth == NOT_IN_CONTEXT (requires a gt column). "
            "'task_or_category' uses task=='not_in_context' OR category=='anti_hallucination'. "
            "'auto' prefers gt if present else task_or_category."
        ),
    )
    args = ap.parse_args()

    paths = sorted(glob.glob(args.inputs))
    if not paths:
        raise SystemExit(f"No files matched: {args.inputs}")

    rows: List[Dict[str, object]] = []

    for p in paths:
        df = pd.read_csv(p)

        run_col = _find_col(df, ["run", "run_id", "runid"])
        model_col = _find_col(df, ["model", "model_name", "ollama_model"])
        temp_col = _find_col(df, ["temperature", "temp", "run_temperature"])
        latency_col = _find_col(df, ["latency_ms", "avg_latency_ms", "latency"])
        score_col = _find_col(df, ["score", "accuracy_score", "task_score"])

        not_in_ctx_col = _find_col(df, ["not_in_context_violation", "not_in_context", "not_in_ctx"])
        numeric_inv_col = _find_col(df, ["numeric_invention_flag", "numeric_invention", "numeric_invent", "num_invention"])

        # Optional helpers for NIC conditional denominator
        task_col = _find_col(df, ["task"])
        category_col = _find_col(df, ["category"])
        gt_col = _find_col(df, ["ground_truth", "gt"])

        # Basic identifiers
        run_id = str(df[run_col].iloc[0]) if run_col else Path(p).stem
        model = str(df[model_col].iloc[0]) if model_col else ""
        temp = float(df[temp_col].iloc[0]) if temp_col and pd.notnull(df[temp_col].iloc[0]) else float("nan")

        # Score stats
        if not score_col:
            raise SystemExit(f"Could not find a score column in {p}. Columns: {list(df.columns)}")

        scores = _to_num(df[score_col])
        n = int(np.sum(~np.isnan(scores)))
        acc = float(np.nanmean(scores)) if n else float("nan")

        unique = set(np.unique(scores[~np.isnan(scores)]).tolist())
        is_binary = unique.issubset({0.0, 1.0}) and len(unique) > 0

        if is_binary:
            k = int(np.nansum(scores))  # count of 1s
            ci_lo, ci_hi = wilson_ci(k=k, n=n)
        else:
            ci_lo, ci_hi = bootstrap_ci_mean(scores, iters=args.bootstrap_iters)

        # Latency
        avg_latency = float("nan")
        if latency_col:
            lat = _to_num(df[latency_col])
            avg_latency = float(np.nanmean(lat)) if np.sum(~np.isnan(lat)) else float("nan")

        # Numeric invention (mean over all prompts)
        numeric_inv = float("nan")
        if numeric_inv_col:
            v = _to_num(df[numeric_inv_col])
            numeric_inv = float(np.nanmean(v)) if np.sum(~np.isnan(v)) else float("nan")

        # NOT_IN_CONTEXT rates
        nic_global = float("nan")
        nic_cond = float("nan")
        nic_k = float("nan")
        nic_n = float("nan")

        if not_in_ctx_col:
            v_all = _to_num(df[not_in_ctx_col])
            nic_global = float(np.nanmean(v_all)) if np.sum(~np.isnan(v_all)) else float("nan")

            # Identify NIC prompt subset for conditional rate
            use_gt = False
            if args.nic_mode == "gt":
                use_gt = True
            elif args.nic_mode == "task_or_category":
                use_gt = False
            else:  # auto
                use_gt = bool(gt_col)

            mask = None
            if use_gt and gt_col:
                gt_series = df[gt_col].astype(str).fillna("")
                mask = (gt_series.str.strip() == "NOT_IN_CONTEXT")
            else:
                # task/category heuristic
                tmask = None
                cmask = None
                if task_col:
                    t = df[task_col].astype(str).fillna("").str.strip().str.lower()
                    tmask = (t == "not_in_context")

                if category_col:
                    c = df[category_col].astype(str).fillna("").str.strip().str.lower()
                    cmask = (c == "anti_hallucination")
                if tmask is None and cmask is None:
                    mask = None
                elif tmask is None:
                    mask = cmask
                elif cmask is None:
                    mask = tmask
                else:
                    mask = (tmask | cmask)

            if mask is not None:
                m = mask.to_numpy(dtype=bool)
                nic_n_int = int(np.sum(m))  # denominator = number of NIC prompts
                if nic_n_int > 0:
                    v_sub = v_all[m]
                    # Treat missing NIC flags as 0 (conservative + stable across older CSVs)
                    v_sub0 = np.nan_to_num(v_sub, nan=0.0)
                    nic_k = float(np.sum(v_sub0))
                    nic_n = float(nic_n_int)
                    nic_cond = float(nic_k / nic_n_int)

        rows.append(
            {
                "run": run_id,
                "model": model,
                "temp": temp,
                "n": n,
                "accuracy_mean": acc,
                "accuracy_ci_low": ci_lo,
                "accuracy_ci_high": ci_hi,
                "avg_latency_ms": avg_latency,
                "not_in_context_violation_rate_global": nic_global,
                "not_in_context_violation_rate_cond": nic_cond,
                "nic_k": nic_k,
                "nic_n": nic_n,
                "numeric_invention_rate": numeric_inv,
                "scored_file": p,
            }
        )

    out = pd.DataFrame(rows)

    # Sort: accuracy desc, then latency asc
    out = out.sort_values(by=["accuracy_mean", "avg_latency_ms"], ascending=[False, True]).reset_index(drop=True)

    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Wrote: {args.out_csv}")

    if args.out_md:
        md_lines: List[str] = []
        md_lines.append("# Leaderboard (with uncertainty)")
        md_lines.append("")
        md_lines.append("| Run | Model | Temp | N | Acc | 95% CI | Avg latency | NIC (global) | NIC (cond) | NIC k/n | Numeric invention |")
        md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

        for _, r in out.iterrows():
            run = f"`{r['run']}`"
            model = f"`{r['model']}`" if r["model"] else "`(unknown)`"
            temp = r["temp"]
            n = int(r["n"])

            acc = _pct(float(r["accuracy_mean"]))
            lo = _pct(float(r["accuracy_ci_low"]))
            hi = _pct(float(r["accuracy_ci_high"]))

            lat = float(r["avg_latency_ms"])
            lat_s = f"{lat:.0f} ms" if pd.notnull(lat) else "NA"

            nic_g = r["not_in_context_violation_rate_global"]
            nic_c = r["not_in_context_violation_rate_cond"]
            nic_k = r["nic_k"]
            nic_n = r["nic_n"]

            nic_g_s = f"{_pct(float(nic_g)):.1f}%" if pd.notnull(nic_g) else "NA"
            nic_c_s = f"{_pct(float(nic_c)):.1f}%" if pd.notnull(nic_c) else "NA"
            nic_kn_s = f"{int(nic_k)}/{int(nic_n)}" if pd.notnull(nic_k) and pd.notnull(nic_n) else "NA"

            ninv = r["numeric_invention_rate"]
            ninv_s = f"{_pct(float(ninv)):.1f}%" if pd.notnull(ninv) else "NA"

            temp_s = f"{temp:.2f}" if pd.notnull(temp) else "NA"

            md_lines.append(
                f"| {run} | {model} | {temp_s} | {n} | {acc:.1f}% | {lo:.1f}–{hi:.1f}% | {lat_s} | "
                f"{nic_g_s} | {nic_c_s} | {nic_kn_s} | {ninv_s} |"
            )

        Path(args.out_md).write_text("\n".join(md_lines) + "\n", encoding="utf-8")
        print(f"Wrote: {args.out_md}")


if __name__ == "__main__":
    main()