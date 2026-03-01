#!/usr/bin/env python3
"""
Generate per-run failure breakdown reports from scored CSVs.

Usage:
  python scripts/failure_breakdown.py \
    --inputs "results/scored/prod_base_full_20260222_*_ee516a1c243b.csv" \
    --out_dir results/reports/failures_base_clean_20260222 \
    --top_k 8
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path
from typing import List, Optional

import pandas as pd
import numpy as np


def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols:
            return cols[cand.lower()]
    for cand in candidates:
        for c in df.columns:
            if cand.lower() in c.lower():
                return c
    return None


def _as_num(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").to_numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help="Glob for scored CSVs")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--max_examples", type=int, default=5)
    args = ap.parse_args()

    paths = sorted(glob.glob(args.inputs))
    if not paths:
        raise SystemExit(f"No files matched: {args.inputs}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in paths:
        df = pd.read_csv(p)

        run_col = _find_col(df, ["run", "run_id", "runid"])
        model_col = _find_col(df, ["model", "model_name", "ollama_model"])
        temp_col = _find_col(df, ["temperature", "temp", "run_temperature"])
        score_col = _find_col(df, ["score", "accuracy_score", "task_score"])
        latency_col = _find_col(df, ["latency_ms", "latency"])

        prompt_id_col = _find_col(df, ["prompt_id", "promptid", "id"])
        scoring_method_col = _find_col(df, ["scoring_method", "method"])
        category_col = _find_col(df, ["category", "task_type", "group"])

        not_in_ctx_col = _find_col(df, ["not_in_context_violation", "not_in_context", "not_in_ctx"])
        numeric_inv_col = _find_col(df, ["numeric_invention", "numeric_invent", "num_invention"])

        json_strict_col = _find_col(df, ["json_strict", "json_schema_strict", "json_valid_strict"])
        json_lenient_col = _find_col(df, ["json_lenient", "json_valid_lenient"])

        # Optional “explain why score=0” columns if you have them
        reason_col = _find_col(df, ["reason", "failure_reason", "error", "explanation"])
        response_col = _find_col(df, ["response", "model_response", "output_text", "completion"])
        ref_col = _find_col(df, ["reference", "ground_truth", "expected"])

        run_id = str(df[run_col].iloc[0]) if run_col else Path(p).stem
        model = str(df[model_col].iloc[0]) if model_col else ""
        temp = float(df[temp_col].iloc[0]) if temp_col and pd.notnull(df[temp_col].iloc[0]) else float("nan")

        scores = _as_num(df[score_col]) if score_col else np.array([])
        n = int(np.sum(~np.isnan(scores)))
        acc = float(np.nanmean(scores)) if n else float("nan")

        lat = _as_num(df[latency_col]) if latency_col else np.array([])
        avg_lat = float(np.nanmean(lat)) if np.sum(~np.isnan(lat)) else float("nan")

        def rate(col: Optional[str]) -> float:
            if not col:
                return float("nan")
            v = _as_num(df[col])
            return float(np.nanmean(v)) if np.sum(~np.isnan(v)) else float("nan")

        not_in_ctx_rate = rate(not_in_ctx_col)
        numeric_inv_rate = rate(numeric_inv_col)

        # Worst prompts
        worst = df.copy()
        if score_col:
            worst["_score_num"] = pd.to_numeric(worst[score_col], errors="coerce")
            worst = worst.sort_values(by=["_score_num"], ascending=True)
        worst = worst.head(args.top_k)

        lines = []
        lines.append(f"# Failure breakdown")
        lines.append("")
        lines.append(f"- **Run:** `{run_id}`")
        lines.append(f"- **Model:** `{model}`" if model else f"- **Model:** `(unknown)`")
        lines.append(f"- **Temp:** `{temp}`" if pd.notnull(temp) else "- **Temp:** `NA`")
        lines.append(f"- **N:** `{n}`")
        lines.append(f"- **Accuracy:** `{acc*100:.1f}%`" if pd.notnull(acc) else "- **Accuracy:** `NA`")
        lines.append(f"- **Avg latency:** `{avg_lat:.0f} ms`" if pd.notnull(avg_lat) else "- **Avg latency:** `NA`")
        if pd.notnull(not_in_ctx_rate):
            lines.append(f"- **NOT_IN_CONTEXT rate:** `{not_in_ctx_rate*100:.1f}%`")
        if pd.notnull(numeric_inv_rate):
            lines.append(f"- **Numeric invention rate:** `{numeric_inv_rate*100:.1f}%`")
        lines.append("")

        # Worst prompts table
        lines.append("## Lowest-scoring prompts")
        lines.append("")
        hdr = ["prompt_id", "score", "category", "scoring_method", "latency_ms"]
        lines.append("| " + " | ".join(hdr) + " |")
        lines.append("|" + "|".join(["---"] * len(hdr)) + "|")

        for _, r in worst.iterrows():
            pid = str(r[prompt_id_col]) if prompt_id_col and pd.notnull(r.get(prompt_id_col)) else "(na)"
            sc = r.get(score_col, np.nan) if score_col else np.nan
            cat = str(r[category_col]) if category_col and pd.notnull(r.get(category_col)) else ""
            sm = str(r[scoring_method_col]) if scoring_method_col and pd.notnull(r.get(scoring_method_col)) else ""
            lm = r.get(latency_col, np.nan) if latency_col else np.nan
            lm_s = f"{float(lm):.0f}" if pd.notnull(lm) else ""
            lines.append(f"| `{pid}` | `{sc}` | `{cat}` | `{sm}` | `{lm_s}` |")
        lines.append("")

        # Proxy examples
        def add_examples(title: str, col: Optional[str]):
            if not col:
                return
            v = pd.to_numeric(df[col], errors="coerce").fillna(0)
            ex = df[v.astype(float) > 0].head(args.max_examples)
            lines.append(f"## {title} examples")
            lines.append("")
            if ex.empty:
                lines.append("_None found in this run._")
                lines.append("")
                return
            for i, (_, r) in enumerate(ex.iterrows(), start=1):
                pid = str(r[prompt_id_col]) if prompt_id_col and pd.notnull(r.get(prompt_id_col)) else "(na)"
                cat = str(r[category_col]) if category_col and pd.notnull(r.get(category_col)) else ""
                lines.append(f"### {i}) `{pid}` `{cat}`".strip())
                if reason_col and pd.notnull(r.get(reason_col)):
                    lines.append(f"- Reason: {str(r[reason_col])}")
                if ref_col and pd.notnull(r.get(ref_col)):
                    lines.append(f"- Expected (ref): `{str(r[ref_col])[:300]}`")
                if response_col and pd.notnull(r.get(response_col)):
                    resp = str(r[response_col])
                    lines.append("")
                    lines.append("Model response (truncated):")
                    lines.append("")
                    lines.append("```")
                    lines.append(resp[:800])
                    lines.append("```")
                lines.append("")

        add_examples("NOT_IN_CONTEXT", not_in_ctx_col)
        add_examples("Numeric invention", numeric_inv_col)

        # JSON validity summary if present
        if json_strict_col or json_lenient_col:
            lines.append("## JSON validity (if applicable)")
            lines.append("")
            if json_strict_col:
                v = pd.to_numeric(df[json_strict_col], errors="coerce")
                lines.append(f"- Strict valid rate: `{float(v.mean())*100:.1f}%`")
            if json_lenient_col:
                v = pd.to_numeric(df[json_lenient_col], errors="coerce")
                lines.append(f"- Lenient valid rate: `{float(v.mean())*100:.1f}%`")
            lines.append("")

        out_path = out_dir / f"{Path(p).stem}.md"
        out_path.write_text("\n".join(lines) + "\n")
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()