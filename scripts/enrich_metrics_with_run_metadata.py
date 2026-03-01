#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Dict, Any, Optional

import pandas as pd


def _first_nonnull(*vals):
    for v in vals:
        if v is not None and v != "":
            return v
    return None


def extract_run_metadata_from_jsonl(jsonl_path: str, max_lines: int = 50) -> Optional[Dict[str, Any]]:
    """
    Read up to max_lines from a run JSONL and extract stable metadata fields.
    Returns dict keyed by run_id or None if nothing usable found.
    """
    run_id = None
    cell_id = None
    profile = None
    model = None
    family = None
    host = None
    num_ctx = None
    temperature = None
    num_predict = None

    try:
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Common top-level fields (best case)
                run_id = _first_nonnull(run_id, obj.get("run_id"))
                cell_id = _first_nonnull(cell_id, obj.get("cell_id"))
                profile = _first_nonnull(profile, obj.get("profile"))
                family = _first_nonnull(family, obj.get("family"))

                # Model name can appear under different keys depending on logger
                model = _first_nonnull(
                    model,
                    obj.get("model"),
                    obj.get("model_name"),
                    obj.get("model_id"),
                    obj.get("modelid"),
                )

                host = _first_nonnull(host, obj.get("host"), obj.get("base_url"))

                # Options may be nested
                opts = obj.get("options") or {}
                if isinstance(opts, dict):
                    num_ctx = _first_nonnull(num_ctx, opts.get("num_ctx"))
                    temperature = _first_nonnull(temperature, opts.get("temperature"))
                    num_predict = _first_nonnull(num_predict, opts.get("num_predict"))

                # Some loggers nest config under "request" or similar
                req = obj.get("request") or {}
                if isinstance(req, dict):
                    ropts = req.get("options") or {}
                    if isinstance(ropts, dict):
                        num_ctx = _first_nonnull(num_ctx, ropts.get("num_ctx"))
                        temperature = _first_nonnull(temperature, ropts.get("temperature"))
                        num_predict = _first_nonnull(num_predict, ropts.get("num_predict"))
                    model = _first_nonnull(model, req.get("model"))

                # If we already have the important ones, we can stop early
                if run_id and (cell_id or profile or model) and (num_ctx is not None):
                    break

        # Fall back: infer run_id from filename if not present
        if not run_id:
            base = os.path.basename(jsonl_path)
            # if your filenames are "<run_id>.jsonl"
            if base.endswith(".jsonl"):
                run_id = base[:-5]

        if not run_id:
            return None

        return {
            "run_id": run_id,
            "cell_id": cell_id,
            "profile": profile,
            "model": model,
            "family": family,
            "host": host,
            "num_ctx": (int(num_ctx) if num_ctx is not None and str(num_ctx).isdigit() else num_ctx),
            "temperature": temperature,
            "num_predict": num_predict,
            "run_file": os.path.basename(jsonl_path),
        }

    except FileNotFoundError:
        return None


def build_metadata_index(runs_glob: str) -> pd.DataFrame:
    rows = []
    paths = sorted(glob.glob(runs_glob))
    if not paths:
        raise FileNotFoundError(f"No run JSONLs found at: {runs_glob}")

    for p in paths:
        meta = extract_run_metadata_from_jsonl(p)
        if meta:
            rows.append(meta)

    if not rows:
        raise RuntimeError("Found run files, but could not extract metadata from any JSONL.")

    df = pd.DataFrame(rows)

    # If multiple files map to same run_id, keep the first non-null for each column
    df = (
        df.sort_values("run_file")
          .groupby("run_id", as_index=False)
          .agg(lambda s: s.dropna().iloc[0] if len(s.dropna()) else None)
    )

    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_glob", default="results/v2/runs/*.jsonl")
    ap.add_argument("--metrics_run", default="metrics_run.csv")
    ap.add_argument("--metrics_turn", default="metrics_turn.csv")
    ap.add_argument("--out_run", default="metrics_run_enriched.csv")
    ap.add_argument("--out_turn", default="metrics_turn_enriched.csv")
    ap.add_argument("--enrich_turn", action="store_true", help="Also write enriched turn-level CSV.")
    args = ap.parse_args()

    meta_df = build_metadata_index(args.runs_glob)

    run_df = pd.read_csv(args.metrics_run)
    run_enriched = run_df.merge(meta_df, on="run_id", how="left")

    # Quick sanity: show missing merges
    missing = run_enriched["cell_id"].isna().sum() if "cell_id" in run_enriched.columns else None
    print(f"[enrich] runs found: {len(meta_df)}")
    print(f"[enrich] metrics_run rows: {len(run_df)}")
    if missing is not None:
        print(f"[enrich] metrics_run rows missing cell_id after merge: {missing}")

    run_enriched.to_csv(args.out_run, index=False)
    print(f"[enrich] wrote: {args.out_run}")

    if args.enrich_turn:
        turn_df = pd.read_csv(args.metrics_turn)
        turn_enriched = turn_df.merge(
            meta_df[["run_id", "cell_id", "profile", "model", "family", "num_ctx"]],
            on="run_id",
            how="left",
        )
        turn_enriched.to_csv(args.out_turn, index=False)
        print(f"[enrich] wrote: {args.out_turn}")


if __name__ == "__main__":
    main()