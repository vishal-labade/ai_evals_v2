#!/usr/bin/env python3
"""
AI Evals V3: Product metrics post-processing

This script converts:
- runs.jsonl (turn-level logs; includes ollama_raw timings + token counts)
- behavioral_metrics.csv (run/scenario-level scores from V2)

into:
- metrics_turn_v3.csv  (turn-level latency/tokens/tps + optional joins)
- metrics_run_v3.csv   (run-level aggregates: cost/latency/throughput + reliability)
- product_leaderboard_v3.csv (profile/model-level rollups + PRI)
- product_scorecard_v3.json  (compact summary for README badges or dashboards)

Hidden / sophisticated metric included:
- Prefill vs Decode decomposition (TTFT proxy + prefill_tps, decode_tps)
  This lets you compare models beyond "avg latency": you can see whether they're
  slow because prefill (context ingestion) is expensive or decode is expensive.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd


# -----------------------------
# Utilities
# -----------------------------

def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else float("nan")


def _safe_div_series(num_s, den_s):
    """Vectorized safe division for pandas Series."""
    import numpy as _np
    out = num_s / den_s
    out = out.replace([_np.inf, -_np.inf], _np.nan)
    return out


def _ns_to_ms(ns: Optional[int]) -> float:
    if ns is None:
        return float("nan")
    return ns / 1_000_000.0


def _ns_to_s(ns: Optional[int]) -> float:
    if ns is None:
        return float("nan")
    return ns / 1_000_000_000.0



def expand_jsonl_inputs(inputs: List[str]) -> List[Path]:
    """
    Expand a list of inputs into concrete JSONL file paths.

    Each entry may be:
      - a file path
      - a directory (we include *.jsonl recursively)
      - a glob pattern (e.g., artifacts/v2/**/*runs*.jsonl)

    Returns a sorted, de-duplicated list of Paths.
    """
    paths: List[Path] = []
    for s in inputs:
        p = Path(s)
        if any(ch in s for ch in ["*", "?", "["]):
            paths.extend([Path(x) for x in sorted(Path().glob(s))])
            continue
        if p.is_dir():
            paths.extend(sorted([x for x in p.rglob("*.jsonl") if x.is_file()]))
            continue
        if p.is_file():
            paths.append(p)
            continue
        raise FileNotFoundError(f"runs input not found: {s}")
    # de-duplicate
    uniq = sorted({x.resolve() for x in paths})
    return uniq


def read_jsonl_multi(paths: List[Path]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        yield from read_jsonl(path)

def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def coerce_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


# -----------------------------
# V3 Product Metrics
# -----------------------------

@dataclass(frozen=True)
class PRIWeights:
    """Product Reliability Index weights (explainable, stable defaults)."""
    memory: float = 0.35
    persona: float = 0.25
    recovery: float = 0.15
    non_contradiction: float = 0.15
    bri: float = 0.10

    def validate(self) -> None:
        s = self.memory + self.persona + self.recovery + self.non_contradiction + self.bri
        if not (0.999 <= s <= 1.001):
            raise ValueError(f"PRIWeights must sum to ~1.0, got {s}")


def compute_pri_row(row: pd.Series, w: PRIWeights) -> float:
    """
    PRI uses existing V2 metrics. These are already 0..1 for your pipeline.
    If any are missing, we treat them as 0 (conservative).
    """
    memory = coerce_float(row.get("memory_score"))
    persona = coerce_float(row.get("persona_pss"))
    recovery = coerce_float(row.get("recovery_score"))
    contradiction = coerce_float(row.get("contradiction_rate"))
    bri = coerce_float(row.get("bri"))

    # Conservative defaults if NaN
    memory = 0.0 if math.isnan(memory) else memory
    persona = 0.0 if math.isnan(persona) else persona
    recovery = 0.0 if math.isnan(recovery) else recovery
    contradiction = 1.0 if math.isnan(contradiction) else contradiction  # worst-case
    bri = 0.0 if math.isnan(bri) else bri

    non_contra = 1.0 - contradiction
    return (
        w.memory * memory
        + w.persona * persona
        + w.recovery * recovery
        + w.non_contradiction * non_contra
        + w.bri * bri
    )


def build_turn_table(runs_jsonl_files: List[Path]) -> pd.DataFrame:
    """
    Extract per-turn telemetry from runs.jsonl.

    Expected schema (based on your file):
      - run_id, profile, model, scenario_id, turn_index, latency_ms
      - ollama_raw: {prompt_eval_count, eval_count, prompt_eval_duration, eval_duration, total_duration, load_duration}
    """
    rows: List[Dict[str, Any]] = []
    for rec in read_jsonl_multi(runs_jsonl_files):
        raw = rec.get("ollama_raw") or {}
        prompt_tok = raw.get("prompt_eval_count")
        gen_tok = raw.get("eval_count")

        prompt_ns = raw.get("prompt_eval_duration")
        gen_ns = raw.get("eval_duration")
        total_ns = raw.get("total_duration")
        load_ns = raw.get("load_duration")

        # Primary observed latency (your pipeline logs latency_ms)
        latency_ms = rec.get("latency_ms")
        latency_ms = coerce_float(latency_ms)

        # Decomposed timings (ns) -> (ms)
        prompt_ms = _ns_to_ms(prompt_ns)
        gen_ms = _ns_to_ms(gen_ns)
        total_ms = _ns_to_ms(total_ns)
        load_ms = _ns_to_ms(load_ns)

        # Derived throughput (tokens/sec)
        prompt_tps = _safe_div(coerce_float(prompt_tok), _ns_to_s(prompt_ns))
        decode_tps = _safe_div(coerce_float(gen_tok), _ns_to_s(gen_ns))
        total_tok = (coerce_float(prompt_tok) if prompt_tok is not None else float("nan")) + (
            coerce_float(gen_tok) if gen_tok is not None else float("nan")
        )

        # "Hidden metric": prefill vs decode split
        # TTFT proxy ≈ prompt_ms (time to process prompt before first decode token)
        ttft_ms = prompt_ms

        prefill_time_share = _safe_div(prompt_ms, total_ms)  # context tax share
        decode_time_share = _safe_div(gen_ms, total_ms)

        prompt_tok_share = _safe_div(coerce_float(prompt_tok), total_tok)

        rows.append(
            dict(
                run_id=rec.get("run_id"),
                experiment_id=rec.get("experiment_id"),
                run_name=rec.get("run_name"),
                cell_id=rec.get("cell_id"),
                profile=rec.get("profile"),
                model=rec.get("model"),
                temperature=(rec.get("options") or {}).get("temperature"),
                num_ctx=(rec.get("options") or {}).get("num_ctx"),
                scenario_id=rec.get("scenario_id"),
                scenario_index=rec.get("scenario_index"),
                turn_index=rec.get("turn_index"),
                started_at=rec.get("started_at"),
                ended_at=rec.get("ended_at"),
                latency_ms=latency_ms,
                # raw timing + tokens
                prompt_eval_count=prompt_tok,
                eval_count=gen_tok,
                prompt_ms=prompt_ms,
                decode_ms=gen_ms,
                total_ms=total_ms,
                load_ms=load_ms,
                # derived
                tokens_total=total_tok,
                prompt_tps=prompt_tps,
                decode_tps=decode_tps,
                ttft_ms=ttft_ms,
                prefill_time_share=prefill_time_share,
                decode_time_share=decode_time_share,
                prompt_tok_share=prompt_tok_share,
            )
        )

    df = pd.DataFrame(rows)

    # Fill temp/ctx from V2 if options were not embedded (common for some pipelines)
    # Leave as-is; join later from behavioral_metrics.csv where needed.
    return df


def build_v3_tables(
    behavioral_metrics_csv: Path,
    runs_jsonl_inputs: List[str],
    outdir: Path,
    weights: PRIWeights,
    price_per_1k_tokens_usd: float,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Creates:
      - metrics_turn_v3.csv
      - metrics_run_v3.csv
      - product_leaderboard_v3.csv
      - product_scorecard_v3.json
    """
    ensure_outdir(outdir)
    weights.validate()

    m = pd.read_csv(behavioral_metrics_csv)

    # Turn telemetry from jsonl
    runs_files = expand_jsonl_inputs(runs_jsonl_inputs)

    t = build_turn_table(runs_files)

    # Ensure join keys exist
    for col in ["run_id", "scenario_id"]:
        if col not in m.columns:
            raise ValueError(f"behavioral_metrics.csv missing required column: {col}")
        if col not in t.columns:
            raise ValueError(f"runs.jsonl-derived table missing required column: {col} (from merged runs files)")

    # Join turn telemetry onto scenario-level metrics (broadcast to turns)
    mt = t.merge(
        m,
        on=["run_id", "scenario_id"],
        how="left",
        suffixes=("", "_m"),
    )

    # Add PRI per scenario (same value for all turns within scenario)
    mt["pri_scenario"] = mt.apply(lambda r: compute_pri_row(r, weights), axis=1)

    # Estimate cost:
    # Ollama counts are "tokens" in its own accounting; treat as "approx tokens".
    mt["tokens_total"] = pd.to_numeric(mt["tokens_total"], errors="coerce")
    mt["cost_usd_est"] = (mt["tokens_total"] / 1000.0) * float(price_per_1k_tokens_usd)

    # Turn-level outputs
    turn_out = outdir / "metrics_turn_v3.csv"
    mt.to_csv(turn_out, index=False)

    # Run-level aggregates (per run_id + scenario_id)
    # NOTE: scenario_id appears once per scenario in behavioral_metrics.csv, but turns repeat; aggregate telemetry over turns.
    agg_cols = {
        "latency_ms": "mean",
        "tokens_total": "sum",
        "cost_usd_est": "sum",
        "prompt_ms": "mean",
        "decode_ms": "mean",
        "total_ms": "mean",
        "ttft_ms": "mean",
        "prompt_tps": "mean",
        "decode_tps": "mean",
        "prefill_time_share": "mean",
        "decode_time_share": "mean",
        "prompt_tok_share": "mean",
        # V2 metrics are stable per scenario; take first
        "persona_pss": "first",
        "memory_score": "first",
        "recovery_score": "first",
        "contradiction_rate": "first",
        "bri": "first",
        "pri_scenario": "first",
        "profile": "first",
        "model": "first",
        "temperature": "first",
        "num_ctx": "first",
        "tags": "first",
    }

    run_scen = mt.groupby(["run_id", "scenario_id"], dropna=False).agg(agg_cols).reset_index()

    # Derived "value" metrics
    run_scen["reliability_per_1k_tokens"] = _safe_div_series(run_scen["pri_scenario"], run_scen["tokens_total"] / 1000.0)
    run_scen["reliability_per_second"] = _safe_div_series(run_scen["pri_scenario"], run_scen["latency_ms"] / 1000.0)

    run_out = outdir / "metrics_run_v3.csv"
    run_scen.to_csv(run_out, index=False)

    # Leaderboard by (profile, model, temperature, num_ctx)
    grp = ["profile", "model", "temperature", "num_ctx"]
    lb = run_scen.groupby(grp, dropna=False).agg(
        n_scenarios=("scenario_id", "count"),
        pri_mean=("pri_scenario", "mean"),
        pri_p50=("pri_scenario", "median"),
        bri_mean=("bri", "mean"),
        memory_mean=("memory_score", "mean"),
        persona_mean=("persona_pss", "mean"),
        contradiction_mean=("contradiction_rate", "mean"),
        latency_ms_mean=("latency_ms", "mean"),
        tokens_total_sum=("tokens_total", "sum"),
        cost_usd_est_sum=("cost_usd_est", "sum"),
        ttft_ms_mean=("ttft_ms", "mean"),
        prompt_tps_mean=("prompt_tps", "mean"),
        decode_tps_mean=("decode_tps", "mean"),
        prefill_time_share_mean=("prefill_time_share", "mean"),
        reliability_per_1k_tokens_mean=("reliability_per_1k_tokens", "mean"),
        reliability_per_second_mean=("reliability_per_second", "mean"),
    ).reset_index()

    lb = lb.sort_values(["pri_mean", "reliability_per_1k_tokens_mean"], ascending=False)

    lb_out = outdir / "product_leaderboard_v3.csv"
    lb.to_csv(lb_out, index=False)

    # Scorecard JSON (compact)
    top = lb.head(10).to_dict(orient="records")
    scorecard = {
        "schema": "ai_evals_product_scorecard_v3",
        "price_per_1k_tokens_usd_assumption": price_per_1k_tokens_usd,
        "pri_weights": {
            "memory": weights.memory,
            "persona": weights.persona,
            "recovery": weights.recovery,
            "non_contradiction": weights.non_contradiction,
            "bri": weights.bri,
        },
        "leaderboard_top10": top,
        "notes": [
            "TTFT proxy is computed from prompt_eval_duration (prefill) when available.",
            "Prefill vs decode TPS help explain latency root causes (context tax vs generation speed).",
            "Cost is an estimate using a constant price_per_1k_tokens_usd; adapt if you have measured $/token.",
        ],
    }

    json_out = outdir / "product_scorecard_v3.json"
    json_out.write_text(json.dumps(scorecard, indent=2), encoding="utf-8")

    return mt, run_scen, lb


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AI Evals V3 product metrics post-processing")
    p.add_argument("--metrics", type=Path, required=True, help="Path to behavioral_metrics.csv (V2 output)")
    p.add_argument(
        "--runs",
        type=str,
        nargs="+",
        required=True,
        help="One or more runs.jsonl paths, directories, or glob patterns (e.g. artifacts/v2/*runs*.jsonl).",
    )
    p.add_argument("--outdir", type=Path, required=True, help="Output directory for V3 artifacts")
    p.add_argument("--price-per-1k-tokens-usd", type=float, default=0.0,
                   help="Optional cost assumption ($ per 1k tokens). Use 0 if you only want relative efficiency.")
    # PRI weights
    p.add_argument("--w-memory", type=float, default=0.35)
    p.add_argument("--w-persona", type=float, default=0.25)
    p.add_argument("--w-recovery", type=float, default=0.15)
    p.add_argument("--w-non-contradiction", type=float, default=0.15)
    p.add_argument("--w-bri", type=float, default=0.10)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    w = PRIWeights(
        memory=args.w_memory,
        persona=args.w_persona,
        recovery=args.w_recovery,
        non_contradiction=args.w_non_contradiction,
        bri=args.w_bri,
    )
    build_v3_tables(
        behavioral_metrics_csv=args.metrics,
        runs_jsonl_inputs=args.runs,
        outdir=args.outdir,
        weights=w,
        price_per_1k_tokens_usd=args.price_per_1k_tokens_usd,
    )
    print(f"[ok] wrote V3 artifacts to: {args.outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())