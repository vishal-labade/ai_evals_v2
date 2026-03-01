#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

from metrics.persona_stability import score_persona_stability
from metrics.instruction_persistence import score_instruction_persistence
from metrics.memory_recall import score_memory_recall
from metrics.recovery_score import score_recovery_score
from metrics.consistency_contradiction import score_consistency_contradiction
from metrics.calibration import score_calibration


REPO_ROOT = Path(__file__).resolve().parents[2]
SCENARIOS_PATH = REPO_ROOT / "data" / "v2_scenarios.jsonl"
RUNS_DIR = REPO_ROOT / "results" / "v2" / "runs"

OUT_CSV = REPO_ROOT / "results" / "v2" / "metrics" / "behavioral_metrics.csv"
OUT_MODEL_JSON = REPO_ROOT / "results" / "v2" / "metrics" / "model_scorecard.json"
OUT_PROFILE_JSON = REPO_ROOT / "results" / "v2" / "metrics" / "profile_scorecard.json"
OUT_MODEL_PROFILE_JSON = REPO_ROOT / "results" / "v2" / "metrics" / "model_profile_scorecard.json"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _load_scenarios(path: Path) -> Dict[str, Dict[str, Any]]:
    scenarios: Dict[str, Dict[str, Any]] = {}
    for row in _read_jsonl(path):
        sid = row.get("scenario_id")
        if sid:
            scenarios[str(sid)] = row
    return scenarios


def _group(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str], List[Dict[str, Any]]]:
    g: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for r in records:
        run_id = str(r.get("run_id", ""))
        scenario_id = str(r.get("scenario_id", ""))
        if run_id and scenario_id:
            g[(run_id, scenario_id)].append(r)
    for k in g:
        g[k].sort(key=lambda x: int(x.get("turn_index", 0)))
    return g


def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _acc_add(acc: Dict[str, List[float]], key: str, val: float) -> None:
    acc[key].append(float(val))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default=str(RUNS_DIR))
    ap.add_argument("--scenarios", type=str, default=str(SCENARIOS_PATH))
    ap.add_argument("--out_csv", type=str, default=str(OUT_CSV))
    ap.add_argument("--out_model_json", type=str, default=str(OUT_MODEL_JSON))
    ap.add_argument("--out_profile_json", type=str, default=str(OUT_PROFILE_JSON))
    ap.add_argument("--out_model_profile_json", type=str, default=str(OUT_MODEL_PROFILE_JSON))
    ap.add_argument("--limit_runs", type=int, default=0)
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir)
    scenarios = _load_scenarios(Path(args.scenarios))

    out_csv = Path(args.out_csv)
    out_model_json = Path(args.out_model_json)
    out_profile_json = Path(args.out_profile_json)
    out_model_profile_json = Path(args.out_model_profile_json)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_model_json.parent.mkdir(parents=True, exist_ok=True)
    out_profile_json.parent.mkdir(parents=True, exist_ok=True)
    out_model_profile_json.parent.mkdir(parents=True, exist_ok=True)

    run_files = sorted(runs_dir.glob("*.jsonl"))
    if args.limit_runs:
        run_files = run_files[: args.limit_runs]

    records: List[Dict[str, Any]] = []
    for rf in run_files:
        records.extend(_read_jsonl(rf))

    grouped = _group(records)

    rows: List[Dict[str, Any]] = []

    # Aggregations
    model_acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    profile_acc: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    model_profile_acc: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for (run_id, scenario_id), turns in grouped.items():
        scenario = scenarios.get(scenario_id, {"scenario_id": scenario_id, "tags": [], "checks": {}})

        first = turns[0]
        model = str(first.get("model", ""))
        profile = str(first.get("profile", "")) or "unknown_profile"
        options = first.get("options") or {}
        temperature = options.get("temperature", None)
        num_ctx = options.get("num_ctx", None)

        persona = score_persona_stability(scenario, turns)
        fmt = score_instruction_persistence(scenario, turns)
        mem = score_memory_recall(scenario, turns)
        rec = score_recovery_score(scenario, turns)
        con = score_consistency_contradiction(scenario, turns)
        cal = score_calibration(scenario, turns)

        # BRI only from evaluated parts
        bri_parts: List[float] = []

        if persona.get("persona_total_turns", 0) > 0:
            bri_parts.append(float(persona["persona_pss"]))

        if fmt.get("format_total", 0) > 0:
            bri_parts.append(float(fmt["format_score"]))

        if mem.get("memory_total", 0) > 0:
            bri_parts.append(float(mem["memory_score"]))

        if rec.get("recovery_total", 0) > 0:
            bri_parts.append(float(rec["recovery_score"]))

        if con.get("contradiction_total_pairs", 0) > 0:
            bri_parts.append(1.0 - float(con["contradiction_rate"]))

        if cal.get("calibration_n", 0) and cal.get("avg_abs_error") is not None:
            bri_parts.append(1.0 - float(cal["avg_abs_error"]))

        bri = round(_mean(bri_parts), 4) if bri_parts else 0.0

        row = {
            "run_id": run_id,
            "profile": profile,
            "model": model,
            "temperature": temperature,
            "num_ctx": num_ctx,
            "scenario_id": scenario_id,
            "tags": ",".join(scenario.get("tags") or []),
            "persona_pss": persona.get("persona_pss", 0.0),
            "persona_drift_turn": persona.get("persona_drift_turn", None),
            "format_score": fmt.get("format_score", 0.0),
            "memory_score": mem.get("memory_score", 0.0),
            "recovery_score": rec.get("recovery_score", 0.0),
            "contradiction_rate": con.get("contradiction_rate", 0.0),
            "avg_confidence": cal.get("avg_confidence", None),
            "avg_abs_error": cal.get("avg_abs_error", None),
            "bri": bri,
        }
        rows.append(row)

        # Helper pointers
        m_acc = model_acc[model]
        p_acc = profile_acc[profile]
        mp_acc = model_profile_acc[model][profile]

        # Always accumulate BRI at all levels
        _acc_add(m_acc, "bri", bri)
        _acc_add(p_acc, "bri", bri)
        _acc_add(mp_acc, "bri", bri)

        if persona.get("persona_total_turns", 0) > 0:
            v = float(persona["persona_pss"])
            _acc_add(m_acc, "persona_pss", v)
            _acc_add(p_acc, "persona_pss", v)
            _acc_add(mp_acc, "persona_pss", v)

        if fmt.get("format_total", 0) > 0:
            v = float(fmt["format_score"])
            _acc_add(m_acc, "format_score", v)
            _acc_add(p_acc, "format_score", v)
            _acc_add(mp_acc, "format_score", v)

        if mem.get("memory_total", 0) > 0:
            v = float(mem["memory_score"])
            _acc_add(m_acc, "memory_score", v)
            _acc_add(p_acc, "memory_score", v)
            _acc_add(mp_acc, "memory_score", v)

        if rec.get("recovery_total", 0) > 0:
            v = float(rec["recovery_score"])
            _acc_add(m_acc, "recovery_score", v)
            _acc_add(p_acc, "recovery_score", v)
            _acc_add(mp_acc, "recovery_score", v)

        if con.get("contradiction_total_pairs", 0) > 0:
            v = 1.0 - float(con["contradiction_rate"])
            _acc_add(m_acc, "consistency_score", v)
            _acc_add(p_acc, "consistency_score", v)
            _acc_add(mp_acc, "consistency_score", v)

        if cal.get("calibration_n", 0) and cal.get("avg_abs_error") is not None:
            v = 1.0 - float(cal["avg_abs_error"])
            _acc_add(m_acc, "calibration_score", v)
            _acc_add(p_acc, "calibration_score", v)
            _acc_add(mp_acc, "calibration_score", v)

    fieldnames = [
        "run_id",
        "profile",
        "model",
        "temperature",
        "num_ctx",
        "scenario_id",
        "tags",
        "persona_pss",
        "persona_drift_turn",
        "format_score",
        "memory_score",
        "recovery_score",
        "contradiction_rate",
        "avg_confidence",
        "avg_abs_error",
        "bri",
    ]

    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # Backward compatible: same structure as before
    model_scorecard: Dict[str, Any] = {"models": {}}
    for model, mets in model_acc.items():
        model_scorecard["models"][model] = {k: round(_mean(v), 4) for k, v in mets.items()}

    profile_scorecard: Dict[str, Any] = {"profiles": {}}
    for profile, mets in profile_acc.items():
        profile_scorecard["profiles"][profile] = {k: round(_mean(v), 4) for k, v in mets.items()}

    model_profile_scorecard: Dict[str, Any] = {"models": {}}
    for model, profs in model_profile_acc.items():
        model_profile_scorecard["models"][model] = {}
        for profile, mets in profs.items():
            model_profile_scorecard["models"][model][profile] = {k: round(_mean(v), 4) for k, v in mets.items()}

    out_model_json.write_text(json.dumps(model_scorecard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_profile_json.write_text(json.dumps(profile_scorecard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    out_model_profile_json.write_text(
        json.dumps(model_profile_scorecard, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    print(f"[v2/scorecard] wrote {out_csv}")
    print(f"[v2/scorecard] wrote {out_model_json}")
    print(f"[v2/scorecard] wrote {out_profile_json}")
    print(f"[v2/scorecard] wrote {out_model_profile_json}")


if __name__ == "__main__":
    main()