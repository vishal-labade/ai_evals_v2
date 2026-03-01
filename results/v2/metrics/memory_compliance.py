#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUNS_DIR = REPO_ROOT / "results" / "v2" / "runs"
DEFAULT_SCENARIOS = REPO_ROOT / "data" / "v2_suite_50.jsonl"
DEFAULT_OUT_TURN = REPO_ROOT / "results" / "v2" / "metrics" / "metrics_turn.csv"
DEFAULT_OUT_RUN = REPO_ROOT / "results" / "v2" / "metrics" / "metrics_run.csv"


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _load_scenarios(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for s in _read_jsonl(path):
        sid = s.get("scenario_id")
        if sid:
            out[str(sid)] = s
    return out


def _contains_any(text: str, items: List[str]) -> bool:
    t = (text or "").lower()
    return any((i or "").lower() in t for i in items)


def _get_memory_requirements(scenario: Dict[str, Any]) -> Dict[int, Dict[str, List[str]]]:
    """
    scenario["checks"]["memory"] can be:
      - { "turn_index": <int>, "require_any": [..], "forbid_any": [..] }  (single)
      - { "turns": [ {"turn_index": <int>, "require_any":[..], "forbid_any":[..]}, ... ] } (multi)

    Returns:
      { turn_index -> {"require_any":[..], "forbid_any":[..]} }
    """
    checks = scenario.get("checks") or {}
    mem = checks.get("memory") or {}
    req: Dict[int, Dict[str, List[str]]] = {}

    def _add_turn(ti_raw: Any, require_any: Any, forbid_any: Any) -> None:
        if ti_raw is None:
            return
        ti = int(ti_raw)
        ra = list(require_any or [])
        fa = list(forbid_any or [])
        if not ra and not fa:
            return
        req[ti] = {"require_any": ra, "forbid_any": fa}

    if isinstance(mem, dict) and "turn_index" in mem:
        _add_turn(mem.get("turn_index"), mem.get("require_any"), mem.get("forbid_any"))
        return req

    turns = mem.get("turns") if isinstance(mem, dict) else None
    if isinstance(turns, list):
        for t in turns:
            if not isinstance(t, dict) or "turn_index" not in t:
                continue
            _add_turn(t.get("turn_index"), t.get("require_any"), t.get("forbid_any"))

    return req


def compute_mcs_for_scenario(
    turns: List[Dict[str, Any]],
    mem_req: Dict[int, Dict[str, List[str]]],
) -> Tuple[List[Optional[int]], Optional[int], Optional[float]]:
    """
    mcs_turn[t]:
      - 1/0 for turns that have memory requirements (require_any and/or forbid_any)
      - None otherwise

    drop_turn:
      first required turn where requirement fails, else None

    AUC:
      prefix_ok[t] = 1 if all required checks up to t passed else 0
      auc = mean(prefix_ok over 0..T-1)
      If there are no memory-required turns, auc = None
    """
    max_t = max(int(r.get("turn_index", 0)) for r in turns) if turns else -1
    T = max_t + 1

    mcs_turn: List[Optional[int]] = [None] * T
    required_indices = sorted(mem_req.keys())
    if not required_indices:
        return mcs_turn, None, None

    out_by_t: Dict[int, str] = {}
    for r in turns:
        ti = int(r.get("turn_index", -1))
        if ti < 0:
            continue
        out_by_t.setdefault(ti, str(r.get("assistant_text", "")))

    fail_turn: Optional[int] = None

    for ti in required_indices:
        out = out_by_t.get(ti, "")
        rule = mem_req[ti]
        require_any = list(rule.get("require_any") or [])
        forbid_any = list(rule.get("forbid_any") or [])

        ok = True
        if require_any:
            ok = ok and _contains_any(out, require_any)
        if forbid_any:
            ok = ok and (not _contains_any(out, forbid_any))

        mcs_turn[ti] = 1 if ok else 0
        if (not ok) and fail_turn is None:
            fail_turn = ti

    prefix_ok: List[int] = []
    all_ok = True
    req_set = set(required_indices)
    for t in range(T):
        if t in req_set:
            all_ok = all_ok and bool(mcs_turn[t] == 1)
        prefix_ok.append(1 if all_ok else 0)

    auc = sum(prefix_ok) / len(prefix_ok) if prefix_ok else None
    return mcs_turn, fail_turn, (round(auc, 4) if auc is not None else None)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_dir", type=str, default=str(DEFAULT_RUNS_DIR))
    ap.add_argument("--scenarios", type=str, default=str(DEFAULT_SCENARIOS))
    ap.add_argument("--out_turn", type=str, default=str(DEFAULT_OUT_TURN))
    ap.add_argument("--out_run", type=str, default=str(DEFAULT_OUT_RUN))
    args = ap.parse_args()

    runs_dir = Path(args.runs_dir).resolve()
    scenarios_path = Path(args.scenarios).resolve()
    out_turn = Path(args.out_turn).resolve()
    out_run = Path(args.out_run).resolve()

    scenarios = _load_scenarios(scenarios_path)

    run_files = sorted(runs_dir.glob("*.jsonl"))
    all_rows: List[Dict[str, Any]] = []
    for rf in run_files:
        all_rows.extend(_read_jsonl(rf))

    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in all_rows:
        run_id = str(r.get("run_id", ""))
        scenario_id = str(r.get("scenario_id", ""))
        if not run_id or not scenario_id:
            continue
        grouped.setdefault((run_id, scenario_id), []).append(r)

    for k in grouped:
        grouped[k].sort(key=lambda x: int(x.get("turn_index", 0)))

    out_turn.parent.mkdir(parents=True, exist_ok=True)
    out_run.parent.mkdir(parents=True, exist_ok=True)

    turn_out_rows: List[Dict[str, Any]] = []
    run_out_acc: Dict[str, Dict[str, Any]] = {}

    for (run_id, scenario_id), turns in grouped.items():
        scenario = scenarios.get(scenario_id, {})
        mem_req = _get_memory_requirements(scenario)

        mcs_turn, drop_turn, auc = compute_mcs_for_scenario(turns, mem_req)

        for r in turns:
            ti = int(r.get("turn_index", 0))
            m = mcs_turn[ti] if 0 <= ti < len(mcs_turn) else None
            if m is None:
                continue
            turn_out_rows.append(
                {"run_id": run_id, "scenario_id": scenario_id, "turn_index": ti, "mcs_turn": m}
            )

        if run_id not in run_out_acc:
            run_out_acc[run_id] = {"mcs_sum": 0, "mcs_n": 0, "drop_turns": [], "auc_values": []}

        mcs_vals = [x for x in mcs_turn if x is not None]
        if mcs_vals:
            run_out_acc[run_id]["mcs_sum"] += sum(int(x) for x in mcs_vals)
            run_out_acc[run_id]["mcs_n"] += len(mcs_vals)
        if drop_turn is not None:
            run_out_acc[run_id]["drop_turns"].append(drop_turn)
        if auc is not None:
            run_out_acc[run_id]["auc_values"].append(float(auc))

    with out_turn.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "scenario_id", "turn_index", "mcs_turn"])
        w.writeheader()
        for r in turn_out_rows:
            w.writerow(r)

    run_rows: List[Dict[str, Any]] = []
    for run_id, acc in run_out_acc.items():
        n = int(acc["mcs_n"])
        mcs = (float(acc["mcs_sum"]) / n) if n else None

        drop_turn_mean = ""
        if acc["drop_turns"]:
            drop_turn_mean = round(sum(acc["drop_turns"]) / len(acc["drop_turns"]), 4)

        auc_mean = ""
        if acc["auc_values"]:
            auc_mean = round(sum(acc["auc_values"]) / len(acc["auc_values"]), 4)

        run_rows.append(
            {
                "run_id": run_id,
                "mcs": round(mcs, 4) if mcs is not None else "",
                "drop_turn_mean": drop_turn_mean,
                "mcs_auc": auc_mean,
                "mcs_n": n,
            }
        )

    with out_run.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["run_id", "mcs", "drop_turn_mean", "mcs_auc", "mcs_n"])
        w.writeheader()
        for r in run_rows:
            w.writerow(r)

    print(f"[memory_compliance] scenarios={scenarios_path}")
    print(f"[memory_compliance] wrote {out_turn}")
    print(f"[memory_compliance] wrote {out_run}")
    print("[memory_compliance] NOTE: MCS only applies to scenarios with checks.memory defined.")


if __name__ == "__main__":
    main()