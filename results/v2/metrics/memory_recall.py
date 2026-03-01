from __future__ import annotations
from typing import Any, Dict, List


def _txt(x: Any) -> str:
    return "" if x is None else str(x)


def score_memory_recall(scenario: Dict[str, Any], turn_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    mem = (scenario.get("checks") or {}).get("memory") or {}
    if not mem:
        return {"memory_pass": 0, "memory_total": 0, "memory_score": 0.0}

    t_idx = int(mem.get("turn_index", -1))
    must = list(mem.get("expected_contains") or [])
    must_not = list(mem.get("expected_not_contains") or [])

    total = 0
    passed = 0
    for rec in turn_records:
        if int(rec.get("turn_index", -2)) != t_idx:
            continue
        total += 1
        out = _txt(rec.get("assistant_text")).lower()
        ok = all(str(x).lower() in out for x in must) and all(str(x).lower() not in out for x in must_not)
        if ok:
            passed += 1

    score = (passed / total) if total else 0.0
    return {"memory_pass": passed, "memory_total": total, "memory_score": round(score, 4)}