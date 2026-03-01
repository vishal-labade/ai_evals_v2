from __future__ import annotations
from typing import Any, Dict, List


def _txt(x: Any) -> str:
    return "" if x is None else str(x)


def score_recovery_score(scenario: Dict[str, Any], turn_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    rec_cfg = (scenario.get("checks") or {}).get("recovery") or {}
    if not rec_cfg:
        return {"recovery_score": 0.0, "recovery_total": 0}

    t_idx = int(rec_cfg.get("score_turn_index", -1))
    total = 0
    score_sum = 0.0

    for rec in turn_records:
        if int(rec.get("turn_index", -2)) != t_idx:
            continue
        total += 1
        out = _txt(rec.get("assistant_text")).lower()

        has_apology = any(k in out for k in ["sorry", "apolog", "my mistake"])
        has_ack = any(k in out for k in ["you're right", "you are right", "i was wrong", "good catch"])
        has_fix = any(k in out for k in ["correct", "correction", "let me fix", "updated", "here's the corrected"])

        if (has_apology or has_ack) and has_fix:
            score_sum += 2.0
        elif has_apology or has_ack or has_fix:
            score_sum += 1.0

    avg = (score_sum / total) if total else 0.0
    # normalize to 0..1 for easy averaging: 0, 0.5, 1.0
    norm = avg / 2.0 if total else 0.0
    return {"recovery_score": round(norm, 4), "recovery_total": total}