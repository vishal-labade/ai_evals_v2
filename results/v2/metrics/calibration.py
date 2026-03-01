from __future__ import annotations
import re
from typing import Any, Dict, List, Optional


def _txt(x: Any) -> str:
    return "" if x is None else str(x)


def _conf(text: str) -> Optional[float]:
    m = re.search(r"confidence\s*[:=]\s*(0(?:\.\d+)?|1(?:\.0+)?)", text, flags=re.IGNORECASE)
    if not m:
        return None
    try:
        v = float(m.group(1))
        return v if 0.0 <= v <= 1.0 else None
    except Exception:
        return None


def score_calibration(scenario: Dict[str, Any], turn_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    cal = (scenario.get("checks") or {}).get("calibration") or {}
    if not cal:
        return {"calibration_n": 0, "avg_confidence": None, "avg_abs_error": None}

    t_idx = int(cal.get("turn_index", -1))
    expected = list(cal.get("expected_contains") or [])

    confs: List[float] = []
    errs: List[float] = []

    for rec in turn_records:
        if int(rec.get("turn_index", -2)) != t_idx:
            continue
        out = _txt(rec.get("assistant_text"))
        c = _conf(out)
        if c is None:
            continue

        low = out.lower()
        correct = all(str(x).lower() in low for x in expected) if expected else None
        if correct is None:
            continue

        y = 1.0 if correct else 0.0
        confs.append(c)
        errs.append(abs(c - y))

    if not confs:
        return {"calibration_n": 0, "avg_confidence": None, "avg_abs_error": None}

    return {
        "calibration_n": len(confs),
        "avg_confidence": round(sum(confs) / len(confs), 4),
        "avg_abs_error": round(sum(errs) / len(errs), 4),
    }