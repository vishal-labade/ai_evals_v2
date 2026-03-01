from __future__ import annotations
import json
from typing import Any, Dict, List, Optional


def _txt(x: Any) -> str:
    return "" if x is None else str(x)


def _bullets(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip().startswith("-")]


def _json_obj(text: str) -> Optional[dict]:
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def score_instruction_persistence(scenario: Dict[str, Any], turn_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    fmt = (scenario.get("checks") or {}).get("format") or {}
    ftype = fmt.get("type")
    if not ftype:
        return {"format_pass": 0, "format_total": 0, "format_score": 0.0}

    target_turn = fmt.get("turn_index", None)

    passed = 0
    total = 0
    for rec in turn_records:
        if target_turn is not None and int(rec.get("turn_index", -1)) != int(target_turn):
            continue

        total += 1
        out = _txt(rec.get("assistant_text"))
        ok = True

        if ftype == "bullets_exact":
            ok = len(_bullets(out)) == int(fmt.get("n_bullets", 0))
        elif ftype == "bullets_min_max":
            nb = len(_bullets(out))
            ok = int(fmt.get("min_bullets", 0)) <= nb <= int(fmt.get("max_bullets", 10**9))
        elif ftype == "json_keys_present":
            obj = _json_obj(out)
            req = list(fmt.get("required_keys") or [])
            ok = obj is not None and all(k in obj for k in req)

        if ok:
            passed += 1

    score = (passed / total) if total else 0.0
    return {"format_pass": passed, "format_total": total, "format_score": round(score, 4)}