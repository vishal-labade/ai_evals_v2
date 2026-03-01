from __future__ import annotations
import re
from typing import Any, Dict, List


def _txt(x: Any) -> str:
    return "" if x is None else str(x)


_YES = {"yes", "yeah", "yep", "true", "correct"}
_NO = {"no", "nope", "false", "incorrect"}


def _first_token(text: str) -> str:
    t = text.strip().lower()
    if not t:
        return ""
    return re.split(r"\s+", t, maxsplit=1)[0]


def _first_number(text: str) -> str:
    m = re.search(r"(-?\d+(?:\.\d+)?)", text)
    return m.group(1) if m else ""


def score_consistency_contradiction(scenario: Dict[str, Any], turn_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    outs = [_txt(r.get("assistant_text")) for r in turn_records]
    if len(outs) < 2:
        return {"contradiction_hits": 0, "contradiction_total_pairs": 0, "contradiction_rate": 0.0}

    hits = 0
    pairs = 0
    for i in range(len(outs) - 1):
        a, b = outs[i], outs[i + 1]
        pairs += 1

        fa, fb = _first_token(a), _first_token(b)
        if (fa in _YES and fb in _NO) or (fa in _NO and fb in _YES):
            hits += 1
            continue

        na, nb = _first_number(a), _first_number(b)
        if na and nb and na != nb:
            hits += 1

    rate = hits / pairs if pairs else 0.0
    return {"contradiction_hits": hits, "contradiction_total_pairs": pairs, "contradiction_rate": round(rate, 4)}