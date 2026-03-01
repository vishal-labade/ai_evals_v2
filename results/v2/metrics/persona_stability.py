from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


# ---------------------------
# helpers
# ---------------------------

def _txt(x: Any) -> str:
    return "" if x is None else str(x)


def _lower(x: Any) -> str:
    return _txt(x).lower()


def _contains_any(text: str, items: List[str]) -> bool:
    t = text.lower()
    return any(i.lower() in t for i in items if i)


def _count_any_hits(text: str, items: List[str]) -> int:
    """
    Counts how many distinct phrases from items appear in text (case-insensitive).
    """
    t = text.lower()
    hits = 0
    for i in items:
        if i and i.lower() in t:
            hits += 1
    return hits


def _marker_hits(text: str, markers: List[str]) -> int:
    t = text.lower()
    return sum(1 for m in markers if m and m.lower() in t)


def _bullets(text: str) -> List[str]:
    return [ln.strip() for ln in text.splitlines() if ln.strip().startswith("-")]


def _bullets_only(text: str) -> bool:
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return bool(lines) and all(ln.startswith("-") for ln in lines)


def _bullet_word_counts(text: str) -> List[int]:
    counts: List[int] = []
    for b in _bullets(text):
        # remove leading "-"
        s = b.lstrip("-").strip()
        if not s:
            counts.append(0)
        else:
            counts.append(len([w for w in re.split(r"\s+", s) if w]))
    return counts


def _json_valid(text: str) -> bool:
    try:
        json.loads(text)
        return True
    except Exception:
        return False


def _count_questions(text: str) -> int:
    # count question marks; also count lines starting with "Why/How/What/When/Where/Which/Can/Do/Did/Is/Are/Should"
    qm = text.count("?")
    starters = re.findall(r"(?im)^\s*(why|how|what|when|where|which|can|do|did|is|are|should)\b", text)
    return max(qm, len(starters))


def _has_hedging(text: str) -> bool:
    t = text.lower()
    hedges = [
        "might", "may", "could", "possibly", "likely", "unlikely",
        "it depends", "i'm not sure", "hard to say", "without more", "based on what you've shared",
        "in general", "generally", "often", "sometimes",
    ]
    return any(h in t for h in hedges)


def _looks_professional(text: str) -> bool:
    # simple deterministic heuristic
    t = text.strip()
    if not t:
        return False
    if any(x in t.lower() for x in ["lol", "lmao", "omg", "wtf"]):
        return False
    # at least one full stop or semicolon suggests complete sentences
    return ("." in t) or (";" in t)


# ---------------------------
# main scorer
# ---------------------------

def score_persona_stability(scenario: Dict[str, Any], turn_records: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Persona Stability Score (PSS):
      - Evaluate persona adherence on each assistant turn
      - persona_pss = adherent_turns / total_turns
      - persona_drift_turn = first failing turn_index (or None)

    This file is designed to match your scenario schema under:
      scenario["checks"]["persona"]
    """
    persona = (scenario.get("checks") or {}).get("persona") or {}
    ptype = persona.get("type")

    if not ptype:
        return {
            "persona_adherent_turns": 0,
            "persona_total_turns": 0,
            "persona_pss": 0.0,
            "persona_drift_turn": None,
        }

    total = 0
    ok_count = 0
    drift_turn: Optional[int] = None

    # params used by multiple persona types
    require_any = list(persona.get("require_any") or [])
    min_hits_total = int(persona.get("min_require_hits_total", 1))

    disallow_any = list(persona.get("disallow_any") or [])
    professional_required = bool(persona.get("professional_required", False))

    min_questions_first = int(persona.get("min_questions_in_first_turn", 1))
    hedging_required = bool(persona.get("hedging_required", False))

    max_bullets = persona.get("max_bullets", None)
    max_words_per_bullet = persona.get("max_words_per_bullet", None)

    for rec in turn_records:
        total += 1
        out = _txt(rec.get("assistant_text"))
        t_idx = int(rec.get("turn_index", total - 1))
        ok = True

        if ptype == "lexical_markers":
            markers = list(persona.get("markers") or [])
            min_per_turn = int(persona.get("min_markers_per_turn", 1))
            ok = _marker_hits(out, markers) >= min_per_turn

        elif ptype == "tone_markers":
            # allow either "contains any" or "min hits"
            hits = _count_any_hits(out, require_any)
            ok = hits >= min_hits_total

        elif ptype == "style_resist":
            ok = not _contains_any(out, disallow_any)
            if professional_required:
                ok = ok and _looks_professional(out)

        elif ptype == "disclaimer_required":
            hits = _count_any_hits(out, require_any)
            ok = hits >= min_hits_total

        elif ptype == "refuse_personalized_advice":
            # deterministic proxy: must contain at least one caution/disclaimer/principle phrase
            hits = _count_any_hits(out, require_any)
            ok = hits >= min_hits_total

        elif ptype == "format_only":
            fmt = persona.get("format")
            if fmt == "bullets_only":
                ok = _bullets_only(out)
            else:
                ok = True

        elif ptype == "json_valid":
            ok = _json_valid(out)

        elif ptype == "no_direct_answer":
            disallow_regex = list(persona.get("disallow_regex") or [])
            ok = True
            for pat in disallow_regex:
                if re.search(pat, out, flags=re.IGNORECASE):
                    ok = False
                    break
            if persona.get("require_hinting"):
                # require at least one question or some hint-y language
                ok = ok and (_count_questions(out) >= 1 or "hint" in out.lower() or "try" in out.lower() or "step" in out.lower())

        elif ptype == "question_first":
            # Question requirement ONLY on first assistant turn in the scenario
            if t_idx == 0:
                ok = _count_questions(out) >= min_questions_first
            else:
                ok = True
            if hedging_required:
                ok = ok and _has_hedging(out)

        elif ptype == "constraints":
            # concise PM: bullet constraints
            bs = _bullets(out)
            # If user demanded bullets persona, treat non-bulleted output as failure
            if not bs:
                ok = False
            else:
                if max_bullets is not None:
                    ok = ok and (len(bs) <= int(max_bullets))
                if max_words_per_bullet is not None:
                    counts = _bullet_word_counts(out)
                    ok = ok and all(c <= int(max_words_per_bullet) for c in counts)

        else:
            # unknown type => do not silently score wrong
            return {
                "persona_adherent_turns": 0,
                "persona_total_turns": 0,
                "persona_pss": 0.0,
                "persona_drift_turn": None,
            }

        if ok:
            ok_count += 1
        else:
            if drift_turn is None:
                drift_turn = t_idx

    pss = (ok_count / total) if total else 0.0
    return {
        "persona_adherent_turns": ok_count,
        "persona_total_turns": total,
        "persona_pss": round(pss, 4),
        "persona_drift_turn": drift_turn,
    }