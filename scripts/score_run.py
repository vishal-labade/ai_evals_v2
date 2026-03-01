#!/usr/bin/env python3
import argparse
import csv
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jsonschema

# -----------------------------
# Numeric extraction + invention proxy (existing behavior)
# -----------------------------

NUM_RE = re.compile(r"(?<!\w)(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?(?!\w)")
FIRST_NUM_RE = re.compile(r"[-+]?\d[\d,]*\.?\d*")

def extract_numbers(text: str):
    return [m.group(0) for m in NUM_RE.finditer(text or "")]

def normalize_num(s: str) -> str:
    return (s or "").replace(",", "").strip().lower()

def numeric_invention_flag(output_text: str, context_text: str) -> int:
    out_nums = {normalize_num(x) for x in extract_numbers(output_text)}
    ctx_nums = {normalize_num(x) for x in extract_numbers(context_text)}
    if not out_nums:
        return 0
    return int(len(out_nums - ctx_nums) > 0)

# -----------------------------
# JSONL utilities
# -----------------------------

def read_jsonl(path: str):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

def load_prompts(prompts_path: str) -> Dict[str, Dict[str, Any]]:
    prompts = {}
    for p in read_jsonl(prompts_path):
        pid = p.get("prompt_id")
        if pid:
            prompts[pid] = p
    return prompts

def load_schema(schema_index_path: str, schema_name: str) -> Dict[str, Any]:
    with open(schema_index_path, "r", encoding="utf-8") as f:
        index = json.load(f)
    if schema_name not in index:
        raise KeyError(f"schema_name '{schema_name}' not found in {schema_index_path}")
    schema_path = index[schema_name]
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

# -----------------------------
# JSON parsing helpers
# -----------------------------

FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE | re.MULTILINE)

def strip_code_fences(text: str) -> str:
    t = (text or "").strip()
    if "```" in t:
        t = FENCE_RE.sub("", t).strip()
    return t

def try_parse_json(text: str, lenient: bool) -> Tuple[bool, Optional[Any], str]:
    raw = text or ""
    if lenient:
        raw = strip_code_fences(raw)
    try:
        return True, json.loads(raw), raw
    except Exception:
        return False, None, raw

# -----------------------------
# Scoring: exact / contains / regex
# -----------------------------

def score_exact_strict(output_text: str, ground_truth: Any) -> int:
    return int((output_text or "").strip() == str(ground_truth).strip())

def score_contains(output_text: str, ground_truth: Any) -> int:
    return int(str(ground_truth).strip().lower() in (output_text or "").strip().lower())

def score_regex(output_text: str, pattern: str) -> int:
    return int(re.search(pattern, output_text or "") is not None)

# -----------------------------
# Canonical numeric scoring (optional)
# -----------------------------

def _norm_space(s: str) -> str:
    return " ".join(str(s).strip().split())

def _extract_first_number(s: str) -> Optional[str]:
    m = FIRST_NUM_RE.search(str(s))
    return m.group(0) if m else None

def _parse_number(num_str: str) -> Optional[float]:
    try:
        return float(num_str.replace(",", ""))
    except Exception:
        return None

def _normalize_unit(u: str) -> str:
    u = _norm_space(u).lower()
    u = u.replace("percent", "%")
    u = u.replace("us dollars", "usd").replace("dollars", "usd")
    return _norm_space(u)

def _split_value_unit(s: str) -> Tuple[Optional[float], str]:
    s = _norm_space(s)
    num = _extract_first_number(s)
    if not num:
        return (None, "")
    v = _parse_number(num)
    if v is None:
        return (None, "")
    idx = s.find(num) + len(num)
    unit = _normalize_unit(s[idx:])
    return (v, unit)

def canonical_numeric_match(pred: str, gt: str, rel_tol: float = 0.0, abs_tol: float = 1e-9) -> bool:
    pv, pu = _split_value_unit(pred)
    gv, gu = _split_value_unit(gt)
    if pv is None or gv is None:
        return False
    diff = abs(pv - gv)
    if diff > max(abs_tol, rel_tol * abs(gv)):
        return False
    if gu:
        return pu == gu
    return True

def score_exact_with_policy(output_text: str, ground_truth: Any, prompt_obj: Dict[str, Any], default_policy: str) -> int:
    pred = (output_text or "").strip()
    gt = str(ground_truth).strip()

    # Never canonical-score NOT_IN_CONTEXT
    if gt == "NOT_IN_CONTEXT":
        return int(pred == "NOT_IN_CONTEXT")

    md = prompt_obj.get("metadata") or {}
    pol = ""
    if isinstance(md, dict):
        pol = (md.get("numeric_policy") or "").strip().lower()
    if not pol:
        pol = default_policy.strip().lower()

    if pol == "canonical":
        return int(canonical_numeric_match(pred, gt))
    return int(pred == gt)

# -----------------------------
# NEW: inherit missing fields from base_prompt_id
# -----------------------------

INHERIT_KEYS = [
    "task", "domain", "difficulty", "category",
    "suite", "suite_version",
    "scoring",
]

def _is_blank(x: Any) -> bool:
    if x is None:
        return True
    if isinstance(x, str) and x.strip() == "":
        return True
    return False

def _enrich_from_base(prompt: Dict[str, Any], prompts_index: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    If an expanded prompt row is missing task/domain/difficulty/category (etc),
    inherit from its base_prompt_id when available.
    """
    if not isinstance(prompt, dict):
        return prompt

    base_id = prompt.get("base_prompt_id")
    if not base_id:
        return prompt

    base = prompts_index.get(base_id)
    if not isinstance(base, dict):
        return prompt

    out = dict(prompt)
    for k in INHERIT_KEYS:
        if _is_blank(out.get(k)) and not _is_blank(base.get(k)):
            out[k] = base.get(k)
    return out

# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="data/prompts/v1_suite_big.jsonl")
    ap.add_argument("--run", required=True, help="Path to results/runs/<run_id>.jsonl")
    ap.add_argument("--schema-index", default="data/prompts/schemas/index.json")
    ap.add_argument("--out", default=None, help="Output CSV path (default: results/metrics/<run_id>.csv)")
    ap.add_argument(
        "--default_numeric_policy",
        choices=["strict", "canonical"],
        default="strict",
        help="Used only when prompt metadata.numeric_policy is missing. NOT_IN_CONTEXT is always strict.",
    )
    args = ap.parse_args()

    prompts = load_prompts(args.prompts)

    run_path = args.run
    run_id = os.path.splitext(os.path.basename(run_path))[0]

    out_path = args.out or f"results/metrics/{run_id}.csv"
    ensure_dir(os.path.dirname(out_path))

    fieldnames = [
        "run_id", "model_name", "prompt_id",
        "task", "domain", "difficulty", "category",
        "scoring_method", "schema_name",
        "score",
        "json_valid_strict", "json_valid_lenient",
        "schema_valid_strict", "schema_valid_lenient",
        "latency_ms",
        "prompt_tokens", "completion_tokens", "total_tokens",
        "error",
        "not_in_context_violation",
        "numeric_invention_flag",
        "temperature",
        "num_predict",
        "num_ctx",
    ]

    with open(out_path, "w", newline="", encoding="utf-8") as out_f:
        w = csv.DictWriter(out_f, fieldnames=fieldnames)
        w.writeheader()

        for row in read_jsonl(run_path):
            pid = row.get("prompt_id")
            p = prompts.get(pid)

            if not p:
                w.writerow({
                    "run_id": row.get("run_id"),
                    "model_name": row.get("model_name"),
                    "prompt_id": pid,
                    "score": 0,
                    "error": "prompt_id_not_found_in_suite",
                })
                continue

            # NEW: backfill task/domain/difficulty/category/etc for variants
            p = _enrich_from_base(p, prompts)

            scoring = p.get("scoring", {}) or {}
            method = scoring.get("method")
            schema_name = scoring.get("schema_name")

            output_text = row.get("output_text", "") or ""
            ground_truth = p.get("ground_truth")

            score = 0
            json_valid_strict = 0
            json_valid_lenient = 0
            schema_valid_strict = 0
            schema_valid_lenient = 0

            error_tag = row.get("error", "") or ""

            try:
                if method == "exact":
                    score = score_exact_with_policy(
                        output_text=output_text,
                        ground_truth=ground_truth,
                        prompt_obj=p,
                        default_policy=args.default_numeric_policy,
                    )
                elif method == "contains":
                    score = score_contains(output_text, ground_truth)
                elif method == "regex":
                    score = score_regex(output_text, scoring.get("pattern", ""))
                elif method == "json_schema":
                    ok_s, obj_s, _ = try_parse_json(output_text, lenient=False)
                    ok_l, obj_l, _ = try_parse_json(output_text, lenient=True)

                    json_valid_strict = int(ok_s)
                    json_valid_lenient = int(ok_l)

                    if schema_name:
                        schema = load_schema(args.schema_index, schema_name)
                        if ok_s:
                            try:
                                jsonschema.validate(instance=obj_s, schema=schema)
                                schema_valid_strict = 1
                            except Exception:
                                schema_valid_strict = 0
                        if ok_l:
                            try:
                                jsonschema.validate(instance=obj_l, schema=schema)
                                schema_valid_lenient = 1
                            except Exception:
                                schema_valid_lenient = 0

                    score = int(schema_valid_lenient == 1)
                else:
                    score = 0
                    if not error_tag:
                        error_tag = "unknown_scoring_method"
            except Exception as e:
                # Never emit blank score
                score = 0
                tag = f"scoring_error:{type(e).__name__}"
                error_tag = f"{error_tag};{tag}".strip(";")

            usage = row.get("usage", {}) or {}
            params = row.get("params", {}) or {}

            context = p.get("context", "") or ""
            gt = p.get("ground_truth")

            not_in_ctx_violation = 0
            if isinstance(gt, str) and gt.strip() == "NOT_IN_CONTEXT":
                not_in_ctx_violation = int(output_text.strip() != "NOT_IN_CONTEXT")

            num_invention = 0
            if p.get("category") == "open_book_extraction" and context:
                num_invention = numeric_invention_flag(output_text, context)

            out_row = {
                "run_id": row.get("run_id"),
                "model_name": row.get("model_name"),
                "prompt_id": pid,
                "task": p.get("task"),
                "domain": p.get("domain"),
                "difficulty": p.get("difficulty"),
                "category": p.get("category"),
                "scoring_method": method,
                "schema_name": schema_name,
                "score": int(score),
                "json_valid_strict": int(json_valid_strict),
                "json_valid_lenient": int(json_valid_lenient),
                "schema_valid_strict": int(schema_valid_strict),
                "schema_valid_lenient": int(schema_valid_lenient),
                "latency_ms": row.get("latency_ms"),
                "prompt_tokens": usage.get("prompt_tokens"),
                "completion_tokens": usage.get("completion_tokens"),
                "total_tokens": usage.get("total_tokens"),
                "error": error_tag,
                "not_in_context_violation": int(not_in_ctx_violation),
                "numeric_invention_flag": int(num_invention),
                "temperature": params.get("temperature"),
                "num_predict": params.get("num_predict"),
                "num_ctx": params.get("num_ctx"),
            }
            w.writerow(out_row)

    print(f"Wrote metrics: {out_path}")

if __name__ == "__main__":
    main()