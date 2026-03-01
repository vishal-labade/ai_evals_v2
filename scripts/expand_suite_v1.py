#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple, Optional


# -----------------------------
# Suite schema (native to yours)
# -----------------------------

PROMPT_FIELD = "prompt"
CONTEXT_FIELD = "context"
GT_FIELD = "ground_truth"
PID_FIELD = "prompt_id"

SCORING_FIELD = "scoring"          # dict
SCORING_METHOD_KEY = "method"      # exact | json_schema
SCHEMA_NAME_KEY = "schema_name"    # for json_schema

METADATA_FIELD = "metadata"
FACTS_KEY = "facts"                # list of {field,value,unit,date}


# -----------------------------
# Utilities
# -----------------------------

def read_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str, rows: Iterable[Dict[str, Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_text(json.dumps(obj, sort_keys=True, ensure_ascii=False))


def stable_sort_key(row: Dict[str, Any]) -> Tuple[str, str]:
    base = row.get("base_prompt_id") or row.get(PID_FIELD, "")
    pid = row.get(PID_FIELD, "")
    return (str(base), str(pid))


def ensure_unique_prompt_ids(rows: List[Dict[str, Any]]) -> None:
    seen = set()
    dups = []
    for r in rows:
        pid = r.get(PID_FIELD)
        if pid in seen:
            dups.append(pid)
        seen.add(pid)
    if dups:
        raise SystemExit(f"Duplicate prompt_id(s) detected: {dups[:10]} (showing up to 10)")


def scoring_method(row: Dict[str, Any]) -> str:
    sc = row.get(SCORING_FIELD, {}) or {}
    return str(sc.get(SCORING_METHOD_KEY, "")).strip()


def schema_name(row: Dict[str, Any]) -> str:
    sc = row.get(SCORING_FIELD, {}) or {}
    return str(sc.get(SCHEMA_NAME_KEY, "")).strip()


def get_facts(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    md = row.get(METADATA_FIELD, {}) or {}
    facts = md.get(FACTS_KEY, []) or []
    if not isinstance(facts, list):
        return []
    return facts


def is_not_in_context_prompt(row: Dict[str, Any]) -> bool:
    # Your v1 uses exact scoring and ground_truth == "NOT_IN_CONTEXT"
    return scoring_method(row) == "exact" and str(row.get(GT_FIELD, "")).strip() == "NOT_IN_CONTEXT"


def normalize_number_str(s: str) -> str:
    # used only for guardrail comparisons (not rewriting)
    return s.replace(",", "").strip()


# -----------------------------
# Paraphrase expansions (safe)
# -----------------------------

# IMPORTANT: keep these deterministic, safe, and semantically neutral.
# These are *prefixes* that we prepend to the original prompt.
DEFAULT_PARAPHRASES = [
    "Use only the provided context. Do not use outside knowledge.",
    "Answer using only the context above. Do not infer or guess.",
    "Rely strictly on the context. Do not add any extra text beyond what is requested.",
    "Use strictly the context above. If the value is missing, follow the instruction exactly.",
    "Only use the given context. Do not use memory or external information.",
    "Use only the context. If something is not present, follow the prompt’s missing-value instruction.",
    "Base your answer solely on the context provided. Do not make assumptions.",
    "Respond using only the information in the context. Do not add commentary.",
    "Use only the context above. Do not introduce new facts.",
    "Answer strictly from the context. Do not compute beyond what is explicitly stated.",
    "Follow the output format exactly and use only the context above.",
    "Do not use outside knowledge. Use only the context and follow the instructions precisely.",
    "Rely exclusively on the provided context. Do not embellish.",
    "Use the context verbatim where possible. Do not add explanations.",
    "Only extract what is explicitly present in the context.",
    "Answer using only the context. If the prompt asks for a specific format, comply exactly.",
    "Do not guess. Use only the context and return only what is requested.",
    "Answer strictly from context. Avoid any additional words or formatting.",
    "Use only the above context; do not infer missing numbers or units.",
    "Respond based solely on the context. No external assumptions.",
]

def paraphrase_expand(row: Dict[str, Any], paraphrases: List[str], max_variants: int) -> List[Dict[str, Any]]:
    base_pid = row[PID_FIELD]
    base_prompt = str(row[PROMPT_FIELD])

    out = []
    for i, p in enumerate(paraphrases[:max_variants], start=1):
        new = dict(row)
        new_pid = f"{base_pid}__paraphrase__v{i:02d}"

        new[PID_FIELD] = new_pid
        new["base_prompt_id"] = base_pid
        new["expansion_type"] = "paraphrase"
        new["variant_index"] = i

        # Deterministic: prepend paraphrase, preserve original prompt verbatim
        new[PROMPT_FIELD] = f"{p}\n\n{base_prompt}"

        out.append(new)
    return out


# -----------------------------
# Format/constraint expansions (safe + deterministic)
# -----------------------------

# These variants tighten output constraints without changing the task.
# They should be compatible with your existing deterministic scorer.
FORMAT_VARIANTS = [
    # For exact extraction prompts: “only value and unit / only number” already present in base,
    # but this reinforces format compliance.
    "Formatting rule: Output must contain only the final answer. No explanations.",
    "Formatting rule: Do not use markdown, bullet points, or code fences. Output only the answer.",
    "Formatting rule: Do not repeat the question or the context. Output only the answer.",
    "Formatting rule: No leading/trailing text. Output exactly the requested value.",
    "Formatting rule: Do not add units unless asked. Follow the prompt’s output instruction exactly.",
    "Formatting rule: If JSON is requested, output raw JSON only (no code fences, no extra keys).",
    "Formatting rule: Preserve commas and punctuation as shown in the context/ground truth.",
    "Formatting rule: Do not round or reformat numbers. Copy the exact value as stated in context.",
    "Formatting rule: Do not add a trailing period or newline beyond the answer itself.",
    "Formatting rule: Use the exact casing and spelling for entity names as in the context.",
]

def format_expand(row: Dict[str, Any], variants: List[str], max_variants: int) -> List[Dict[str, Any]]:
    base_pid = row[PID_FIELD]
    base_prompt = str(row[PROMPT_FIELD])

    out = []
    for i, v in enumerate(variants[:max_variants], start=1):
        new = dict(row)
        new_pid = f"{base_pid}__format__v{i:02d}"

        new[PID_FIELD] = new_pid
        new["base_prompt_id"] = base_pid
        new["expansion_type"] = "format"
        new["variant_index"] = i
        new["format_variant"] = v

        # Deterministic: prepend format rule, preserve original prompt verbatim
        new[PROMPT_FIELD] = f"{v}\n\n{base_prompt}"

        out.append(new)
    return out


# -----------------------------
# Numeric perturbations (safe + deterministic)
# -----------------------------

@dataclass(frozen=True)
class NumericRule:
    # apply to numeric facts (value) and to any matching ground_truth string (for exact)
    add_choices: Tuple[float, ...] = (0.0, 1.0, -1.0, 2.0, -2.0, 5.0, -5.0)
    mul_choices: Tuple[float, ...] = (1.0, 1.05, 0.95, 1.1, 0.9)

def _parse_float_maybe(x: str) -> Optional[float]:
    try:
        return float(normalize_number_str(x))
    except Exception:
        return None

def _format_value_like(original: str, new_value: float) -> str:
    """
    Product-grade rule: preserve original formatting as much as possible.
    - If original has commas, keep comma formatting for integers.
    - Preserve decimal places if original had decimals.
    """
    orig = str(original).strip()
    has_comma = "," in orig
    if "." in orig:
        # preserve decimal places count
        decimals = len(orig.split(".")[1])
        s = f"{new_value:.{decimals}f}"
    else:
        # integer-like
        s = str(int(round(new_value)))

    if has_comma:
        # add commas to integer part only
        if "." in s:
            a, b = s.split(".", 1)
            a = f"{int(a):,}"
            s = f"{a}.{b}"
        else:
            s = f"{int(s):,}"
    return s

def numeric_expand_exact(row: Dict[str, Any], rule: NumericRule, seed: int, max_variants: int) -> List[Dict[str, Any]]:
    """
    Safe numeric expansion for exact-scoring prompts where:
    - metadata.facts contains at least one fact with numeric 'value'
    - ground_truth matches "<value> <unit>" OR equals value-only
    We mutate the matched fact value in both context and ground_truth deterministically.
    """
    if scoring_method(row) != "exact":
        return []
    if is_not_in_context_prompt(row):
        # Guardrail: never perturb NOT_IN_CONTEXT prompts
        return []

    facts = get_facts(row)
    if not facts:
        return []

    # Find the fact that appears in ground_truth (value + unit match preferred)
    gt = str(row.get(GT_FIELD, "")).strip()

    chosen_idx = None
    chosen_fact = None
    for idx, f in enumerate(facts):
        v = str(f.get("value", "")).strip()
        u = str(f.get("unit", "")).strip()
        candidate1 = f"{v} {u}".strip()
        if gt == candidate1 or gt == v:
            chosen_idx = idx
            chosen_fact = f
            break

    if chosen_fact is None:
        return []

    v0_str = str(chosen_fact.get("value", "")).strip()
    u0_str = str(chosen_fact.get("unit", "")).strip()
    v0 = _parse_float_maybe(v0_str)
    if v0 is None:
        return []

    base_pid = row[PID_FIELD]
    ctx0 = str(row.get(CONTEXT_FIELD, ""))

    # deterministic variant generation
    rng = random.Random(seed + int(row.get("metadata", {}).get("seed", 0) or 0))

    variants = []
    # build operations list deterministically
    ops = []
    for a in rule.add_choices:
        if a != 0.0:
            ops.append(("add", a))
    for m in rule.mul_choices:
        if m != 1.0:
            ops.append(("mul", m))
    rng.shuffle(ops)

    for i, (op, amt) in enumerate(ops[:max_variants], start=1):
        new = dict(row)
        new_pid = f"{base_pid}__numeric__v{i:02d}"

        # compute new value
        if op == "add":
            v1 = v0 + float(amt)
        else:
            v1 = v0 * float(amt)

        # format similar to original fact value
        v1_str = _format_value_like(v0_str, v1)

        # rewrite context: replace exact "value unit" if possible, else value alone
        old_token = f"{v0_str} {u0_str}".strip()
        new_token = f"{v1_str} {u0_str}".strip()
        ctx1 = ctx0.replace(old_token, new_token)
        if ctx1 == ctx0:
            # fallback: replace just the numeric value
            ctx1 = ctx0.replace(v0_str, v1_str)

        if ctx1 == ctx0:
            # Guardrail: if we couldn't rewrite deterministically, skip variant
            continue

        # rewrite ground_truth correspondingly
        if gt == old_token:
            gt1 = new_token
        elif gt == v0_str:
            gt1 = v1_str
        else:
            continue

        new[PID_FIELD] = new_pid
        new["base_prompt_id"] = base_pid
        new["expansion_type"] = "numeric"
        new["variant_index"] = i
        new["numeric_op"] = {"type": op, "amount": amt}
        new["numeric_anchor"] = {"fact_index": chosen_idx, "field": chosen_fact.get("field"), "unit": u0_str}

        new[CONTEXT_FIELD] = ctx1
        new[GT_FIELD] = gt1

        # update metadata facts (so scorer/debug can use it)
        md = dict(new.get(METADATA_FIELD, {}) or {})
        facts1 = [dict(x) for x in (md.get(FACTS_KEY, []) or [])]
        if 0 <= chosen_idx < len(facts1):
            facts1[chosen_idx]["value"] = str(v1_str)
        md[FACTS_KEY] = facts1
        new[METADATA_FIELD] = md

        variants.append(new)

    return variants


def numeric_expand_json_schema(row: Dict[str, Any], rule: NumericRule, seed: int, max_variants: int) -> List[Dict[str, Any]]:
    """
    Numeric expansion for json_schema prompts:
    - Uses metadata.facts as anchor to find a fact that appears in ground_truth dict as a string.
    - Rewrites both context and that ground_truth field string.
    - Keeps schema_name unchanged (guardrail).
    """
    if scoring_method(row) != "json_schema":
        return []

    facts = get_facts(row)
    if not facts:
        return []

    gt_obj = row.get(GT_FIELD)
    if not isinstance(gt_obj, dict):
        return []

    # find a fact whose "value unit" appears as an exact string in gt dict values
    chosen_idx = None
    chosen_fact = None
    chosen_gt_key = None

    for idx, f in enumerate(facts):
        v = str(f.get("value", "")).strip()
        u = str(f.get("unit", "")).strip()
        token = f"{v} {u}".strip() if u else v
        for k, vv in gt_obj.items():
            if str(vv).strip() == token or str(vv).strip() == v:
                chosen_idx = idx
                chosen_fact = f
                chosen_gt_key = k
                break
        if chosen_fact:
            break

    if chosen_fact is None:
        return []

    v0_str = str(chosen_fact.get("value", "")).strip()
    u0_str = str(chosen_fact.get("unit", "")).strip()
    v0 = _parse_float_maybe(v0_str)
    if v0 is None:
        return []

    base_pid = row[PID_FIELD]
    ctx0 = str(row.get(CONTEXT_FIELD, ""))
    schema0 = schema_name(row)

    rng = random.Random(seed + int(row.get("metadata", {}).get("seed", 0) or 0))

    ops = []
    for a in rule.add_choices:
        if a != 0.0:
            ops.append(("add", a))
    for m in rule.mul_choices:
        if m != 1.0:
            ops.append(("mul", m))
    rng.shuffle(ops)

    variants = []
    for i, (op, amt) in enumerate(ops[:max_variants], start=1):
        new = dict(row)
        new_pid = f"{base_pid}__numeric__v{i:02d}"

        if op == "add":
            v1 = v0 + float(amt)
        else:
            v1 = v0 * float(amt)

        v1_str = _format_value_like(v0_str, v1)
        old_token = f"{v0_str} {u0_str}".strip() if u0_str else v0_str
        new_token = f"{v1_str} {u0_str}".strip() if u0_str else v1_str

        ctx1 = ctx0.replace(old_token, new_token)
        if ctx1 == ctx0:
            ctx1 = ctx0.replace(v0_str, v1_str)
        if ctx1 == ctx0:
            continue

        gt1 = dict(gt_obj)
        # rewrite only the matched field
        if str(gt1.get(chosen_gt_key, "")).strip() == old_token:
            gt1[chosen_gt_key] = new_token
        elif str(gt1.get(chosen_gt_key, "")).strip() == v0_str:
            gt1[chosen_gt_key] = v1_str
        else:
            continue

        # Guardrail: preserve schema_name exactly
        sc = dict(new.get(SCORING_FIELD, {}) or {})
        sc[SCHEMA_NAME_KEY] = schema0
        new[SCORING_FIELD] = sc

        new[PID_FIELD] = new_pid
        new["base_prompt_id"] = base_pid
        new["expansion_type"] = "numeric"
        new["variant_index"] = i
        new["numeric_op"] = {"type": op, "amount": amt}
        new["numeric_anchor"] = {"fact_index": chosen_idx, "field": chosen_fact.get("field"), "unit": u0_str, "gt_key": chosen_gt_key}

        new[CONTEXT_FIELD] = ctx1
        new[GT_FIELD] = gt1

        # update metadata facts
        md = dict(new.get(METADATA_FIELD, {}) or {})
        facts1 = [dict(x) for x in (md.get(FACTS_KEY, []) or [])]
        if 0 <= chosen_idx < len(facts1):
            facts1[chosen_idx]["value"] = str(v1_str)
        md[FACTS_KEY] = facts1
        new[METADATA_FIELD] = md

        variants.append(new)

    return variants


# -----------------------------
# Guardrails (product-grade)
# -----------------------------

def validate_expansion(base_rows: List[Dict[str, Any]], out_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Hard checks + helpful stats. Raises on failures.
    """
    ensure_unique_prompt_ids(out_rows)

    base_by_id = {r[PID_FIELD]: r for r in base_rows}

    # 1) Original prompts must be unchanged if included
    for r in out_rows:
        if r.get("expansion_type") == "original":
            bid = r.get("base_prompt_id") or r.get(PID_FIELD)
            if bid not in base_by_id:
                raise SystemExit(f"original row has unknown base_prompt_id: {bid}")
            b = base_by_id[bid]
            for k in [PROMPT_FIELD, CONTEXT_FIELD, GT_FIELD, SCORING_FIELD, "task", "category", "domain", "difficulty"]:
                if b.get(k) != r.get(k):
                    raise SystemExit(f"Original row mutated for {bid} field={k}")

    # 2) NOT_IN_CONTEXT prompts: must stay NOT_IN_CONTEXT and exact scoring
    for r in out_rows:
        bid = r.get("base_prompt_id")
        if bid and bid in base_by_id and is_not_in_context_prompt(base_by_id[bid]):
            if str(r.get(GT_FIELD, "")).strip() != "NOT_IN_CONTEXT":
                raise SystemExit(f"NOT_IN_CONTEXT base expanded into non-NOT_IN_CONTEXT: {r[PID_FIELD]}")
            if scoring_method(r) != "exact":
                raise SystemExit(f"NOT_IN_CONTEXT base expanded into non-exact scoring: {r[PID_FIELD]}")

    # 3) json_schema prompts: schema_name must not change
    for r in out_rows:
        bid = r.get("base_prompt_id")
        if bid and bid in base_by_id and scoring_method(base_by_id[bid]) == "json_schema":
            if schema_name(r) != schema_name(base_by_id[bid]):
                raise SystemExit(f"schema_name changed for {r[PID_FIELD]}")

    # 4) Exact prompts that have units in GT: ensure unit preserved in numeric expansions
    for r in out_rows:
        if r.get("expansion_type") == "numeric" and scoring_method(r) == "exact":
            bid = r.get("base_prompt_id")
            if not bid or bid not in base_by_id:
                continue
            gt0 = str(base_by_id[bid].get(GT_FIELD, "")).strip()
            gt1 = str(r.get(GT_FIELD, "")).strip()
            if " " in gt0:
                unit0 = " ".join(gt0.split(" ")[1:])
                unit1 = " ".join(gt1.split(" ")[1:]) if " " in gt1 else ""
                if unit0 != unit1:
                    raise SystemExit(
                        f"Unit changed in numeric exact expansion {r[PID_FIELD]}: '{unit0}' -> '{unit1}'"
                    )

    counts = {}
    for r in out_rows:
        et = r.get("expansion_type", "unknown")
        counts[et] = counts.get(et, 0) + 1

    return {"counts_by_type": counts, "n_out": len(out_rows), "n_base": len(base_rows)}


# -----------------------------
# Manifest
# -----------------------------

def write_manifest(path: str, base_rows: List[Dict[str, Any]], out_rows: List[Dict[str, Any]], cfg: Dict[str, Any], validation: Dict[str, Any]) -> None:
    base_hash = sha256_json([base_rows[0][PID_FIELD], len(base_rows)])  # stable-ish
    out_hash = sha256_text("".join(r[PID_FIELD] for r in sorted(out_rows, key=stable_sort_key)))

    m = {
        "suite": (base_rows[0].get("suite") if base_rows else ""),
        "suite_version": (base_rows[0].get("suite_version") if base_rows else ""),
        "base_count": len(base_rows),
        "expanded_count": len(out_rows),
        "counts_by_type": validation.get("counts_by_type", {}),
        "config": cfg,
        "base_fingerprint": base_hash,
        "expanded_fingerprint": out_hash,
    }
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(m, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--manifest_out", default=None)
    ap.add_argument("--include_original", action="store_true")
    ap.add_argument("--seed", type=int, default=1337)

    ap.add_argument("--paraphrase_k", type=int, default=3, help="Paraphrase variants per base prompt")
    ap.add_argument("--numeric_k", type=int, default=4, help="Numeric variants per eligible base prompt")
    ap.add_argument("--format_k", type=int, default=0, help="Format/constraint variants per base prompt")

    ap.add_argument("--enable_paraphrase", action="store_true")
    ap.add_argument("--enable_numeric", action="store_true")
    ap.add_argument("--enable_format", action="store_true")
    args = ap.parse_args()

    base = read_jsonl(args.in_jsonl)
    if not base:
        raise SystemExit("Empty input JSONL")

    # Sanity check schema
    for r in base:
        for k in [PID_FIELD, PROMPT_FIELD, CONTEXT_FIELD, GT_FIELD, SCORING_FIELD]:
            if k not in r:
                raise SystemExit(f"Missing required key '{k}' in row {r.get(PID_FIELD)}")

    # Hard guardrails: fail loudly instead of silently capping.
    if args.enable_paraphrase and args.paraphrase_k > len(DEFAULT_PARAPHRASES):
        raise SystemExit(
            f"--paraphrase_k={args.paraphrase_k} but only {len(DEFAULT_PARAPHRASES)} paraphrase templates exist. "
            "Add templates to DEFAULT_PARAPHRASES or reduce paraphrase_k."
        )
    if args.enable_format and args.format_k > len(FORMAT_VARIANTS):
        raise SystemExit(
            f"--format_k={args.format_k} but only {len(FORMAT_VARIANTS)} format variants exist. "
            "Add templates to FORMAT_VARIANTS or reduce format_k."
        )

    out: List[Dict[str, Any]] = []

    if args.include_original:
        for r in base:
            rr = dict(r)
            rr["base_prompt_id"] = r[PID_FIELD]
            rr["expansion_type"] = "original"
            rr["variant_index"] = 0
            out.append(rr)

    paraphrases = DEFAULT_PARAPHRASES
    formats = FORMAT_VARIANTS
    rule = NumericRule()

    for r in base:
        # paraphrase expands everything (safe)
        if args.enable_paraphrase and args.paraphrase_k > 0:
            out.extend(paraphrase_expand(r, paraphrases, args.paraphrase_k))

        # format expands everything (safe)
        if args.enable_format and args.format_k > 0:
            out.extend(format_expand(r, formats, args.format_k))

        # numeric expands only when anchored via facts + GT match
        if args.enable_numeric and args.numeric_k > 0:
            out.extend(numeric_expand_exact(r, rule, seed=args.seed, max_variants=args.numeric_k))
            out.extend(numeric_expand_json_schema(r, rule, seed=args.seed, max_variants=args.numeric_k))

    # deterministic order
    out = sorted(out, key=stable_sort_key)

    # guardrails + manifest
    validation = validate_expansion(base, out)

    write_jsonl(args.out_jsonl, out)
    print(f"Wrote {len(out)} prompts -> {args.out_jsonl}")
    print("Counts:", validation["counts_by_type"])

    if args.manifest_out:
        cfg = {
            "include_original": args.include_original,
            "enable_paraphrase": args.enable_paraphrase,
            "enable_format": args.enable_format,
            "enable_numeric": args.enable_numeric,
            "paraphrase_k": args.paraphrase_k,
            "format_k": args.format_k,
            "numeric_k": args.numeric_k,
            "seed": args.seed,
        }
        write_manifest(args.manifest_out, base, out, cfg, validation)
        print(f"Wrote manifest -> {args.manifest_out}")


if __name__ == "__main__":
    main()