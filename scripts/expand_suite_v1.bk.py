#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List

import yaml

from ai_eval.expand.io import read_jsonl, write_jsonl
from ai_eval.expand.registry import TransformRegistry
from ai_eval.expand.transforms.paraphrase import make_paraphrase_transform
from ai_eval.expand.transforms.numeric import make_numeric_transform
from ai_eval.expand.schema import get_prompt_id


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_jsonl", required=True)
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--paraphrases_yaml", default="configs/expansions/paraphrases.yaml")
    ap.add_argument("--numeric_specs_yaml", default="configs/expansions/numeric_specs.yaml")
    ap.add_argument("--enable", nargs="+", default=["paraphrase", "numeric"],
                    help="Which transforms to run. Options: paraphrase numeric")
    ap.add_argument("--include_original", action="store_true",
                    help="Include original prompts in output (default: no)")
    ap.add_argument("--max_per_base", type=int, default=0,
                    help="If >0, cap expansions per base prompt (after all transforms)")
    args = ap.parse_args()

    base = read_jsonl(args.in_jsonl)

    par_cfg = load_yaml(args.paraphrases_yaml)
    paraphrases = list(par_cfg.get("variants", []))

    num_cfg = load_yaml(args.numeric_specs_yaml)
    numeric_specs = dict(num_cfg.get("specs", {}))

    reg = TransformRegistry()
    reg.register("paraphrase", make_paraphrase_transform(paraphrases))
    reg.register("numeric", make_numeric_transform(numeric_specs))

    enabled = args.enable

    out_rows: List[Dict[str, Any]] = []
    if args.include_original:
        for r in base:
            rr = dict(r)
            rr["base_prompt_id"] = get_prompt_id(r)
            rr["expansion_type"] = "original"
            rr["variant_index"] = 0
            out_rows.append(rr)

    # Expand each base prompt with each enabled transform
    for r in base:
        expansions: List[Dict[str, Any]] = []
        for name in enabled:
            t = reg.get(name)
            expansions.extend(t.fn(r))

        if args.max_per_base and args.max_per_base > 0:
            expansions = expansions[: args.max_per_base]

        out_rows.extend(expansions)

    # Minimal safety: unique prompt_id
    seen = set()
    for r in out_rows:
        pid = r.get("prompt_id") or r.get("id")
        if pid in seen:
            raise SystemExit(f"Duplicate prompt_id detected: {pid}")
        seen.add(pid)

    write_jsonl(args.out_jsonl, out_rows)
    print(f"Wrote {len(out_rows)} prompts -> {args.out_jsonl}")


if __name__ == "__main__":
    main()