from __future__ import annotations
from typing import Any, Dict, List
from ..schema import get_prompt_field, get_prompt_id

def make_paraphrase_transform(paraphrases: List[str], tag: str = "paraphrase"):

    def apply(row: Dict[str, Any]) -> List[Dict[str, Any]]:
        prompt_key = get_prompt_field(row)
        base_pid = get_prompt_id(row)

        base_prompt = str(row[prompt_key])

        # Heuristic: if prompt already contains a strong "JSON only" constraint, we still create variants,
        # but we prepend the paraphrase to avoid destroying original semantics.
        out: List[Dict[str, Any]] = []
        for i, p in enumerate(paraphrases, start=1):
            new = dict(row)
            new_id = f"{base_pid}__{tag}__v{i:02d}"

            new["prompt_id"] = new_id
            new["base_prompt_id"] = base_pid
            new["expansion_type"] = tag
            new["variant_index"] = i

            # Deterministic rewrite: prepend constraint then original prompt
            new[prompt_key] = f"{p}\n\n{base_prompt}"
            out.append(new)

        return out

    return apply