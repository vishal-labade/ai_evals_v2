import json
import yaml
from copy import deepcopy
from pathlib import Path

BASE_PATH = Path("data/v2_scenarios_base_20.jsonl")
FAMILY_PATH = Path("data/v2_families.yaml")
OUT_PATH = Path("data/v2_scenarios_expanded.jsonl")


def _format_with_placeholders(obj, mapping):
    """Recursively replace {PLACEHOLDER} tokens inside strings across dict/list trees."""
    if isinstance(obj, str):
        out = obj
        for k, v in mapping.items():
            out = out.replace("{" + k + "}", str(v))
        return out
    if isinstance(obj, list):
        return [_format_with_placeholders(x, mapping) for x in obj]
    if isinstance(obj, dict):
        return {k: _format_with_placeholders(v, mapping) for k, v in obj.items()}
    return obj


def _comma_number(n: int) -> str:
    return f"{n:,}"


base = [json.loads(l) for l in BASE_PATH.read_text(encoding="utf-8").splitlines() if l.strip()]
families = yaml.safe_load(FAMILY_PATH.read_text(encoding="utf-8"))

expanded = []
expanded.extend(deepcopy(base))

# Expand persona variants
for s in base:
    if "persona" in s.get("tags", []):
        for name, variant in (families.get("persona_variants") or {}).items():
            new = deepcopy(s)
            new["scenario_id"] = f"{s['scenario_id']}_{name}"
            new["system"] = variant.get("system", s.get("system"))
            expanded.append(new)

# Expand memory variants (requires checks.memory placeholders in base scenarios)
mem_cfg = families.get("memory_variants") or {}

for s in base:
    if "memory" not in s.get("tags", []):
        continue

    sid = s.get("scenario_id", "")

    if sid == "memory_numeric_recall_01":
        for n in (mem_cfg.get("numbers") or []):
            new = deepcopy(s)
            new["scenario_id"] = f"{sid}_{n}"
            mapping = {"NUM": int(n), "NUM_COMMA": _comma_number(int(n))}
            new = _format_with_placeholders(new, mapping)
            expanded.append(new)

    elif sid == "memory_name_recall_01":
        for name in (mem_cfg.get("names") or []):
            new = deepcopy(s)
            new["scenario_id"] = f"{sid}_{str(name).lower()}"
            mapping = {"NAME": str(name)}
            new = _format_with_placeholders(new, mapping)
            expanded.append(new)

    elif sid == "memory_structured_recall_01":
        for plan in (mem_cfg.get("plans") or []):
            new = deepcopy(s)
            new["scenario_id"] = f"{sid}_{str(plan).lower()}"
            mapping = {"PLAN": str(plan)}
            new = _format_with_placeholders(new, mapping)
            expanded.append(new)

OUT_PATH.write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in expanded) + "\n", encoding="utf-8")
print(f"Wrote {len(expanded)} scenarios to {OUT_PATH}")