from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import ast
import math
from ..schema import get_prompt_id, get_prompt_field, get_context_field, get_gt_field

# -------- Safe expression evaluation (only arithmetic over vars) --------

_ALLOWED_NODES = (
    ast.Expression, ast.BinOp, ast.UnaryOp,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.USub, ast.UAdd,
    ast.Num, ast.Constant,
    ast.Name,
    ast.Call
)

_ALLOWED_FUNCS = {
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "min": min,
    "max": max,
}

def _safe_eval(expr: str, vars: Dict[str, Any]) -> Any:
    tree = ast.parse(expr, mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _ALLOWED_NODES):
            raise ValueError(f"Unsafe expression node: {type(node).__name__}")
        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name) or node.func.id not in _ALLOWED_FUNCS:
                raise ValueError("Only whitelisted functions allowed")
    code = compile(tree, "<expr>", "eval")
    return eval(code, {"__builtins__": {} , **_ALLOWED_FUNCS}, dict(vars))

def _render_template(tpl: str, vars: Dict[str, Any]) -> str:
    # supports {{var}} and {{expr}}
    out = ""
    i = 0
    while i < len(tpl):
        j = tpl.find("{{", i)
        if j < 0:
            out += tpl[i:]
            break
        out += tpl[i:j]
        k = tpl.find("}}", j)
        if k < 0:
            raise ValueError("Unclosed {{ in template")
        expr = tpl[j+2:k].strip()
        val = _safe_eval(expr, vars) if any(ch in expr for ch in "+-*/()") else vars.get(expr)
        if val is None and expr not in vars:
            # allow expr evaluation even without ops
            val = _safe_eval(expr, vars)
        out += str(val)
        i = k + 2
    return out

def _apply_ops(vars: Dict[str, Any], ops: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    v = dict(vars)
    for key, spec in ops.items():
        if key not in v:
            raise KeyError(f"Numeric spec refers to unknown var '{key}'")
        x = float(v[key])
        if "mul" in spec:
            x *= float(spec["mul"])
        if "add" in spec:
            x += float(spec["add"])
        if "round" in spec:
            x = round(x, int(spec["round"]))
        v[key] = int(x) if float(v[key]).is_integer() and float(x).is_integer() else x
    return v

def make_numeric_transform(numeric_specs: Dict[str, Any], tag: str = "numeric"):

    def apply(row: Dict[str, Any]) -> List[Dict[str, Any]]:
        base_pid = get_prompt_id(row)
        if base_pid not in numeric_specs:
            return []

        spec = numeric_specs[base_pid]
        vars0: Dict[str, Any] = dict(spec.get("variables", {}))
        templates = spec.get("templates", {})
        ctx_tpl = templates.get("context")
        prm_tpl = templates.get("prompt")
        gt_tpl = templates.get("ground_truth")

        if not (ctx_tpl and prm_tpl and gt_tpl):
            raise ValueError(f"Numeric spec for {base_pid} missing templates.context/prompt/ground_truth")

        prompt_key = get_prompt_field(row)
        ctx_key = get_context_field(row)
        gt_key = get_gt_field(row)

        if ctx_key is None:
            # If your suite doesn't use context, you can still render prompt+gt
            ctx_key = "context"
        if gt_key is None:
            gt_key = "ground_truth"

        out: List[Dict[str, Any]] = []
        variants = spec.get("variants", [])
        for i, vdef in enumerate(variants, start=1):
            name = vdef.get("name", f"v{i:02d}")
            ops = vdef.get("ops", {})
            vvars = _apply_ops(vars0, ops)

            new = dict(row)
            new_id = f"{base_pid}__{tag}__{name}"

            new["prompt_id"] = new_id
            new["base_prompt_id"] = base_pid
            new["expansion_type"] = tag
            new["variant_index"] = i
            new["variant_name"] = name
            new["expansion_vars"] = vvars

            new[ctx_key] = _render_template(ctx_tpl, vvars)
            new[prompt_key] = _render_template(prm_tpl, vvars)
            new[gt_key] = _render_template(gt_tpl, vvars)

            out.append(new)

        return out

    return apply