from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

PROMPT_KEYS = ["prompt", "question", "instruction"]
CONTEXT_KEYS = ["context", "ctx", "passage"]
GT_KEYS = ["ground_truth", "expected", "answer", "reference"]

def _first_present(d: Dict[str, Any], keys: list[str]) -> Optional[str]:
    for k in keys:
        if k in d and d[k] is not None:
            return k
    return None

def get_prompt_field(row: Dict[str, Any]) -> str:
    k = _first_present(row, PROMPT_KEYS)
    if not k:
        raise KeyError(f"No prompt field found. Expected one of {PROMPT_KEYS}. Keys={list(row.keys())}")
    return k

def get_context_field(row: Dict[str, Any]) -> Optional[str]:
    return _first_present(row, CONTEXT_KEYS)

def get_gt_field(row: Dict[str, Any]) -> Optional[str]:
    return _first_present(row, GT_KEYS)

def get_prompt_id(row: Dict[str, Any]) -> str:
    if "prompt_id" in row and row["prompt_id"]:
        return str(row["prompt_id"])
    if "id" in row and row["id"]:
        return str(row["id"])
    raise KeyError("No prompt_id/id in row")