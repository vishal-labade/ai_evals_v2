from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Callable, List

TransformFn = Callable[[Dict[str, Any]], List[Dict[str, Any]]]

@dataclass(frozen=True)
class Transform:
    name: str
    fn: TransformFn

class TransformRegistry:
    def __init__(self) -> None:
        self._t: Dict[str, Transform] = {}

    def register(self, name: str, fn: TransformFn) -> None:
        if name in self._t:
            raise ValueError(f"Transform already registered: {name}")
        self._t[name] = Transform(name=name, fn=fn)

    def get(self, name: str) -> Transform:
        if name not in self._t:
            raise KeyError(f"Unknown transform: {name}. Available={list(self._t.keys())}")
        return self._t[name]

    def list(self) -> List[str]:
        return sorted(self._t.keys())