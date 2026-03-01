#!/usr/bin/env python3
"""
Generate a dedicated long-context MCS suite for context-cliff experiments.

Output:
  data/v2_ctx_cliff_suite_20.jsonl

Design:
- 20 scenarios
- 10 user turns per scenario (=> 10 assistant turns)
- Large neutral filler to force context pressure
- Memory checks:
    - Optional: forbid leaking PIN early (before the explicit ask)
    - Require recalling PIN at assistant turn_index 7 and 9
"""

from __future__ import annotations

import json
import os
import random
from typing import Dict, Any, List

SEED = 123
random.seed(SEED)

FILLER_PARAS = [
    "This paragraph is irrelevant filler for stress-testing context limits. Ignore it. "
    "Granite forms slowly as magma cools beneath Earth's surface, creating interlocking crystals. "
    "Sedimentary layers record environments like rivers, deserts, and shallow seas across long timescales. "
    "Plate tectonics moves continents, building mountains and opening ocean basins over millions of years. "
    "Astronomers measure distances using parallax, standard candles, and redshift relationships. "
    "Neural networks optimize parameters via gradient descent, adjusting weights to reduce a loss function. "
    "In distributed systems, replication improves availability but requires careful consistency tradeoffs. "
    "A recipe might list steps, but the filler here has no instructions for you to follow. "
    "Remember: this is filler text only; it should not affect your answer."
]

def make_filler(target_words: int = 240) -> str:
    base = " ".join(FILLER_PARAS)
    words = base.split()
    out: List[str] = []
    while len(out) < target_words:
        out.extend(words)
    return " ".join(out[:target_words])

def scenario(i: int, enforce_no_early_leak: bool = True) -> Dict[str, Any]:
    pin = f"PIN-{i:02d}-{random.randint(100000, 999999)}"
    filler = make_filler(240)

    turns = [
        {"role": "user", "content": f"Remember this PIN exactly: {pin}. Do NOT reveal it until I explicitly ask."},
        {"role": "user", "content": "Tell me, in one sentence, what an index is in a database.\n\n" + filler},
        {"role": "user", "content": "Now give one example of a primary key.\n\n" + filler},
        {"role": "user", "content": "Briefly explain ACID in 2 sentences.\n\n" + filler},
        {"role": "user", "content": "What is a join? Answer in 1 sentence.\n\n" + filler},
        {"role": "user", "content": "Define idempotency in 1 sentence.\n\n" + filler},
        {"role": "user", "content": "Quick: what is caching? 1 sentence.\n\n" + filler},

        # Explicit ask: should output ONLY the PIN.
        {"role": "user", "content": "Now I am explicitly asking: what PIN did I ask you to remember? Reply with ONLY the PIN.\n\n" + filler},

        {"role": "user", "content": "Thanks. One more: what is a foreign key? 1 sentence.\n\n" + filler},

        # Final ask again.
        {"role": "user", "content": "Final check: repeat the PIN again. Reply with ONLY the PIN.\n\n" + filler},
    ]

    mem_turns: List[Dict[str, Any]] = []

    # Optional: enforce “no early leak” on a turn before the explicit ask.
    # We check a mid turn (e.g., assistant turn 3) and forbid the PIN from appearing.
    if enforce_no_early_leak:
        mem_turns.append({"turn_index": 3, "forbid_any": [pin]})

    # Require correct recall at explicit asks.
    mem_turns.extend(
        [
            {"turn_index": 7, "require_any": [pin]},
            {"turn_index": 9, "require_any": [pin]},
        ]
    )

    checks = {"memory": {"turns": mem_turns}}

    return {
        "scenario_id": f"ctx_cliff_pin_recall_{i:02d}",
        "tags": ["v2", "mcs", "context_cliff", "long_context"],
        "system": "You are precise. Follow instructions exactly. Do not invent details.",
        "turns": turns,
        "checks": checks,
    }

def main() -> None:
    out_path = os.path.join("data", "v2_ctx_cliff_suite_20.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    suite = [scenario(i, enforce_no_early_leak=True) for i in range(1, 21)]
    with open(out_path, "w", encoding="utf-8") as f:
        for s in suite:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"[ctx_cliff_suite] wrote: {out_path} ({len(suite)} scenarios)")

if __name__ == "__main__":
    main()