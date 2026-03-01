#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import urllib.request
import urllib.error

import yaml  # pyyaml


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENARIOS = REPO_ROOT / "data" / "v2_scenarios.jsonl"
DEFAULT_OUTDIR = REPO_ROOT / "results" / "v2" / "runs"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _safe_mkdir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _slug(s: str) -> str:
    return "".join(ch if ch.isalnum() else "_" for ch in s)[:60]


def _make_run_id(run_name: str, profile: str, model: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    h = hashlib.sha1(f"{stamp}:{profile}:{model}".encode("utf-8")).hexdigest()[:10]
    return f"{run_name}_{stamp}_{_slug(profile)}_{_slug(model)}_{h}"


def _ollama_chat(
    host: str,
    model: str,
    messages: List[Dict[str, str]],
    options: Dict[str, Any],
    timeout_s: int = 180,
) -> Tuple[str, Dict[str, Any], int]:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": options or {},
    }
    body = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url=f"{host.rstrip('/')}/api/chat",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            raw_text = resp.read().decode("utf-8")
            raw = json.loads(raw_text)
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"Ollama HTTPError {e.code}: {err}") from e
    except Exception as e:
        raise RuntimeError(f"Ollama request failed: {e}") from e
    t1 = time.perf_counter()

    latency_ms = int((t1 - t0) * 1000)
    assistant_text = (raw.get("message") or {}).get("content") or ""
    return assistant_text, raw, latency_ms


def _build_messages(system: str, history: List[Dict[str, str]], user_text: str) -> List[Dict[str, str]]:
    msgs: List[Dict[str, str]] = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.extend(history)
    msgs.append({"role": "user", "content": user_text})
    return msgs


def _select_profile(cfg: Dict[str, Any], profile: str) -> Dict[str, Any]:
    if profile not in cfg:
        known = [k for k in cfg.keys() if k not in ("run", "input_format", "prompts")]
        raise SystemExit(f"Unknown --profile '{profile}'. Known profiles: {', '.join(sorted(known))}")
    p = cfg[profile] or {}
    if "host" not in p or "model" not in p:
        raise SystemExit(f"Profile '{profile}' must contain host and model.")
    p.setdefault("options", {})
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="configs/*.yaml")
    ap.add_argument("--profile", type=str, required=True, help="e.g. qwen_14b_t0, mixtral_moe_7b")
    ap.add_argument("--scenarios", type=str, default=str(DEFAULT_SCENARIOS))
    ap.add_argument("--outdir", type=str, default=str(DEFAULT_OUTDIR))
    ap.add_argument("--limit", type=int, default=None, help="Override run.limit for number of scenarios.")
    ap.add_argument("--shuffle", action="store_true", help="Shuffle scenarios (overrides run.shuffle).")
    ap.add_argument("--no-shuffle", action="store_true", help="Disable shuffle (overrides run.shuffle).")
    ap.add_argument("--seed", type=int, default=None, help="Override run.seed.")
    ap.add_argument("--timeout_s", type=int, default=180, help="HTTP timeout for Ollama calls.")
    args = ap.parse_args()

    cfg = _load_yaml(Path(args.config))
    run_cfg = cfg.get("run", {}) or {}

    profile_cfg = _select_profile(cfg, args.profile)
    host = str(profile_cfg["host"])
    model = str(profile_cfg["model"])
    options = dict(profile_cfg.get("options") or {})

    # scenario controls
    scenarios_path = Path(args.scenarios)
    scenarios = _read_jsonl(scenarios_path)

    shuffle = bool(run_cfg.get("shuffle", False))
    if args.shuffle:
        shuffle = True
    if args.no_shuffle:
        shuffle = False

    seed = int(run_cfg.get("seed", 123))
    if args.seed is not None:
        seed = int(args.seed)

    rng = random.Random(seed)
    if shuffle:
        rng.shuffle(scenarios)

    limit = run_cfg.get("limit", None)
    if args.limit is not None:
        limit = args.limit
    if limit:
        scenarios = scenarios[: int(limit)]

    run_name = str(run_cfg.get("run_name", "v2_run"))
    outdir = Path(args.outdir)
    _safe_mkdir(outdir)

    run_id = _make_run_id(run_name, args.profile, model)
    outpath = outdir / f"{run_id}.jsonl"

    print(f"[v2/run_sessions] run_id={run_id}")
    print(f"[v2/run_sessions] profile={args.profile} model={model}")
    print(f"[v2/run_sessions] scenarios={len(scenarios)} shuffle={shuffle} seed={seed}")
    print(f"[v2/run_sessions] writing -> {outpath}")

    total_turns = 0
    with outpath.open("w", encoding="utf-8") as f:
        for sidx, scenario in enumerate(scenarios):
            scenario_id = scenario.get("scenario_id", f"scenario_{sidx:04d}")
            system = scenario.get("system", "")
            turns = scenario.get("turns", [])

            if not isinstance(turns, list) or not turns:
                continue

            history: List[Dict[str, str]] = []
            for tidx, turn in enumerate(turns):
                if (turn or {}).get("role") != "user":
                    continue
                user_text = (turn or {}).get("content", "")

                messages = _build_messages(system=system, history=history, user_text=user_text)

                started_at = _utc_now_iso()
                assistant_text, raw, latency_ms = _ollama_chat(
                    host=host,
                    model=model,
                    messages=messages,
                    options=options,
                    timeout_s=args.timeout_s,
                )
                ended_at = _utc_now_iso()

                rec = {
                    "run_id": run_id,
                    "run_name": run_name,
                    "profile": args.profile,
                    "scenario_id": scenario_id,
                    "scenario_index": sidx,
                    "turn_index": tidx,
                    "started_at": started_at,
                    "ended_at": ended_at,
                    "latency_ms": latency_ms,
                    "host": host,
                    "model": model,
                    "options": options,
                    "system": system,
                    "user_text": user_text,
                    "messages": messages,
                    "assistant_text": assistant_text,
                    "ollama_raw": raw,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

                history.append({"role": "user", "content": user_text})
                history.append({"role": "assistant", "content": assistant_text})
                total_turns += 1

            print(f"[v2/run_sessions] {sidx+1}/{len(scenarios)} {scenario_id}")

    print(f"[v2/run_sessions] done. turns_logged={total_turns} file={outpath}")


if __name__ == "__main__":
    main()