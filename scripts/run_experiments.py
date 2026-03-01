#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError

import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ts_compact_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _slug(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    return s.strip("_")[:80]


def _sha8(obj: Any) -> str:
    b = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(b).hexdigest()[:8]


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _post_json(url: str, payload: Dict[str, Any], timeout_s: int = 300) -> Dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except HTTPError as e:
        body = e.read().decode("utf-8", errors="replace") if hasattr(e, "read") else ""
        raise RuntimeError(f"HTTPError {e.code} calling {url}: {body}") from e
    except URLError as e:
        raise RuntimeError(f"URLError calling {url}: {e}") from e


@dataclass
class Profile:
    name: str
    host: str
    model: str
    options: Dict[str, Any]


def _load_profiles(path: Path) -> Dict[str, Profile]:
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    profiles: Dict[str, Profile] = {}

    # ignore cfg["run"] and cfg["input_format"] and cfg["prompts"]
    for key, val in cfg.items():
        if key in ("run", "input_format", "prompts"):
            continue
        if not isinstance(val, dict):
            continue
        host = val.get("host")
        model = val.get("model")
        options = val.get("options") or {}
        if host and model:
            profiles[key] = Profile(name=key, host=str(host), model=str(model), options=dict(options))
    return profiles


def _merge_options(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base or {})
    for k, v in (override or {}).items():
        out[k] = v
    return out


def _merge_profile(defaults: Dict[str, Any], prof: Profile, overrides: Dict[str, Any]) -> Tuple[str, str, Dict[str, Any]]:
    host = str(overrides.get("host") or prof.host or defaults.get("host"))
    model = str(overrides.get("model") or prof.model)

    base_opts = dict(defaults.get("options") or {})
    prof_opts = dict(prof.options or {})
    ov_opts = dict((overrides.get("options") or {}))

    # precedence: defaults < profile < overrides
    opts = _merge_options(base_opts, prof_opts)
    opts = _merge_options(opts, ov_opts)
    return host, model, opts


def run_cell(
    *,
    run_name: str,
    experiment_id: str,
    cell_id: str,
    profile_name: str,
    host: str,
    model: str,
    options: Dict[str, Any],
    scenarios: List[Dict[str, Any]],
    out_dir: Path,
    shuffle: bool,
    seed: int,
    limit_scenarios: Optional[int],
    limit_turns_per_scenario: Optional[int],
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    scen_list = list(scenarios)

    if shuffle:
        rng.shuffle(scen_list)

    if limit_scenarios is not None:
        scen_list = scen_list[: int(limit_scenarios)]

    # create a stable run_id prefix; each turn gets same run_id
    run_meta = {
        "run_name": run_name,
        "experiment_id": experiment_id,
        "cell_id": cell_id,
        "profile": profile_name,
        "host": host,
        "model": model,
        "options": options,
        "suite_n": len(scen_list),
    }
    run_id = f"{run_name}_{_ts_compact_utc()}_{_slug(profile_name)}_{_slug(model)}_{_sha8(run_meta)}"

    out_path = out_dir / f"{run_id}.jsonl"

    url = host.rstrip("/") + "/api/chat"

    with out_path.open("w", encoding="utf-8") as f:
        for si, scenario in enumerate(scen_list):
            scenario_id = str(scenario.get("scenario_id"))
            system = str(scenario.get("system") or "")
            turns = scenario.get("turns") or []

            if not isinstance(turns, list) or not system or not scenario_id:
                continue

            # Build conversation incrementally; user provides one message per turn in scenario["turns"]
            # We log one JSON row per assistant turn (aligned with your existing schema)
            messages: List[Dict[str, str]] = [{"role": "system", "content": system}]

            max_turns = len(turns)
            if limit_turns_per_scenario is not None:
                max_turns = min(max_turns, int(limit_turns_per_scenario))

            for ti in range(max_turns):
                user_msg = turns[ti]
                user_text = str(user_msg.get("content") or "")
                messages.append({"role": "user", "content": user_text})

                started_at = _now_utc_iso()
                t0 = time.time()

                payload = {
                    "model": model,
                    "messages": messages,
                    "options": options,
                    "stream": False,
                }

                resp = _post_json(url, payload)
                assistant_text = ((resp.get("message") or {}).get("content")) if isinstance(resp, dict) else ""
                assistant_text = "" if assistant_text is None else str(assistant_text)

                ended_at = _now_utc_iso()
                latency_ms = int(round((time.time() - t0) * 1000))

                row = {
                    "run_id": run_id,
                    "run_name": run_name,
                    "experiment_id": experiment_id,
                    "cell_id": cell_id,
                    "profile": profile_name,
                    "scenario_id": scenario_id,
                    "scenario_index": si,
                    "turn_index": ti,
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
                    "ollama_raw": resp,
                }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                f.flush()

                # append assistant to messages for next turn
                messages.append({"role": "assistant", "content": assistant_text})

    print(f"[run_experiments] wrote {out_path}")
    return out_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiments", type=str, default="configs/experiments.yaml")
    ap.add_argument("--only", type=str, default="", help="Comma-separated experiment keys to run (optional)")
    ap.add_argument("--only_cells", type=str, default="", help="Comma-separated cell_ids to run (optional)")
    args = ap.parse_args()

    exp_cfg_path = (REPO_ROOT / args.experiments).resolve()
    cfg = yaml.safe_load(exp_cfg_path.read_text(encoding="utf-8"))

    run_cfg = cfg.get("run") or {}
    profiles_cfg_path = (REPO_ROOT / str(cfg.get("profiles_config", "configs/run_config.yaml"))).resolve()

    run_name = str(run_cfg.get("run_name", "v2_experiments"))
    scenarios_path = (REPO_ROOT / str(run_cfg.get("scenarios_path", "data/v2_suite_50.jsonl"))).resolve()
    out_dir = (REPO_ROOT / str(run_cfg.get("output_dir", "results/v2/runs"))).resolve()
    shuffle = bool(run_cfg.get("shuffle", True))
    seed = int(run_cfg.get("seed", 123))
    limit_scenarios = run_cfg.get("limit_scenarios", None)
    limit_turns = run_cfg.get("limit_turns_per_scenario", None)

    if not scenarios_path.exists():
        raise SystemExit(f"Missing suite file: {scenarios_path}. Create it by freezing your 50 scenarios into data/v2_suite_50.jsonl")

    scenarios = _read_jsonl(scenarios_path)
    profiles = _load_profiles(profiles_cfg_path)

    defaults = cfg.get("defaults") or {}
    experiments = cfg.get("experiments") or {}

    only_set = set([x.strip() for x in args.only.split(",") if x.strip()])
    only_cells = set([x.strip() for x in args.only_cells.split(",") if x.strip()])

    experiment_id = f"{run_name}_{_ts_compact_utc()}_{_sha8({'run': run_cfg, 'defaults': defaults, 'experiments': list(experiments.keys())})}"

    for exp_key, exp_def in experiments.items():
        if only_set and exp_key not in only_set:
            continue

        cells = exp_def.get("cells") or []
        for cell in cells:
            cell_id = str(cell.get("cell_id"))
            if only_cells and cell_id not in only_cells:
                continue

            profile_name = str(cell.get("profile"))
            if profile_name not in profiles:
                raise SystemExit(f"Profile '{profile_name}' not found in {profiles_cfg_path}")

            prof = profiles[profile_name]
            overrides = cell.get("overrides") or {}

            host, model, options = _merge_profile(defaults, prof, overrides)

            run_cell(
                run_name=run_name,
                experiment_id=experiment_id,
                cell_id=cell_id,
                profile_name=profile_name,
                host=host,
                model=model,
                options=options,
                scenarios=scenarios,
                out_dir=out_dir,
                shuffle=shuffle,
                seed=seed,
                limit_scenarios=limit_scenarios,
                limit_turns_per_scenario=limit_turns,
            )

    print(f"[run_experiments] done. experiment_id={experiment_id}")


if __name__ == "__main__":
    main()