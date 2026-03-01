import argparse
import hashlib
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import requests
import yaml


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_hash(obj: Any) -> str:
    s = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(s).hexdigest()[:12]


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def ollama_generate(host: str, model: str, prompt: str, options: Dict[str, Any]) -> Dict[str, Any]:
    url = host.rstrip("/") + "/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": options or {},
    }
    r = requests.post(url, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="Path to YAML config")
    ap.add_argument("--modelid", required=True, help="Path to YAML config")
    args = ap.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    prompts_path = cfg["prompts"]["path"]
    template = cfg["input_format"]["template"]

    run_cfg = cfg["run"]
    # oll_cfg = cfg["ollama"]
    oll_cfg = cfg[args.modelid]

    host = oll_cfg["host"]
    model = oll_cfg["model"]
    options = oll_cfg.get("options", {})

    limit = run_cfg.get("limit", None)
    shuffle = bool(run_cfg.get("shuffle", False))
    seed = int(run_cfg.get("seed", 0))
    run_name = run_cfg.get("run_name", "run")
    out_dir = run_cfg.get("output_dir", "results/runs")

    ensure_dir(out_dir)

    # Load prompts into memory (fine for 1k). If you later scale to 100k, stream instead.
    items = list(read_jsonl(prompts_path))
    if shuffle:
        random.Random(seed).shuffle(items)

    if isinstance(limit, int):
        items = items[:limit]

    cfg_fingerprint = stable_hash(cfg)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_id = f"{run_name}_{ts}_{args.modelid}_{model.replace(':','-')}_{cfg_fingerprint}"
    out_path = os.path.join(out_dir, f"{run_id}.jsonl")

    print(f"Run ID: {run_id}")
    print(f"Prompts: {len(items)}")
    print(f"Writing: {out_path}")

    with open(out_path, "w", encoding="utf-8") as out:
        for i, p in enumerate(items, 1):
            prompt_id = p["prompt_id"]
            context = p.get("context", "")
            instr = p.get("prompt", "")

            input_text = template.format(context=context, prompt=instr)

            t0 = time.perf_counter()
            try:
                resp = ollama_generate(host=host, model=model, prompt=input_text, options=options)
                latency_ms = int((time.perf_counter() - t0) * 1000)

                output_text = resp.get("response", "")
                usage = {
                    "prompt_tokens": resp.get("prompt_eval_count"),
                    "completion_tokens": resp.get("eval_count"),
                    "total_tokens": None
                }
                if usage["prompt_tokens"] is not None and usage["completion_tokens"] is not None:
                    usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]

                row = {
                    "run_id": run_id,
                    "model_name": model,
                    "params": options,
                    "prompt_id": prompt_id,
                    "input_text": input_text,
                    "output_text": output_text,
                    "latency_ms": latency_ms,
                    "usage": usage,
                    "ts_utc": utc_now_iso(),
                }
            except Exception as e:
                latency_ms = int((time.perf_counter() - t0) * 1000)
                row = {
                    "run_id": run_id,
                    "model_name": model,
                    "params": options,
                    "prompt_id": prompt_id,
                    "input_text": input_text,
                    "output_text": "",
                    "latency_ms": latency_ms,
                    "usage": {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None},
                    "ts_utc": utc_now_iso(),
                    "error": str(e),
                }

            out.write(json.dumps(row, ensure_ascii=False) + "\n")

            if i % 10 == 0 or i == len(items):
                print(f"[{i}/{len(items)}] last_prompt={prompt_id} latency_ms={row['latency_ms']}")

    print("Done.")


if __name__ == "__main__":
    main()