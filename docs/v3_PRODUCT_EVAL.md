# AI Evals V3: Product-Level Evaluation

V3 is a *post-processing layer* on top of V2. 

## Inputs

1) `behavioral_metrics.csv` (from V2)
- scenario-level scores (persona, memory, recovery, contradiction, BRI, ...)

2) `runs.jsonl` (from V2)
- turn-level logs
- includes Ollama telemetry inside `ollama_raw`:
  - `prompt_eval_count`, `eval_count`
  - `prompt_eval_duration`, `eval_duration`
  - `total_duration`, `load_duration`

## Outputs

The V3 script produces:

- `metrics_turn_v3.csv`
  - Adds turn-level product telemetry (tokens, TPS, TTFT proxy, prefill/decode time split)

- `metrics_run_v3.csv`
  - Aggregates scenario-level product metrics:
    - avg latency, token totals, cost estimate
    - reliability per 1k tokens
    - reliability per second

- `product_leaderboard_v3.csv`
  - Aggregates by `(profile, model, temperature, num_ctx)`:
    - PRI (mean + median)
    - cost/latency/tokens
    - throughput
    - *prefill vs decode split*

- `product_scorecard_v3.json`
  - Compact top-10 leaderboard + weight assumptions

## Hidden metric: Prefill vs Decode decomposition

Most eval repos report "average latency" only.

V3 additionally reports:
- `ttft_ms` (proxy) ≈ prefill time from `prompt_eval_duration`
- `prompt_tps` (prefill tokens/sec)
- `decode_tps` (generation tokens/sec)
- `prefill_time_share` (context tax share of total time)

This makes your evaluation *look and feel like production inference profiling*.

## How to run

From repo root:

```bash
python3 analysis/v3_product_scorecard.py \
  --metrics artifacts/v2/behavioral_metrics.csv \
  --runs artifacts/v2/runs.jsonl \
  --outdir artifacts/v3 \
  --price-per-1k-tokens-usd 0
```

If you want a cost estimate, set `--price-per-1k-tokens-usd` to a constant.
(For local inference this is optional; relative efficiency is often enough.)

## PRI definition (default)

```
PRI =
0.35 * memory_score
+0.25 * persona_pss
+0.15 * recovery_score
+0.15 * (1 - contradiction_rate)
+0.10 * bri
```

All terms are in 0..1 in your pipeline (conservative defaults if missing).

## Suggested README snippet

Add:

- "V3: Product Metrics" section
- Link to `docs/V3_PRODUCT_EVAL.md`
- Show a tiny excerpt of `product_scorecard_v3.json` (top-3 only)