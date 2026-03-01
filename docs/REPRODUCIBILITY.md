# Reproducibility --- AI Evals v2

This document describes how to reproduce AI Evals v2 results in a
deterministic and controlled manner.

This version aligns with:

-   README (project intent + architecture)
-   RUNBOOK_v2.md (execution flow)
-   DATA_CONTRACT_v2.md (schema invariants)
-   SCORING_v2.md (metric definitions)

AI Evals v2 is fully local and deterministic. No external APIs or manual
labeling are used.

------------------------------------------------------------------------

# 1. Reproducibility Philosophy

V2 ensures reproducibility through:

1.  Frozen scenario files
2.  Deterministic scoring logic
3.  Logged model configuration
4.  Controlled experiment families
5.  Separation of behavioral and context tracks

All reported metrics are derived from logged JSONL run files using
rule-based scoring.

------------------------------------------------------------------------

# 2. Frozen Benchmark Policy

Two frozen datasets define the benchmark state:

-   `data/v2_suite_50.jsonl` (behavioral benchmark)
-   `data/v2_ctx_cliff_suite_20.jsonl` (context-boundary benchmark)

Rules:

-   These files must not be modified after freezing.
-   Any change requires:
    -   generating a new expanded suite
    -   freezing under a new filename (e.g., v2_suite_50_v2.jsonl)
    -   re-running all experiments

This guarantees longitudinal comparability.

------------------------------------------------------------------------

# 3. Determinism Controls

## 3.1 Scenario Expansion

Suite expansion is deterministic:

``` bash
python3 scripts/expand_v2_suite.py
```

-   Uses fixed transformation rules
-   No stochastic rewriting
-   No external model calls

If randomness is introduced in future expansion logic, a fixed seed must
be logged.

------------------------------------------------------------------------

## 3.2 Model Execution Controls

Each run logs:

-   model
-   profile
-   temperature
-   num_ctx
-   num_predict
-   host
-   run_id

For deterministic baseline runs:

-   temperature = 0.0
-   fixed num_ctx
-   fixed num_predict

Variation in these parameters is explicit and controlled via experiment
families.

------------------------------------------------------------------------

## 3.3 Scoring Determinism

All metrics are computed via rule-based logic:

-   persona checks → string/structure rules
-   memory checks → require_any / forbid_any rules
-   contradiction detection → deterministic comparison rules
-   calibration → numeric difference computation
-   MCS → strict boolean evaluation

Scoring does not: - call external APIs - call another LLM - depend on
human annotation

Given identical inputs, scoring outputs are identical.

------------------------------------------------------------------------

# 4. Required Environment Logging

For reproducible publication, record:

-   OS
-   CPU / GPU model
-   Python version
-   Ollama version
-   Exact model tags (e.g., qwen2.5:14b)
-   Git commit hash of this repo

Example:

``` bash
python --version
ollama --version
git rev-parse HEAD
```

These values should be stored in a publication note or experiment log.

------------------------------------------------------------------------

# 5. Run Reproduction Procedure

## 5.1 Expand and Freeze

``` bash
python3 scripts/expand_v2_suite.py
head -n 50 data/v2_scenarios_expanded.jsonl > data/v2_suite_50.jsonl
```

## 5.2 Run Behavioral Baseline

``` bash
python3 scripts/run_experiments.py   --experiments configs/experiments.yaml   --only behavioral_baseline
```

## 5.3 Run Context Cliff Sweep

``` bash
python3 scripts/run_experiments.py   --experiments configs/experiments_ctx_cliff.yaml
```

## 5.4 Extract Behavioral Metrics

``` bash
python3 scripts/v2/scorecard.py   --runs_dir results/v2/runs/behavioral   --scenarios data/v2_suite_50.jsonl
```

## 5.5 Compute Context Memory Compliance

``` bash
python3 results/v2/metrics/memory_compliance.py   --runs_dir results/v2/runs/ctx_cliff   --scenarios data/v2_ctx_cliff_suite_20.jsonl
```

## 5.6 Enrichment

``` bash
python3 scripts/enrich_metrics_with_run_metadata.py   --runs_glob "results/v2/runs/**/*.jsonl"   --metrics_run results/v2/metrics/...   --metrics_turn results/v2/metrics/...   --enrich_turn
```

## 5.7 Plotting

``` bash
python3 scripts/plots/plot_v2_results.py   --behavioral_run_csv ...   --ctx_cliff_run_csv ...   --out_dir artifacts/figures
```

Following these steps yields identical metrics given identical models
and configurations.

------------------------------------------------------------------------

# 6. What Can Vary

The following may vary across machines:

-   Latency
-   Wall-clock runtime
-   GPU utilization
-   Minor formatting artifacts (e.g., Markdown fences)

The following must not vary:

-   persona_pss
-   memory_score
-   recovery_score
-   contradiction_rate
-   bri
-   mcs
-   drop_turn_mean
-   mcs_auc

If these vary, investigate: - scenario mismatch - incorrect runs_dir -
modified suite - model version mismatch

------------------------------------------------------------------------

# 7. Context Cliff Stability

The context cliff experiment is particularly sensitive to:

-   num_ctx parameter
-   actual model context limit
-   model implementation (quantization or truncation behavior)

To reproduce the cliff boundary:

-   Ensure num_ctx override is active
-   Confirm model supports requested context size
-   Verify that filler length in ctx suite exceeds lower context sizes

------------------------------------------------------------------------

# 8. Publication Checklist

Before publishing results:

-   Confirm frozen scenario files unchanged
-   Confirm scoring scripts unchanged
-   Archive raw run JSONLs (optional)
-   Publish metrics CSV + plots
-   Record environment + commit hash

------------------------------------------------------------------------

# 9. Versioning Strategy

Future improvements must:

1.  Version scenario files
2.  Version scoring logic if changed
3.  Recompute all metrics
4.  Clearly label outputs with version identifier

Never silently change frozen benchmarks.

------------------------------------------------------------------------

End of reproducibility specification.