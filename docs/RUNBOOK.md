# RUNBOOK --- AI Evals v2 (End‑to‑End)

This runbook describes the complete lifecycle of AI Evals v2:

1.  Base scenario construction
2.  Expansion and freezing
3.  Behavioral runs
4.  Experiment families
5.  Context cliff sweeps
6.  Metrics extraction
7.  Enrichment
8.  Plot generation

This document reflects the finalized v2 architecture and aligns with
README, DATA_CONTRACT, and SCORING.

------------------------------------------------------------------------

# 0. Directory Overview

Core inputs: - `data/v2_scenarios.jsonl` -
`data/v2_scenarios_expanded.jsonl` - `data/v2_suite_50.jsonl` -
`data/v2_ctx_cliff_suite_20.jsonl`

Configs: - `configs/run_config.yaml` - `configs/experiments.yaml` -
`configs/experiments_ctx_cliff.yaml`

Outputs: - `results/v2/runs/behavioral/` -
`results/v2/runs/ctx_cliff/` - `results/v2/metrics/behavioral/` -
`results/v2/metrics/ctx_cliff/` - `artifacts/figures/`

------------------------------------------------------------------------

# 1. Base Scenario Development

Author base templates in:

`data/v2_scenarios.jsonl`

Each scenario defines: - multi-turn dialogue - tags - deterministic
checks

Do not directly edit frozen benchmark files.

------------------------------------------------------------------------

# 2. Expand Suite

Generate expanded variants:

``` bash
python3 scripts/expand_v2_suite.py
```

This produces:

`data/v2_scenarios_expanded.jsonl`

------------------------------------------------------------------------

# 3. Freeze Behavioral Benchmark

Create frozen suite:

``` bash
head -n 50 data/v2_scenarios_expanded.jsonl > data/v2_suite_50.jsonl
```

This file becomes immutable for benchmarking.

Do not modify once experiments begin.

------------------------------------------------------------------------

# 4. Behavioral Baseline Runs

Run all baseline profiles:

``` bash
python3 scripts/run_experiments.py   --experiments configs/experiments.yaml   --only behavioral_baseline
```

Output: - `results/v2/runs/behavioral/*.jsonl`

Each file corresponds to one (profile × config cell).

------------------------------------------------------------------------

# 5. Experiment Families

Run experiment families individually to isolate variables.

## Model size (fixed context)

``` bash
python3 scripts/run_experiments.py   --experiments configs/experiments.yaml   --only fixed_ctx_model_size
```

## Tradeoff (size vs context)

``` bash
python3 scripts/run_experiments.py   --experiments configs/experiments.yaml   --only tradeoff_ctx_vs_size
```

## MoE vs dense

``` bash
python3 scripts/run_experiments.py   --experiments configs/experiments.yaml   --only moe_fixed_ctx
```

All outputs stored in:

`results/v2/runs/behavioral/`

------------------------------------------------------------------------

# 6. Context Cliff Sweep

Run dedicated context sweep:

``` bash
python3 scripts/run_experiments.py   --experiments configs/experiments_ctx_cliff.yaml
```

Move results to dedicated folder:

``` bash
mv results/v2/runs/v2_experiments_ctx_cliff_*.jsonl    results/v2/runs/ctx_cliff/
```

------------------------------------------------------------------------

# 7. Behavioral Metrics Extraction

Generate behavioral scorecard:

``` bash
python3 scripts/v2/scorecard.py   --runs_dir results/v2/runs/behavioral   --scenarios data/v2_suite_50.jsonl
```

Outputs:

-   `behavioral_metrics.csv`
-   `model_scorecard.json`
-   `profile_scorecard.json`
-   `model_profile_scorecard.json`

------------------------------------------------------------------------

# 8. Context Memory Compliance Metrics (MCS)

Compute MCS for behavioral runs (optional diagnostic):

``` bash
python3 results/v2/metrics/memory_compliance.py   --runs_dir results/v2/runs/behavioral   --scenarios data/v2_suite_50.jsonl
```

Compute MCS for context cliff runs:

``` bash
python3 results/v2/metrics/memory_compliance.py   --runs_dir results/v2/runs/ctx_cliff   --scenarios data/v2_ctx_cliff_suite_20.jsonl
```

Outputs:

-   `metrics_run_mcs.csv`
-   `metrics_turn_mcs.csv`

------------------------------------------------------------------------

# 9. Enrichment

Join metadata to metric outputs:

``` bash
python3 scripts/enrich_metrics_with_run_metadata.py   --runs_glob "results/v2/runs/behavioral/*.jsonl"   --metrics_run results/v2/metrics/behavioral/metrics_run_mcs.csv   --metrics_turn results/v2/metrics/behavioral/metrics_turn_mcs.csv   --out_run results/v2/metrics/behavioral/metrics_run_mcs_enriched.csv   --enrich_turn
```

Repeat for ctx_cliff folder.

Enrichment adds:

-   profile
-   model
-   cell_id
-   family
-   num_ctx
-   temperature
-   run_file

Metrics are not modified.

------------------------------------------------------------------------

# 10. Plot Generation

Generate summary figures:

``` bash
python3 scripts/plots/plot_v2_results.py   --behavioral_run_csv results/v2/metrics/behavioral/metrics_run_mcs_enriched.csv   --ctx_cliff_run_csv results/v2/metrics/ctx_cliff/metrics_run_mcs_enriched.csv   --out_dir artifacts/figures
```

Typical plots:

-   MCS vs num_ctx (context cliff boundary)
-   drop_turn vs num_ctx
-   mcs_auc vs num_ctx
-   BRI comparison across models

------------------------------------------------------------------------

# 11. Sanity Checks

Check non-empty memory metrics:

``` bash
cat results/v2/metrics/ctx_cliff/metrics_run_mcs.csv
```

Verify:

-   mcs_n \> 0
-   drop_turn_mean populated for small contexts
-   mcs ≈ 1 for large contexts

------------------------------------------------------------------------

# 12. Clean Before Publishing

Remove raw run logs if publishing:

``` bash
rm results/v2/runs/**/*.jsonl
```

Keep:

-   frozen suites
-   metrics CSV
-   scorecards
-   plots

------------------------------------------------------------------------

# 13. Versioning Rule

If scenarios change:

1.  Expand again
2.  Freeze to new version
3.  Re-run all experiments
4.  Publish new metrics separately

Never overwrite frozen benchmark files.

------------------------------------------------------------------------

End of runbook.