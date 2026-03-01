#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p artifacts/metrics artifacts/figures artifacts/hashes
mkdir -p results/v2/metrics/behavioral results/v2/metrics/experiments results/v2/metrics/ctx_cliff

echo "[bootstrap] Hash suites + configs"
sha256sum data/v2_suite_50.jsonl data/v2_ctx_cliff_suite_20.jsonl > artifacts/suite_hashes.sha256
sha256sum configs/run_config.yaml configs/experiments.yaml configs/experiments_ctx_cliff.yaml > artifacts/config_hashes.sha256

echo "[bootstrap] 1) Compute MCS (behavioral suite)"
python3 results/v2/metrics/memory_compliance.py \
  --runs_dir results/v2/runs \
  --scenarios data/v2_suite_50.jsonl \
  --out_turn results/v2/metrics/behavioral/metrics_turn_mcs.csv \
  --out_run  results/v2/metrics/behavioral/metrics_run_mcs.csv

echo "[bootstrap] 2) Compute MCS (ctx-cliff suite)"
python3 results/v2/metrics/memory_compliance.py \
  --runs_dir results/v2/runs \
  --scenarios data/v2_ctx_cliff_suite_20.jsonl \
  --out_turn results/v2/metrics/ctx_cliff/metrics_turn_mcs.csv \
  --out_run  results/v2/metrics/ctx_cliff/metrics_run_mcs.csv

echo "[bootstrap] 3) Enrich run-level MCS metrics (behavioral suite runs)"
python3 scripts/enrich_metrics_with_run_metadata.py \
  --runs_glob "results/v2/runs/*.jsonl" \
  --metrics_run results/v2/metrics/behavioral/metrics_run_mcs.csv \
  --metrics_turn results/v2/metrics/behavioral/metrics_turn_mcs.csv \
  --out_run results/v2/metrics/behavioral/metrics_run_mcs_enriched.csv \
  --enrich_turn

echo "[bootstrap] 4) Enrich run-level MCS metrics (ctx-cliff suite runs)"
python3 scripts/enrich_metrics_with_run_metadata.py \
  --runs_glob "results/v2/runs/*.jsonl" \
  --metrics_run results/v2/metrics/ctx_cliff/metrics_run_mcs.csv \
  --metrics_turn results/v2/metrics/ctx_cliff/metrics_turn_mcs.csv \
  --out_run results/v2/metrics/ctx_cliff/metrics_run_mcs_enriched.csv \
  --enrich_turn

echo "[bootstrap] 5) Plots"
python3 scripts/plots/plot_v2_results.py \
  --behavioral_run_csv results/v2/metrics/behavioral/metrics_run_mcs_enriched.csv \
  --ctx_cliff_run_csv results/v2/metrics/ctx_cliff/metrics_run_mcs_enriched.csv \
  --out_dir artifacts/figures

echo "[bootstrap] 6) Curate"
cp -f results/v2/metrics/behavioral/metrics_run_mcs_enriched.csv artifacts/metrics/
cp -f results/v2/metrics/ctx_cliff/metrics_run_mcs_enriched.csv artifacts/metrics/
cp -f results/v2/metrics/ctx_cliff/metrics_turn_mcs.csv artifacts/metrics/

echo "[bootstrap] DONE"