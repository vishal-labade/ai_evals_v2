#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p artifacts/metrics artifacts/figures artifacts/hashes
mkdir -p results/v2/metrics

echo "[v2] Hash suites + configs"
sha256sum data/v2_suite_50.jsonl data/v2_ctx_cliff_suite_20.jsonl > artifacts/suite_hashes.sha256
sha256sum configs/run_config.yaml configs/experiments.yaml configs/experiments_ctx_cliff.yaml > artifacts/config_hashes.sha256

echo "[v2] 1) Run experiments on frozen behavioral suite (data/v2_suite_50.jsonl)"
# Uses configs/experiments.yaml :contentReference[oaicite:2]{index=2}
python3 scripts/run_experiments.py --experiments configs/experiments.yaml --only fixed_ctx_model_size
python3 scripts/run_experiments.py --experiments configs/experiments.yaml --only tradeoff_ctx_vs_size
python3 scripts/run_experiments.py --experiments configs/experiments.yaml --only moe_fixed_ctx
# Optional: keep if you want ctx sweep on the behavioral suite too
python3 scripts/run_experiments.py --experiments configs/experiments.yaml --only context_cliff_qwen14b

echo "[v2] 2) Run dedicated ctx-cliff sweep (data/v2_ctx_cliff_suite_20.jsonl)"
# Uses configs/experiments_ctx_cliff.yaml :contentReference[oaicite:3]{index=3}
python3 scripts/run_experiments.py --experiments configs/experiments_ctx_cliff.yaml --only context_cliff_qwen14b

echo "[v2] 3) Compute MCS metrics for behavioral suite (v2_suite_50)"
python3 results/v2/metrics/memory_compliance.py \
  --runs_dir results/v2/runs \
  --scenarios data/v2_suite_50.jsonl \
  --out_turn results/v2/metrics/metrics_turn_behavioral.csv \
  --out_run  results/v2/metrics/metrics_run_behavioral.csv

echo "[v2] 4) Compute MCS metrics for ctx-cliff suite (v2_ctx_cliff_suite_20)"
python3 results/v2/metrics/memory_compliance.py \
  --runs_dir results/v2/runs \
  --scenarios data/v2_ctx_cliff_suite_20.jsonl \
  --out_turn results/v2/metrics/metrics_turn_ctx_cliff.csv \
  --out_run  results/v2/metrics/metrics_run_ctx_cliff.csv

echo "[v2] 5) Enrich run-level metrics with metadata from run JSONLs"
python3 scripts/enrich_metrics_with_run_metadata.py \
  --runs_glob "results/v2/runs/*.jsonl" \
  --metrics_run results/v2/metrics/metrics_run_behavioral.csv \
  --metrics_turn results/v2/metrics/metrics_turn_behavioral.csv \
  --out_run results/v2/metrics/metrics_run_behavioral_enriched.csv \
  --enrich_turn

python3 scripts/enrich_metrics_with_run_metadata.py \
  --runs_glob "results/v2/runs/*.jsonl" \
  --metrics_run results/v2/metrics/metrics_run_ctx_cliff.csv \
  --metrics_turn results/v2/metrics/metrics_turn_ctx_cliff.csv \
  --out_run results/v2/metrics/metrics_run_ctx_cliff_enriched.csv \
  --enrich_turn

echo "[v2] 6) Curate outputs into artifacts/"
cp -f results/v2/metrics/metrics_run_behavioral_enriched.csv artifacts/metrics/
cp -f results/v2/metrics/metrics_run_ctx_cliff_enriched.csv artifacts/metrics/
cp -f results/v2/metrics/metrics_turn_ctx_cliff.csv artifacts/metrics/

echo "[v2] 7) Plots (canonical)"
python3 scripts/plots/plot_v2_results.py \
  --behavioral_run_csv results/v2/metrics/metrics_run_behavioral_enriched.csv \
  --ctx_cliff_run_csv results/v2/metrics/metrics_run_ctx_cliff_enriched.csv \
  --out_dir artifacts/figures

echo "[v2] DONE"
echo "Curated outputs:"
echo "  artifacts/metrics/metrics_run_behavioral_enriched.csv"
echo "  artifacts/metrics/metrics_run_ctx_cliff_enriched.csv"
echo "  artifacts/metrics/metrics_turn_ctx_cliff.csv"
echo "  artifacts/figures/*.png"