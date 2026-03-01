#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Run behavioral OR MCS suite across profiles, then score.
#
# Usage:
#   ./scripts/run_behavioral_and_score.sh
#   ./scripts/run_behavioral_and_score.sh --lane behavioral
#   ./scripts/run_behavioral_and_score.sh --lane mcs
#   ./scripts/run_behavioral_and_score.sh --profiles qwen_3b_t0,phi_mini_t0 --lane mcs
#   ./scripts/run_behavioral_and_score.sh --limit 5 --lane mcs
#   ./scripts/run_behavioral_and_score.sh --clean --lane behavioral
# ============================================================

CONFIG="configs/run_config.yaml"
LANE="behavioral"  # behavioral | mcs
SCENARIOS_BEHAVIORAL="data/v2_suite_50.jsonl"
SCENARIOS_MCS="data/v2_mcs_suite_20.jsonl"
OUTDIR="results/v2/runs"
PROFILES=""
LIMIT=""
CLEAN="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --config) CONFIG="$2"; shift 2 ;;
    --lane) LANE="$2"; shift 2 ;;
    --scenarios-behavioral) SCENARIOS_BEHAVIORAL="$2"; shift 2 ;;
    --scenarios-mcs) SCENARIOS_MCS="$2"; shift 2 ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    --profiles) PROFILES="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --clean) CLEAN="true"; shift 1 ;;
    *) echo "Unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ "$LANE" != "behavioral" && "$LANE" != "mcs" ]]; then
  echo "Invalid --lane '$LANE'. Use: behavioral | mcs" >&2
  exit 2
fi

if [[ "$LANE" == "behavioral" ]]; then
  SCENARIOS="$SCENARIOS_BEHAVIORAL"
else
  SCENARIOS="$SCENARIOS_MCS"
fi

echo "[wrapper] config=$CONFIG"
echo "[wrapper] lane=$LANE"
echo "[wrapper] scenarios=$SCENARIOS"
echo "[wrapper] outdir=$OUTDIR"
echo "[wrapper] profiles=${PROFILES:-<ALL>}"
echo "[wrapper] limit=${LIMIT:-<none>}"
echo "[wrapper] clean=$CLEAN"

if [[ "$CLEAN" == "true" ]]; then
  echo "[wrapper] cleaning prior v2 run logs in $OUTDIR"
  mkdir -p "$OUTDIR"
  rm -f "$OUTDIR"/*.jsonl
fi

RUN_CMD=(python3 scripts/run_sessions.py --config "$CONFIG" --scenarios "$SCENARIOS" --outdir "$OUTDIR")

if [[ -n "$PROFILES" ]]; then
  RUN_CMD+=(--profiles "$PROFILES")
fi

if [[ -n "$LIMIT" ]]; then
  RUN_CMD+=(--limit "$LIMIT")
fi

echo "[wrapper] running lane=$LANE ..."
"${RUN_CMD[@]}"

echo "[wrapper] scoring lane=$LANE ..."
python3 results/v2/scorecard.py --scenarios "$SCENARIOS"

echo "[wrapper] done."