#!/usr/bin/env bash
# Resumable Exp. 07 worker for Pegasus. The tuning phase produces the shared
# hyperparameter file; portion workers wait for it before claiming their range.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RESULTS="$ROOT/experiments/exp07_mhealth_generalization/results"
HYPERPARAMETERS="$RESULTS/tuned_hyperparameters.json"
PHASE="${1:-}"

mkdir -p "$RESULTS/tuning_parts" "$ROOT/logs"

case "$PHASE" in
  tune)
    for job_index in $(seq 0 71); do
      bash "$ROOT/scripts/run.sh" exp07-tune \
        --job-index "$job_index" \
        --device cuda
    done
    bash "$ROOT/scripts/run.sh" exp07-tune --aggregate
    ;;
  portion)
    first="${2:?usage: $0 portion FIRST LAST}"
    last="${3:?usage: $0 portion FIRST LAST}"
    if (( first < 0 || last > 509 || first > last )); then
      echo "invalid Exp. 07 range: $first-$last" >&2
      exit 2
    fi
    while [[ ! -s "$HYPERPARAMETERS" ]]; do
      echo ">>> Waiting for Pegasus tuning: $HYPERPARAMETERS"
      sleep 30
    done
    for config_index in $(seq "$first" "$last"); do
      bash "$ROOT/scripts/run.sh" exp07 \
        --config-index "$config_index" \
        --device cuda \
        --hyperparameters-file "$HYPERPARAMETERS"
    done
    ;;
  *)
    echo "usage: $0 tune | portion FIRST LAST" >&2
    exit 2
    ;;
esac
