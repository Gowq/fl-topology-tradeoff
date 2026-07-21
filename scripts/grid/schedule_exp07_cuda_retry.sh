#!/bin/bash
# Release a broken GPU allocation and retry the same Exp. 07 batch later.
set -euo pipefail

if (( $# != 2 )); then
    echo "usage: $0 CONFIG_INDICES CURRENT_RETRY" >&2
    exit 2
fi

config_indices="$1"
current_retry="$2"
max_retries="${EXP07_CUDA_MAX_RETRIES:-12}"

if [[ ! "$config_indices" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "invalid Exp. 07 config indices: $config_indices" >&2
    exit 2
fi
if [[ ! "$current_retry" =~ ^[0-9]+$ || ! "$max_retries" =~ ^[1-9][0-9]*$ ]]; then
    echo "retry counters must be non-negative integers" >&2
    exit 2
fi
if (( current_retry >= max_retries )); then
    echo "[cuda-guard] giving up after ${current_retry} delayed retries" >&2
    exit 1
fi

next_retry=$((current_retry + 1))
job_id=$(sbatch \
    --parsable \
    --array=0 \
    --begin=now+5minutes \
    --dependency=singleton \
    --job-name=exp07_cuda_retry \
    --export="ALL,EXP07_CONFIG_INDICES=${config_indices},EXP07_CUDA_RETRY=${next_retry}" \
    scripts/grid/run_exp07_mhealth.sh)

echo "[cuda-guard] retry_job=${job_id} attempt=${next_retry}/${max_retries} configs=${config_indices}"
