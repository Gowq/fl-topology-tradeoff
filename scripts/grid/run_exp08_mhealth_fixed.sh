#!/bin/bash
#SBATCH --job-name=exp08_mhealth
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --array=0-5%4
#SBATCH --requeue

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required for Grid jobs}"
RESULTS="$ROOT/experiments/exp08_mhealth_fixed_rounds/results"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fl_env_grid
mkdir -p logs "$RESULTS"
EXP08_BATCH_SIZE=4
EXP08_CONFIG_COUNT=24
if [[ -n "${EXP08_CONFIG_INDICES:-}" ]]; then
    config_indices="$EXP08_CONFIG_INDICES"
else
    start=$((SLURM_ARRAY_TASK_ID * EXP08_BATCH_SIZE))
    stop=$((start + EXP08_BATCH_SIZE - 1))
    (( stop >= EXP08_CONFIG_COUNT )) && stop=$((EXP08_CONFIG_COUNT - 1))
    config_indices=""
    for ((index = start; index <= stop; index++)); do
        config_indices+="${config_indices:+,}$index"
    done
fi

if ! srun --unbuffered python3 -c \
    "import torch; assert torch.cuda.is_available(); torch.zeros(1, device='cuda'); torch.cuda.synchronize()"; then
    retries="${EXP08_CUDA_RETRY:-0}"
    if (( retries >= 12 )); then
        echo "[cuda-guard] retry limit reached for $config_indices" >&2
        exit 1
    fi
    retry_key="${config_indices//,/_}"
    echo "[cuda-guard] no usable GPU; scheduling batch $config_indices retry $((retries + 1))" >&2
    sbatch --array=0 --job-name="exp08_r_$retry_key" \
        --begin=now+5minutes --dependency=singleton \
        --export="ALL,EXP08_CONFIG_INDICES=$config_indices,EXP08_CUDA_RETRY=$((retries + 1))" \
        scripts/grid/run_exp08_mhealth_fixed.sh
    exit 0
fi

task_tmp=$(mktemp -d "/tmp/exp08_${USER}_${SLURM_JOB_ID}.XXXXXX")
trap 'rm -rf -- "$task_tmp"' EXIT
cp -a "$ROOT/data/MHEALTHDATASET" "$task_tmp/"
srun --unbuffered bash scripts/run.sh exp08 \
    --config-indices "$config_indices" --device cuda \
    --data-root "$task_tmp/MHEALTHDATASET" --output-dir "$RESULTS"
