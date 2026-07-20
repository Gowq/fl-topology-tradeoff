#!/bin/bash
#SBATCH --job-name=exp07_mhealth
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=1-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --array=0-509
#SBATCH --requeue

set -euo pipefail

ROOT="${SLURM_SUBMIT_DIR:?SLURM_SUBMIT_DIR is required for Grid jobs}"
HYPERPARAMETERS="$ROOT/experiments/exp07_mhealth_generalization/results/tuned_hyperparameters.json"
RESULTS="$ROOT/experiments/exp07_mhealth_generalization/results"
cd "$ROOT"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fl_env_grid

mkdir -p logs "$RESULTS"

if [[ -n "${EXP07_BATCH_MANIFEST:-}" ]]; then
    config_indices=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$EXP07_BATCH_MANIFEST")
    if [[ -z "$config_indices" ]]; then
        echo "empty Exp. 07 batch for array task $SLURM_ARRAY_TASK_ID" >&2
        exit 2
    fi
else
    config_indices="$SLURM_ARRAY_TASK_ID"
fi

# The Grid occasionally hands out a gres:gpu:1 slot whose GPU is not usable
# (torch sees zero devices and dies in seconds, which recycles the bad slot
# and burns through the whole array — see jobs 4785922-4785926, 2026-07-19).
# Requeue immediately instead of holding an unusable shared GPU allocation.
if ! srun --unbuffered python3 -c \
    "import torch; assert torch.cuda.is_available(); torch.zeros(1, device='cuda'); torch.cuda.synchronize()"; then
    restarts="${SLURM_RESTART_COUNT:-0}"
    echo "[cuda-guard] no usable GPU on $(hostname) (restart ${restarts})" >&2
    echo "[cuda-guard] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}" >&2
    nvidia-smi -L >&2 || true
    if [ "$restarts" -ge 2 ]; then
        echo "[cuda-guard] giving up after ${restarts} requeues" >&2
        exit 1
    fi
    scontrol requeue "$SLURM_JOB_ID"
    exit 0
fi

task_tmp=$(mktemp -d "/tmp/exp07_${USER}_${SLURM_JOB_ID}.XXXXXX")
cleanup() {
    rm -rf -- "$task_tmp"
}
trap cleanup EXIT
cp -a "$ROOT/data/MHEALTHDATASET" "$task_tmp/"

srun --unbuffered bash scripts/run.sh exp07 \
    --config-indices "$config_indices" \
    --device cuda \
    --data-root "$task_tmp/MHEALTHDATASET" \
    --output-dir "$RESULTS" \
    --hyperparameters-file "$HYPERPARAMETERS"
