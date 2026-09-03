#!/usr/bin/env bash
#SBATCH --job-name=exp07_corrected
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --array=0-179%2
#SBATCH --no-requeue
#SBATCH --output=logs/exp07_primary_%A_%a.out
#SBATCH --error=logs/exp07_primary_%A_%a.err

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs"
cd "$ROOT"

# nvidia-smi alone is not enough: gpunode001 has served allocations where the
# driver answered but torch still raised "No CUDA GPUs are available".
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null; then
    echo "CUDA preflight failed (driver) before Exp. 07 primary block ${SLURM_ARRAY_TASK_ID}" >&2
    exit 42
fi
if ! python3 -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "CUDA preflight failed (torch) before Exp. 07 primary block ${SLURM_ARRAY_TASK_ID}" >&2
    exit 42
fi

BLOCK_SIZE=6
CONFIG_START=$((SLURM_ARRAY_TASK_ID * BLOCK_SIZE))

srun --unbuffered bash scripts/run.sh exp07 \
    --config-start "$CONFIG_START" \
    --config-count "$BLOCK_SIZE" \
    --device cuda
