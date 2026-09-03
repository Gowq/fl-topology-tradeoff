#!/usr/bin/env bash
#SBATCH --job-name=exp07_secondary
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --array=0-161%2
#SBATCH --no-requeue
#SBATCH --output=logs/exp07_secondary_%A_%a.out
#SBATCH --error=logs/exp07_secondary_%A_%a.err

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
mkdir -p "$ROOT/logs"
cd "$ROOT"

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null; then
    echo "CUDA preflight failed (driver) before Exp. 07 secondary block ${SLURM_ARRAY_TASK_ID}" >&2
    exit 42
fi
if ! python3 -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "CUDA preflight failed (torch) before Exp. 07 secondary block ${SLURM_ARRAY_TASK_ID}" >&2
    exit 42
fi

BLOCK_SIZE=6
CONFIG_START=$((1080 + SLURM_ARRAY_TASK_ID * BLOCK_SIZE))

srun --unbuffered bash scripts/run.sh exp07 \
    --config-start "$CONFIG_START" \
    --config-count "$BLOCK_SIZE" \
    --device cuda
