#!/usr/bin/env bash
#SBATCH --job-name=exp08_tt_artifacts
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --array=0-9%2
#SBATCH --requeue
#SBATCH --output=logs/exp08_timetrojan_artifact_%A_%a.out
#SBATCH --error=logs/exp08_timetrojan_artifact_%A_%a.err

# Stage 1 of Exp. 08: generate the 10 frozen TimeTrojan artifacts (2 datasets x
# 5 primary seeds). The federated matrix (stage 2) depends on these via
# --dependency=afterok, so every HFL/VFL cell reuses identical poisoned windows.
# One artifact per task; %2 keeps to two of the four L40S on the single GPU node.

set -euo pipefail

# SLURM runs a spooled copy of this script, so BASH_SOURCE does not point at the
# repo. Rely on the submission directory (the repo root, where sbatch was run).
cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p logs

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fl_env_grid

if [ ! -e "./data" ]; then
    if [ -d "../data" ]; then ln -s ../data ./data; fi
fi

# nvidia-smi alone is not enough: gpunode001 has served allocations where the
# driver answered but torch still raised "No CUDA GPUs are available".
if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null; then
    echo "CUDA preflight failed (driver) before Exp. 08 artifact ${SLURM_ARRAY_TASK_ID}" >&2
    exit 42
fi
if ! python3 -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "CUDA preflight failed (torch) before Exp. 08 artifact ${SLURM_ARRAY_TASK_ID}" >&2
    exit 42
fi

bash scripts/run.sh exp08 \
    --prepare-artifact \
    --artifact-index "$SLURM_ARRAY_TASK_ID" \
    --device cuda
