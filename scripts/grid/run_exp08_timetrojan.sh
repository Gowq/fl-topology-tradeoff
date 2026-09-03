#!/usr/bin/env bash
#SBATCH --job-name=exp08_timetrojan
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --array=0-63%2
#SBATCH --requeue
#SBATCH --output=logs/exp08_timetrojan_%A_%a.out
#SBATCH --error=logs/exp08_timetrojan_%A_%a.err

# Stage 2 of Exp. 08: the 384-cell federated matrix (240 primary + 144
# secondary). Per GridUNESP guidance this is a single partial array of small
# batches: BLOCK_SIZE configs per task (64 tasks x 6 = 384), %2 concurrency on
# the single GPU node, and a realistic walltime instead of the 24h ceiling.
# run_exp08.py writes one JSON per config and skips already-complete configs, so
# a requeue resumes inside the batch without recomputation. Submit AFTER the
# artifact stage, e.g.:
#   JID=$(sbatch --parsable scripts/grid/run_exp08_timetrojan_artifacts.sh)
#   sbatch --dependency=afterok:"$JID" scripts/grid/run_exp08_timetrojan.sh

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

if ! command -v nvidia-smi >/dev/null || ! nvidia-smi -L >/dev/null; then
    echo "CUDA preflight failed (driver) before Exp. 08 batch ${SLURM_ARRAY_TASK_ID}" >&2
    exit 42
fi
if ! python3 -c 'import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)'; then
    echo "CUDA preflight failed (torch) before Exp. 08 batch ${SLURM_ARRAY_TASK_ID}" >&2
    exit 42
fi

BLOCK_SIZE=6
TOTAL=384
START=$((SLURM_ARRAY_TASK_ID * BLOCK_SIZE))
END=$((START + BLOCK_SIZE - 1))

for index in $(seq "$START" "$END"); do
    if [ "$index" -ge "$TOTAL" ]; then
        break
    fi
    echo ">>> Exp. 08 batch ${SLURM_ARRAY_TASK_ID}: config-index ${index}"
    bash scripts/run.sh exp08 --config-index "$index" --device cuda
done
