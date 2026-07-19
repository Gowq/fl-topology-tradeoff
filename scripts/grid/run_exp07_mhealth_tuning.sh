#!/bin/bash
#SBATCH --job-name=exp07_tune
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --array=0-71%4

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fl_env_grid

mkdir -p logs
bash scripts/run.sh exp07-tune --job-index "$SLURM_ARRAY_TASK_ID" --device cuda
