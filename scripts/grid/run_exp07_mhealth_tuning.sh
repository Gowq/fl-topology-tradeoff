#!/bin/bash
#SBATCH --job-name=exp07_tune
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1

set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fl_env_grid

mkdir -p logs
bash scripts/run.sh exp07-tune --device cuda
