#!/bin/bash
#SBATCH --job-name=exp05_atk_stop
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --array=0-7

set -euo pipefail

PARTS=(a b c d e f g h)
PART="${PARTS[$SLURM_ARRAY_TASK_ID]}"

echo ">>> Job Exp05 attack early-stopping audit part ${PART} started on $(date)"
echo ">>> Node: $(hostname)"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate fl_env_grid

mkdir -p results logs

if [ ! -e "./data" ]; then
    if [ -d "../data" ]; then
        echo ">>> Found data at ../data, creating symlink..."
        ln -s ../data ./data
    elif [ -d "../../data" ]; then
        echo ">>> Found data at ../../data, creating symlink..."
        ln -s ../../data ./data
    elif [ -d "$HOME/data" ]; then
        echo ">>> Found data at $HOME/data, creating symlink..."
        ln -s "$HOME/data" ./data
    else
        echo "ERROR: data directory not found."
        exit 1
    fi
fi

python -u experiments/exp02_dp_frontier/code/05_FL_AttackEarlyStop_Audit_v25.py --part "$PART"

echo ">>> Job Exp05 attack early-stopping audit part ${PART} finished on $(date)"
