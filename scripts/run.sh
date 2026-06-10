#!/usr/bin/env bash
# Orchestrate the experiment suite. Each experiment reads ./data and writes its
# own results/ dir; the shipped results/ already contain the published runs, so
# this is only needed to *reproduce* them from scratch (GPU strongly advised).
#
# Usage:
#   bash scripts/run.sh <experiment> [script-args...]
#   bash scripts/run.sh figures            # regenerate all figures from results/
#   bash scripts/run.sh list
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$ROOT/experiments/shared/code:${PYTHONPATH:-}"
export MPLBACKEND=Agg

run_exp() {
    local dir="$1"; shift
    local code="$ROOT/experiments/$dir/code"
    # link dataset + results dir into the experiment's code/ (scripts read ./data, ./results)
    [ -e "$code/data" ]    || ln -sf "$ROOT/data" "$code/data"
    mkdir -p "$code/results"
    cd "$code"
    echo ">>> Running $dir : $*"
    python3 -u "$@"
}

case "${1:-list}" in
  exp01|baseline)
    run_exp exp01_baseline 01_FL_Baseline_Grid_v24_part1.py
    run_exp exp01_baseline 01_FL_Baseline_Grid_v24_part2.py
    run_exp exp01_baseline 01_FL_Baseline_Grid_v24_part3.py
    run_exp exp01_baseline 01_FL_Baseline_Grid_v24_cifar100.py ;;
  exp02|frontier)
    run_exp exp02_dp_frontier 06_FL_Frontier_v24.py ;;
  exp03|attacks)
    run_exp exp03_attacks_aggregation 08_FL_RobustAgg_DP_Frontier_part1.py
    run_exp exp03_attacks_aggregation 08_FL_RobustAgg_DP_Frontier_part2.py
    run_exp exp03_attacks_aggregation 08_FL_RobustAgg_DP_Frontier_part3.py ;;
  exp04|overhead)
    run_exp exp04_overhead 04_overhead_benchmark.py ;;
  defense-semantic)
    run_exp defense_semantic_filtering 09_FL_VFL_Semantic_Defenses.py "${@:2}" ;;
  defense-losses)
    run_exp defense_robust_losses 10_FL_VFL_Robust_Losses.py "${@:2}" ;;
  figures)
    cd "$ROOT" && python3 figures/generate_figures.py ;;
  smoke)
    # fast end-to-end stack check (1 seed, smallest matrix) — minutes on CPU
    run_exp defense_robust_losses 10_FL_VFL_Robust_Losses.py --smoke-seeds 1 ;;
  all|reproduce)
    # full reproduction of the four primary experiments, then figures.
    # heavy: days of GPU time for Exp.03; intended for GPU hosts.
    bash "$0" exp01
    bash "$0" exp02
    bash "$0" exp03
    bash "$0" exp04
    bash "$0" figures ;;
  list|*)
    echo "usage: scripts/run.sh <target>"
    echo "  primary:    exp01|baseline  exp02|frontier  exp03|attacks  exp04|overhead"
    echo "  defenses:   defense-semantic [--smoke-seeds N]   defense-losses [--smoke-seeds N]"
    echo "  pipelines:  all|reproduce (exp01-04 + figures)   smoke (fast stack check)   figures" ;;
esac
