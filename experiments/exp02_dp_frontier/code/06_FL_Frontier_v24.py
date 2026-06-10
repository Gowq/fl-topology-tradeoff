#!/usr/bin/env python3
"""Exp06 — Privacy-Utility Frontier estendida (Ideia A).

Re-roda OPPORTUNITY HFL + VFL × {Intermediate, Late} sob εs altos
{10, 20, 50, 100, 200} para mapear onde cada (topologia × fusion) sai
do random floor após a correção do bug de calibração DP.

Reusa run_horizontal_fl e run_vertical_fl do exp01_baseline via importlib.

Output: results/exp06_frontier_results.json

Decisão registrada em silo/decisions/2026-05-23_paper-rescue-experiments.md
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

HERE     = Path(__file__).resolve().parent
REPO     = HERE.parents[2]
SHARED   = REPO / "experiments" / "shared" / "code"
EXP01    = REPO / "experiments" / "exp01_baseline" / "code"

if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from timeout_utils import run_with_timeout, format_timeout_duration


def _load(module_name: str, fname: str):
    spec = importlib.util.spec_from_file_location(module_name, EXP01 / fname)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


p1 = _load("baseline_part1", "01_FL_Baseline_Grid_v24_part1.py")
p2 = _load("baseline_part2", "01_FL_Baseline_Grid_v24_part2.py")

# ── Sweep config ────────────────────────────────────────────────────────────
FRONTIER_EPSILONS   = [10.0, 20.0, 50.0, 100.0, 200.0]
DATASET_OPPORTUNITY = "OPPORTUNITY"
SEEDS               = p1.SEEDS  # [42, 123, 456]

# 4 topologia×fusion combos × 5 ε × 3 seeds = 60 runs total
configs = []
for fusion in ("Intermediate", "Late"):
    for eps in FRONTIER_EPSILONS:
        configs.append({"dataset": DATASET_OPPORTUNITY, "topo": "horizontal", "fusion": fusion, "eps": eps})
for fusion in ("Intermediate", "Late"):
    for eps in FRONTIER_EPSILONS:
        configs.append({"dataset": DATASET_OPPORTUNITY, "topo": "vertical",   "fusion": fusion, "eps": eps})


def main():
    os.makedirs("results", exist_ok=True)
    out_path     = Path("results/exp06_frontier_results.json")
    partial_path = Path("results/partial_exp06_frontier_results.json")

    # Resume: skip configs already saved
    done = []
    if partial_path.exists():
        done = json.loads(partial_path.read_text())
        print(f">>> Resume: {len(done)}/{len(configs)} configs already done, skipping.")

    print(f"=== Exp06 FRONTIER: OPPORTUNITY × {{HFL, VFL}} × {{Inter, Late}} × ε={FRONTIER_EPSILONS} ===")
    print(f"Total configs: {len(configs)} | seeds: {len(SEEDS)} | total runs: {len(configs)*len(SEEDS)}")
    print(f"Timeout per run: {format_timeout_duration(7200)}")

    all_results = list(done)
    done_indices = set(range(len(done)))

    for i, exp in enumerate(configs):
        if i in done_indices:
            continue
        topo_label = exp["topo"].capitalize()
        print(f"\n[{i+1}/{len(configs)}] OPPORTUNITY {topo_label}/{exp['fusion']} eps={exp['eps']}")

        seed_results = []
        for seed in SEEDS:
            print(f"  Seed {seed}", end=" ", flush=True)

            def run_one(e=exp, s=seed):
                if e["topo"] == "horizontal":
                    return p1.run_horizontal_fl(e["dataset"], e["fusion"], e["eps"], s)
                else:
                    return p2.run_vertical_fl(e["fusion"], e["eps"], s)

            success, res = run_with_timeout(run_one, timeout_seconds=7200,
                                            on_timeout=lambda: print("\n  ⚠️ TIMEOUT 2h"))
            if not success:
                print(" [TIMEOUT]")
                seed_results.append([])
                continue
            seed_results.append(res or [])
            if res:
                print(f" F1={res[-1].get('f1', 0.0):.4f}")
            else:
                print(" [FAILED]")

        all_results.append({"config": exp, "runs": seed_results})
        partial_path.write_text(json.dumps(all_results, indent=2))

    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
