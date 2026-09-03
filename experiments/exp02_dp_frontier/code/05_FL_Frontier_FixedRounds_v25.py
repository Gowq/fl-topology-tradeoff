#!/usr/bin/env python3
"""Exp05 diagnostic: fixed-round DP frontier tail.

This experiment isolates the epsilon=200 HFL/VFL crossover observed in the
frontier results by disabling patience/loss-stagnation early stopping while
keeping the same DP accountant calibration, model code, seeds, and best-checkpoint
reporting. The goal is to test whether VFL's apparent plateau is caused by the
training loop stopping before it consumes the target privacy budget.

Default sweep:
  OPPORTUNITY x {HFL,VFL} x {Intermediate,Late} x epsilon {100,200} x 3 seeds

Use --smoke-only before Pegasus/Grid deployment.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SHARED = REPO / "experiments" / "shared" / "code"
EXP01 = REPO / "experiments" / "exp01_baseline" / "code"

if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from timeout_utils import format_timeout_duration, run_with_timeout


def _load(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, EXP01 / filename)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


p1 = _load("baseline_part1_fixed_rounds", "01_FL_Baseline_Grid_v24_part1.py")
p2 = _load("baseline_part2_fixed_rounds", "01_FL_Baseline_Grid_v24_part2.py")

DATASET_OPPORTUNITY = "OPPORTUNITY"
TAIL_EPSILONS = [100.0, 200.0]
PARTS = {
    "a": [("horizontal", "Intermediate")],
    "b": [("vertical", "Intermediate")],
    "c": [("horizontal", "Late")],
    "d": [("vertical", "Late")],
}


def build_configs(part: str, smoke_only: bool) -> list[dict]:
    if smoke_only:
        return [
            {"dataset": DATASET_OPPORTUNITY, "topo": "horizontal", "fusion": "Intermediate", "eps": 200.0},
            {"dataset": DATASET_OPPORTUNITY, "topo": "vertical", "fusion": "Intermediate", "eps": 200.0},
        ]

    pairs = []
    if part == "all":
        for value in ("a", "b", "c", "d"):
            pairs.extend(PARTS[value])
    else:
        pairs = PARTS[part]

    configs = []
    for topo, fusion in pairs:
        for eps in TAIL_EPSILONS:
            configs.append({"dataset": DATASET_OPPORTUNITY, "topo": topo, "fusion": fusion, "eps": eps})
    return configs


def output_paths(part: str, smoke_only: bool) -> tuple[Path, Path]:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    suffix = "smoke" if smoke_only else f"part{part}"
    out_path = results_dir / f"exp05_fixed_rounds_{suffix}.json"
    partial_path = results_dir / f"partial_exp05_fixed_rounds_{suffix}.json"
    return out_path, partial_path


def run_config(exp: dict, seed: int, rounds: int, epochs: int):
    if exp["topo"] == "horizontal":
        return p1.run_horizontal_fl(
            exp["dataset"],
            exp["fusion"],
            exp["eps"],
            seed,
            num_rounds=rounds,
            epochs=epochs,
            early_stopping=False,
        )
    return p2.run_vertical_fl(
        exp["fusion"],
        exp["eps"],
        seed,
        num_rounds=rounds,
        epochs=epochs,
        early_stopping=False,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["all", "a", "b", "c", "d"], default="all")
    parser.add_argument("--smoke-only", action="store_true", help="Run 1 seed, 1 round, 1 epoch for Pegasus validation.")
    parser.add_argument("--rounds", type=int, default=p1.NUM_ROUNDS)
    parser.add_argument("--epochs", type=int, default=p1.LOCAL_EPOCHS)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()

    seeds = [p1.SEEDS[0]] if args.smoke_only else list(p1.SEEDS)
    rounds = 1 if args.smoke_only else args.rounds
    epochs = 1 if args.smoke_only else args.epochs
    configs = build_configs(args.part, args.smoke_only)
    out_path, partial_path = output_paths(args.part, args.smoke_only)

    done = []
    if partial_path.exists() and not args.smoke_only:
        done = json.loads(partial_path.read_text())
        print(f">>> Resume: {len(done)}/{len(configs)} configs already done, skipping.")

    print("=== Exp05 fixed-round frontier diagnostic ===")
    print(f"Part: {args.part} | smoke: {args.smoke_only}")
    print(f"Configs: {len(configs)} | seeds: {seeds} | rounds: {rounds} | epochs: {epochs}")
    print(f"Early stopping: disabled except loss explosion safety")
    print(f"Timeout per run: {format_timeout_duration(args.timeout_seconds)}")

    all_results = list(done)
    done_count = len(done)

    for i, exp in enumerate(configs[done_count:], start=done_count):
        print(f"\n[{i + 1}/{len(configs)}] {exp['topo']} {exp['fusion']} eps={exp['eps']}")
        seed_results = []
        for seed in seeds:
            print(f"  Seed {seed}", end=" ", flush=True)

            success, result = run_with_timeout(
                lambda e=exp, s=seed: run_config(e, s, rounds, epochs),
                timeout_seconds=args.timeout_seconds,
                on_timeout=lambda: print("\n  TIMEOUT"),
            )
            if not success:
                print(" [TIMEOUT]")
                seed_results.append([])
                continue
            seed_results.append(result or [])
            if result:
                last = result[-1]
                print(f" F1={last.get('f1', 0.0):.4f} eps={last.get('epsilon', 0.0):.2f}")
            else:
                print(" [FAILED]")

        all_results.append({
            "config": {
                **exp,
                "fixed_rounds": True,
                "early_stopping": False,
                "rounds": rounds,
                "epochs": epochs,
            },
            "runs": seed_results,
        })
        partial_path.write_text(json.dumps(all_results, indent=2))

    out_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved -> {out_path}")
    if args.smoke_only:
        print("SMOKE TEST PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
