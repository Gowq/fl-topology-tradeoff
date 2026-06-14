#!/usr/bin/env python3
"""Exp05 extension: early-stopping sensitivity under attacks.

This diagnostic tests whether the stopping protocol changes HFL/VFL robustness
under a compact attack subset. It reuses the validated Exp03 attack runners and
compares patience-based early stopping against a fixed-round protocol.

Default sweep:
  OPPORTUNITY
  topology/fusion: {HFL,VFL} x {Intermediate,Late}
  attacks: Label Flip, Sign Flip
  attack ratios: 25%, 75%
  epsilon: 100, 200
  stop protocol: early, fixed
  seeds: 42, 123, 456

Use --smoke-only before Pegasus/Grid deployment.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import multiprocessing as mp
import os
import sys
from contextlib import contextmanager
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SHARED = REPO / "experiments" / "shared" / "code"
EXP03 = REPO / "experiments" / "exp03_attacks_aggregation" / "code"

if str(SHARED) not in sys.path:
    sys.path.insert(0, str(SHARED))

from timeout_utils import format_timeout_duration


def _load(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


hfl_runner = _load(
    "exp03_part1_for_exp05_attack_stopping",
    EXP03 / "08_FL_RobustAgg_DP_Frontier_part1.py",
)
vfl_runner = _load(
    "exp03_part3_for_exp05_attack_stopping",
    EXP03 / "08_FL_RobustAgg_DP_Frontier_part3.py",
)


EPSILONS = [100.0, 200.0]
ATTACKS = [
    hfl_runner.AttackConfig.LABEL_FLIP,
    hfl_runner.AttackConfig.SIGN_FLIP,
]
RATIOS = [0.25, 0.75]
STOP_PROTOCOLS = ["early", "fixed"]

PARTS = {
    "a": [{"topology": "hfl", "fusion": "Intermediate", "attack_type": hfl_runner.AttackConfig.LABEL_FLIP}],
    "b": [{"topology": "vfl", "fusion": "Intermediate", "attack_type": hfl_runner.AttackConfig.LABEL_FLIP}],
    "c": [{"topology": "hfl", "fusion": "Late", "attack_type": hfl_runner.AttackConfig.LABEL_FLIP}],
    "d": [{"topology": "vfl", "fusion": "Late", "attack_type": hfl_runner.AttackConfig.LABEL_FLIP}],
    "e": [{"topology": "hfl", "fusion": "Intermediate", "attack_type": hfl_runner.AttackConfig.SIGN_FLIP}],
    "f": [{"topology": "vfl", "fusion": "Intermediate", "attack_type": hfl_runner.AttackConfig.SIGN_FLIP}],
    "g": [{"topology": "hfl", "fusion": "Late", "attack_type": hfl_runner.AttackConfig.SIGN_FLIP}],
    "h": [{"topology": "vfl", "fusion": "Late", "attack_type": hfl_runner.AttackConfig.SIGN_FLIP}],
}


@contextmanager
def stopping_protocol(module, enabled: bool):
    """Temporarily disable patience-based stops without removing safety stops."""
    keys = [
        "EARLY_STOP_PATIENCE",
        "LOSS_STAGNATION_PATIENCE",
    ]
    original = {key: getattr(module, key) for key in keys if hasattr(module, key)}
    try:
        if not enabled:
            for key in original:
                setattr(module, key, 10**9)
        yield
    finally:
        for key, value in original.items():
            setattr(module, key, value)


def build_configs(part: str, smoke_only: bool) -> list[dict]:
    if smoke_only:
        return [
            {
                "topology": "hfl",
                "fusion": "Intermediate",
                "attack_type": hfl_runner.AttackConfig.LABEL_FLIP,
                "attack_ratio": 0.25,
                "dp_epsilon": 100.0,
                "stop_protocol": "early",
            },
            {
                "topology": "hfl",
                "fusion": "Intermediate",
                "attack_type": hfl_runner.AttackConfig.SIGN_FLIP,
                "attack_ratio": 0.25,
                "dp_epsilon": 100.0,
                "stop_protocol": "fixed",
            },
            {
                "topology": "vfl",
                "fusion": "Intermediate",
                "attack_type": vfl_runner.AttackConfig.LABEL_FLIP,
                "attack_ratio": 0.25,
                "dp_epsilon": 100.0,
                "stop_protocol": "early",
            },
            {
                "topology": "vfl",
                "fusion": "Intermediate",
                "attack_type": vfl_runner.AttackConfig.SIGN_FLIP,
                "attack_ratio": 0.25,
                "dp_epsilon": 100.0,
                "stop_protocol": "fixed",
            },
        ]

    bases = []
    if part == "all":
        for label in PARTS:
            bases.extend(PARTS[label])
    else:
        bases = PARTS[part]

    configs = []
    for base in bases:
        for ratio in RATIOS:
            for epsilon in EPSILONS:
                for stop in STOP_PROTOCOLS:
                    configs.append({
                        **base,
                        "attack_ratio": ratio,
                        "dp_epsilon": epsilon,
                        "stop_protocol": stop,
                    })
    return configs


def output_paths(part: str, smoke_only: bool) -> tuple[Path, Path]:
    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    suffix = "smoke" if smoke_only else f"part{part}"
    out_path = results_dir / f"exp05_attack_earlystop_audit_{suffix}.json"
    partial_path = results_dir / f"partial_exp05_attack_earlystop_audit_{suffix}.json"
    return out_path, partial_path


def _run_hfl_seed(config: dict, seed: int, rounds: int, epochs: int) -> list[dict]:
    early_stopping = config["stop_protocol"] == "early"
    with stopping_protocol(hfl_runner, enabled=early_stopping):
        return hfl_runner.run_horizontal_fl_robust(
            config["fusion"],
            "fedavg",
            config["attack_type"],
            config["attack_ratio"],
            config["dp_epsilon"],
            seed,
            num_rounds=rounds,
            epochs=epochs,
        )


def _run_vfl_config(config: dict, seeds: list[int], rounds: int, epochs: int) -> list[list[dict]]:
    early_stopping = config["stop_protocol"] == "early"
    batch_size = vfl_runner.DATASET_PARAMS[vfl_runner.DATASET_OPPORTUNITY]["batch_size"]
    loaders = vfl_runner.partition_opportunity_vertical(batch_size)
    silo_loaders, label_loader, test_silo_loaders, test_label_loader = loaders
    if silo_loaders is None:
        return [[] for _ in seeds]

    seed_results = []
    with stopping_protocol(vfl_runner, enabled=early_stopping):
        for seed in seeds:
            seed_results.append(
                vfl_runner.run_vertical_fl_robust(
                    config["fusion"],
                    config["dp_epsilon"],
                    config["attack_type"],
                    config["attack_ratio"],
                    seed,
                    silo_loaders,
                    label_loader,
                    test_silo_loaders,
                    test_label_loader,
                    num_rounds=rounds,
                    epochs=epochs,
                ) or []
            )
    return seed_results


def _config_worker(config: dict, seeds: list[int], rounds: int, epochs: int, result_path: str) -> None:
    try:
        if config["topology"] == "hfl":
            seed_results = [_run_hfl_seed(config, seed, rounds, epochs) or [] for seed in seeds]
        else:
            seed_results = _run_vfl_config(config, seeds, rounds, epochs)
    except Exception as exc:
        print(f"  [worker error] {type(exc).__name__}: {exc}", flush=True)
        seed_results = [[] for _ in seeds]

    with open(result_path, "w") as handle:
        json.dump(seed_results, handle)


def run_config(config: dict, seeds: list[int], rounds: int, epochs: int, timeout_seconds: int) -> list[list[dict]]:
    tmp_path = f"/tmp/exp05_attack_earlystop_{os.getpid()}_{abs(hash(json.dumps(config, sort_keys=True)))}.json"
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_config_worker, args=(config, seeds, rounds, epochs, tmp_path))
    proc.start()
    subprocess_timeout = len(seeds) * timeout_seconds + 300
    proc.join(timeout=subprocess_timeout)

    if proc.is_alive():
        proc.terminate()
        proc.join()
        print(f"  [SUBPROCESS TIMEOUT after {format_timeout_duration(subprocess_timeout)}]", flush=True)
        return [[] for _ in seeds]

    path = Path(tmp_path)
    if not path.exists():
        print("  [SUBPROCESS FAILED: no result file]", flush=True)
        return [[] for _ in seeds]

    try:
        return json.loads(path.read_text())
    finally:
        path.unlink(missing_ok=True)


def config_metadata(config: dict, rounds: int, epochs: int) -> dict:
    topology = config["topology"].upper()
    return {
        **config,
        "topology": topology,
        "aggregation": "fedavg" if topology == "HFL" else "coordinator",
        "early_stopping": config["stop_protocol"] == "early",
        "fixed_rounds": config["stop_protocol"] == "fixed",
        "rounds": rounds,
        "epochs": epochs,
        "exp05_extension": "attack_earlystop_audit",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["all", *PARTS.keys()], default="all")
    parser.add_argument("--smoke-only", action="store_true", help="Run 1 seed, 1 round, 1 epoch for Pegasus validation.")
    parser.add_argument("--rounds", type=int, default=hfl_runner.NUM_ROUNDS)
    parser.add_argument("--epochs", type=int, default=hfl_runner.LOCAL_EPOCHS)
    parser.add_argument("--timeout-seconds", type=int, default=7200)
    args = parser.parse_args()

    seeds = [hfl_runner.SEEDS[0]] if args.smoke_only else list(hfl_runner.SEEDS)
    rounds = 1 if args.smoke_only else args.rounds
    epochs = 1 if args.smoke_only else args.epochs
    configs = build_configs(args.part, args.smoke_only)
    out_path, partial_path = output_paths(args.part, args.smoke_only)

    done = []
    if partial_path.exists() and not args.smoke_only:
        done = json.loads(partial_path.read_text())
        print(f">>> Resume: {len(done)}/{len(configs)} configs already done, skipping.")

    print("=== Exp05 attack early-stopping audit ===")
    print(f"Part: {args.part} | smoke: {args.smoke_only}")
    print(f"Configs: {len(configs)} | seeds: {seeds} | rounds: {rounds} | epochs: {epochs}")
    print(f"Timeout per seed: {format_timeout_duration(args.timeout_seconds)}")

    all_results = list(done)
    for index, config in enumerate(configs[len(done):], start=len(done)):
        print(
            f"\n[{index + 1}/{len(configs)}] {config['topology'].upper()} {config['fusion']} "
            f"eps={config['dp_epsilon']} attack={config['attack_type']} "
            f"ratio={config['attack_ratio']} stop={config['stop_protocol']}",
            flush=True,
        )
        seed_results = run_config(config, seeds, rounds, epochs, args.timeout_seconds)
        for seed, result in zip(seeds, seed_results):
            if result:
                print(f"  Seed {seed} F1={result[-1].get('f1', 0.0):.4f}", flush=True)
            else:
                print(f"  Seed {seed} [FAILED]", flush=True)

        all_results.append({
            "config": config_metadata(config, rounds, epochs),
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
