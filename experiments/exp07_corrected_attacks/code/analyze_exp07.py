#!/usr/bin/env python3
"""Validate Exp. 07 and estimate paired topology degradation effects."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from protocol import build_protocol


PARAMETER_ATTACKS = {"sign_flip", "scaling", "free_rider"}


def load_results(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("experiment") == "exp07_corrected_cross_dataset_attacks":
            rows.append(payload)
    return rows


def completeness(rows: list[dict]) -> dict:
    expected = {config.config_id for config in build_protocol()}
    observed = {row["config"]["config_id"] for row in rows}
    return {
        "expected": len(expected),
        "observed": len(observed),
        "missing": sorted(expected - observed),
        "extra": sorted(observed - expected),
    }


def degradation_rows(rows: list[dict]) -> list[dict]:
    clean = {}
    for row in rows:
        config = row["config"]
        if config["attack"] != "none":
            continue
        key = tuple(config[name] for name in (
            "arm", "dataset", "topology", "fusion", "epsilon", "aggregator", "seed"
        ))
        clean[key] = row["final"]["f1_macro"]

    degraded = []
    for row in rows:
        config = row["config"]
        if config["attack"] == "none":
            continue
        key = tuple(config[name] for name in (
            "arm", "dataset", "topology", "fusion", "epsilon", "aggregator", "seed"
        ))
        if key not in clean:
            continue
        degraded.append({
            **config,
            "f1_macro": row["final"]["f1_macro"],
            "clean_f1_macro": clean[key],
            "degradation": row["final"]["f1_macro"] - clean[key],
        })
    return degraded


def topology_effects(degraded: list[dict]) -> list[dict]:
    indexed = {}
    for row in degraded:
        if row["arm"] != "primary":
            continue
        key = tuple(row[name] for name in (
            "dataset", "fusion", "epsilon", "attack", "attack_ratio", "seed"
        ))
        indexed[(key, row["topology"])] = row
    effects = []
    for (key, topology), hfl in indexed.items():
        if topology != "hfl" or (key, "vfl") not in indexed:
            continue
        vfl = indexed[(key, "vfl")]
        effects.append({
            "dataset": key[0], "fusion": key[1], "epsilon": key[2],
            "attack": key[3], "attack_ratio": key[4], "seed": key[5],
            "hfl_degradation": hfl["degradation"],
            "vfl_degradation": vfl["degradation"],
            "topology_effect": vfl["degradation"] - hfl["degradation"],
        })
    return effects


def dataset_confirmatory(effects: list[dict], bootstrap_samples: int = 10_000) -> list[dict]:
    """Seed-clustered bootstrap over parameter-attack cells, separately by dataset."""

    output = []
    rng = np.random.default_rng(7)
    for dataset in sorted({row["dataset"] for row in effects}):
        selected = [row for row in effects
                    if row["dataset"] == dataset and row["attack"] in PARAMETER_ATTACKS]
        by_seed = defaultdict(list)
        for row in selected:
            by_seed[row["seed"]].append(row["topology_effect"])
        seed_means = np.asarray([np.mean(values) for values in by_seed.values()], dtype=float)
        if not len(seed_means):
            continue
        boot = rng.choice(seed_means, size=(bootstrap_samples, len(seed_means)), replace=True).mean(axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
        output.append({
            "dataset": dataset,
            "mean_topology_effect": float(seed_means.mean()),
            "ci95": [float(low), float(high)],
            "seed_count": int(len(seed_means)),
            "supports_vfl_generalization": bool(low > 0),
        })
    return output


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--allow-partial", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = load_results(args.results_dir)
    integrity = completeness(rows)
    if integrity["missing"] and not args.allow_partial:
        raise SystemExit(f"Exp. 07 incomplete: {len(integrity['missing'])} missing jobs")
    degraded = degradation_rows(rows)
    effects = topology_effects(degraded)
    report = {
        "integrity": integrity,
        "confirmatory": dataset_confirmatory(effects),
        "topology_effects": effects,
        "label_flip_effects": [row for row in effects if row["attack"] == "label_flip"],
    }
    output = args.results_dir / "analysis_exp07_corrected.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "integrity": integrity,
                      "confirmatory": report["confirmatory"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

