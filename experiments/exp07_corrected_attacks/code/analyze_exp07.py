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


def completeness(rows: list[dict], arm: str | None = None) -> dict:
    expected = {
        config.config_id for config in build_protocol()
        if arm is None or config.arm == arm
    }
    observed = {
        row["config"]["config_id"] for row in rows
        if arm is None or row["config"]["arm"] == arm
    }
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
        aggregation = row.get("final", {}).get("aggregation")
        if aggregation is None and row.get("round_metrics"):
            aggregation = row["round_metrics"][-1].get("aggregation")
        degraded.append({
            **config,
            "f1_macro": row["final"]["f1_macro"],
            "clean_f1_macro": clean[key],
            "degradation": row["final"]["f1_macro"] - clean[key],
            "theoretical_condition_met": (
                aggregation.get("theoretical_condition_met")
                if aggregation is not None else None
            ),
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


def _paired_wilcoxon(values: np.ndarray) -> dict:
    """Exact paired Wilcoxon test on within-seed topology differences."""

    values = np.asarray(values, dtype=float)
    values = values[~np.isclose(values, 0.0)]
    if not len(values):
        return {"method": "paired-wilcoxon-exact", "statistic": 0.0,
                "pvalue_two_sided": 1.0, "nonzero_pairs": 0}
    absolute = np.abs(values)
    order = np.argsort(absolute, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and np.isclose(
            absolute[order[stop]], absolute[order[start]]
        ):
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    rank_sum = float(ranks.sum())
    positive = float(ranks[values > 0].sum())
    statistic = min(positive, rank_sum - positive)
    enumerated = []
    for signs in range(1 << len(values)):
        candidate = sum(
            rank for index, rank in enumerate(ranks) if signs & (1 << index)
        )
        enumerated.append(min(candidate, rank_sum - candidate))
    pvalue = float(np.mean(np.asarray(enumerated) <= statistic + 1e-12))
    return {
        "method": "paired-wilcoxon-exact",
        "statistic": statistic,
        "pvalue_two_sided": pvalue,
        "nonzero_pairs": int(len(values)),
    }


def _bootstrap_summary(values: np.ndarray, rng, bootstrap_samples: int) -> tuple[float, list[float]]:
    boot = rng.choice(values, size=(bootstrap_samples, len(values)), replace=True).mean(axis=1)
    low, high = np.quantile(boot, [0.025, 0.975])
    return float(values.mean()), [float(low), float(high)]


def dataset_summary(effects: list[dict], bootstrap_samples: int = 10_000) -> list[dict]:
    """Indicative seed-clustered summaries over parameter-attack cells."""

    output = []
    rng = np.random.default_rng(7)
    for dataset in sorted({row["dataset"] for row in effects}):
        selected = [row for row in effects
                    if row["dataset"] == dataset and row["attack"] in PARAMETER_ATTACKS]
        by_seed = defaultdict(list)
        for row in selected:
            by_seed[row["seed"]].append(row["topology_effect"])
        seed_means = np.asarray(
            [np.mean(by_seed[seed]) for seed in sorted(by_seed)], dtype=float
        )
        if not len(seed_means):
            continue
        mean, ci95 = _bootstrap_summary(seed_means, rng, bootstrap_samples)
        output.append({
            "dataset": dataset,
            "mean_topology_effect": mean,
            "ci95": ci95,
            "seed_count": int(len(seed_means)),
            "paired_test": _paired_wilcoxon(seed_means),
            "supports_vfl_generalization": bool(ci95[0] > 0),
            "inference_warning": (
                "Only five seed clusters: the bootstrap interval and paired test are "
                "indicative, not strong confirmatory evidence. Inspect condition_effects "
                "for reversals before interpreting this aggregate."
            ),
        })
    return output


def dataset_confirmatory(effects: list[dict], bootstrap_samples: int = 10_000) -> list[dict]:
    """Backward-compatible alias; returned records explicitly limit the inference."""

    return dataset_summary(effects, bootstrap_samples)


def condition_effects(effects: list[dict], bootstrap_samples: int = 10_000) -> list[dict]:
    """Expose topology-effect reversals instead of hiding them in a dataset mean."""

    grouped = defaultdict(list)
    fields = ("dataset", "fusion", "epsilon", "attack", "attack_ratio")
    for row in effects:
        grouped[tuple(row[name] for name in fields)].append(row)
    rng = np.random.default_rng(11)
    output = []
    for key in sorted(grouped, key=str):
        by_seed = defaultdict(list)
        for row in grouped[key]:
            by_seed[row["seed"]].append(row["topology_effect"])
        seed_means = np.asarray(
            [np.mean(by_seed[seed]) for seed in sorted(by_seed)], dtype=float
        )
        mean, ci95 = _bootstrap_summary(seed_means, rng, bootstrap_samples)
        output.append({
            **dict(zip(fields, key)),
            "mean_topology_effect": mean,
            "ci95": ci95,
            "seed_count": int(len(seed_means)),
            "paired_test": _paired_wilcoxon(seed_means),
            "inference_warning": "Five seed clusters; interval is indicative and cell-specific.",
        })
    return output


def secondary_aggregator_effects(degraded: list[dict]) -> list[dict]:
    warning = (
        "Secondary robust-aggregation arm uses only three seeds; report descriptively "
        "and do not use it for broad superiority claims."
    )
    return [
        {**row, "inference_warning": warning}
        for row in degraded if row["arm"] == "secondary"
    ]


def secondary_aggregator_summary(rows: list[dict]) -> list[dict]:
    fields = ("dataset", "fusion", "epsilon", "attack", "attack_ratio", "aggregator")
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[name] for name in fields)].append(row)
    output = []
    for key in sorted(grouped, key=str):
        selected = grouped[key]
        validity = {row["theoretical_condition_met"] for row in selected}
        output.append({
            **dict(zip(fields, key)),
            "mean_degradation": float(np.mean([row["degradation"] for row in selected])),
            "seed_count": len({row["seed"] for row in selected}),
            "theoretical_condition_met": validity.pop() if len(validity) == 1 else None,
            "inference_warning": selected[0]["inference_warning"],
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
    integrity = {
        "overall": completeness(rows),
        "primary": completeness(rows, "primary"),
        "secondary": completeness(rows, "secondary"),
    }
    if integrity["primary"]["missing"] and not args.allow_partial:
        raise SystemExit(
            f"Exp. 07 primary arm incomplete: {len(integrity['primary']['missing'])} missing jobs"
        )
    degraded = degradation_rows(rows)
    effects = topology_effects(degraded)
    secondary_rows = secondary_aggregator_effects(degraded)
    report = {
        "integrity": integrity,
        "dataset_level_indicative": dataset_summary(effects),
        "condition_effects": condition_effects(effects),
        "topology_effects": effects,
        "label_flip_effects": [row for row in effects if row["attack"] == "label_flip"],
        "secondary_robust_aggregation": {
            "paired_rows": secondary_rows,
            "cell_summary": secondary_aggregator_summary(secondary_rows),
            "inference_warning": (
                "This HFL-only arm has three seeds and is not part of the causal "
                "HFL-versus-VFL topology comparison."
            ),
        },
    }
    output = args.results_dir / "analysis_exp07_corrected.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "integrity": integrity,
                      "dataset_level_indicative": report["dataset_level_indicative"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
