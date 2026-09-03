#!/usr/bin/env python3
"""Validate Exp. 08 and estimate paired ASR-uplift topology effects."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from protocol import build_protocol


def load_results(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if payload.get("experiment") == "exp08_timetrojan_topology":
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


def uplift_rows(rows: list[dict]) -> list[dict]:
    clean = {}
    for row in rows:
        config = row["config"]
        if config["poison_rate"] != 0.0:
            continue
        key = tuple(config[name] for name in (
            "arm", "dataset", "topology", "fusion", "epsilon", "aggregator", "seed"
        ))
        clean[key] = row

    output = []
    for row in rows:
        config = row["config"]
        if config["poison_rate"] == 0.0:
            continue
        key = tuple(config[name] for name in (
            "arm", "dataset", "topology", "fusion", "epsilon", "aggregator", "seed"
        ))
        if key not in clean:
            continue
        clean_final = clean[key]["final"]
        poison_final = row["final"]
        output.append({
            **config,
            "clean_asr_non_target": clean_final["asr_non_target"],
            "poisoned_asr_non_target": poison_final["asr_non_target"],
            "asr_uplift": poison_final["asr_non_target"] - clean_final["asr_non_target"],
            "clean_f1_macro": clean_final["clean_f1_macro"],
            "poisoned_clean_f1_macro": poison_final["clean_f1_macro"],
            "clean_f1_delta": poison_final["clean_f1_macro"] - clean_final["clean_f1_macro"],
            "artifact_hash": row["attack"]["artifact_hash"],
        })
    return output


def topology_effects(uplifts: list[dict]) -> list[dict]:
    indexed = {}
    for row in uplifts:
        if row["arm"] != "primary":
            continue
        key = tuple(row[name] for name in ("dataset", "fusion", "epsilon", "seed"))
        indexed[(key, row["topology"])] = row
    output = []
    for (key, topology), hfl in indexed.items():
        if topology != "hfl" or (key, "vfl") not in indexed:
            continue
        vfl = indexed[(key, "vfl")]
        output.append({
            "dataset": key[0],
            "fusion": key[1],
            "epsilon": key[2],
            "seed": key[3],
            "hfl_asr_uplift": hfl["asr_uplift"],
            "vfl_asr_uplift": vfl["asr_uplift"],
            "delta_topologia": hfl["asr_uplift"] - vfl["asr_uplift"],
            "interpretation": "VFL_more_resistant"
            if hfl["asr_uplift"] > vfl["asr_uplift"] else "HFL_more_resistant",
            "same_artifact_hash": hfl["artifact_hash"] == vfl["artifact_hash"],
            "artifact_hash": hfl["artifact_hash"],
        })
    return output


def _paired_wilcoxon(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[~np.isclose(values, 0.0)]
    if not len(values):
        return {
            "method": "paired-wilcoxon-exact",
            "statistic": 0.0,
            "pvalue_two_sided": 1.0,
            "nonzero_pairs": 0,
        }
    absolute = np.abs(values)
    order = np.argsort(absolute, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and np.isclose(absolute[order[stop]], absolute[order[start]]):
            stop += 1
        ranks[order[start:stop]] = (start + 1 + stop) / 2.0
        start = stop
    rank_sum = float(ranks.sum())
    positive = float(ranks[values > 0].sum())
    statistic = min(positive, rank_sum - positive)
    enumerated = []
    for signs in range(1 << len(values)):
        candidate = sum(rank for index, rank in enumerate(ranks) if signs & (1 << index))
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


def dataset_summary(rows: list[dict], bootstrap_samples: int = 10_000) -> list[dict]:
    rng = np.random.default_rng(8)
    output = []
    for dataset in sorted({row["dataset"] for row in rows}):
        by_seed = defaultdict(list)
        for row in rows:
            if row["dataset"] == dataset:
                by_seed[row["seed"]].append(row["delta_topologia"])
        seed_means = np.asarray(
            [np.mean(by_seed[seed]) for seed in sorted(by_seed)], dtype=float
        )
        if not len(seed_means):
            continue
        mean, ci95 = _bootstrap_summary(seed_means, rng, bootstrap_samples)
        output.append({
            "dataset": dataset,
            "mean_delta_topologia": mean,
            "ci95": ci95,
            "seed_count": int(len(seed_means)),
            "paired_test": _paired_wilcoxon(seed_means),
            "supports_vfl_resistance": bool(ci95[0] > 0),
            "supports_hfl_resistance": bool(ci95[1] < 0),
            "inference_warning": (
                "Only five seed clusters: the bootstrap interval and paired test are "
                "indicative, not strong confirmatory evidence. Inspect condition_effects "
                "for reversals before interpreting this aggregate."
            ),
        })
    return output


def condition_effects(rows: list[dict], bootstrap_samples: int = 10_000) -> list[dict]:
    grouped = defaultdict(list)
    fields = ("dataset", "fusion", "epsilon")
    for row in rows:
        grouped[tuple(row[name] for name in fields)].append(row)
    rng = np.random.default_rng(9)
    output = []
    for key in sorted(grouped, key=str):
        by_seed = defaultdict(list)
        for row in grouped[key]:
            by_seed[row["seed"]].append(row["delta_topologia"])
        seed_means = np.asarray(
            [np.mean(by_seed[seed]) for seed in sorted(by_seed)], dtype=float
        )
        mean, ci95 = _bootstrap_summary(seed_means, rng, bootstrap_samples)
        output.append({
            **dict(zip(fields, key)),
            "mean_delta_topologia": mean,
            "ci95": ci95,
            "seed_count": int(len(seed_means)),
            "paired_test": _paired_wilcoxon(seed_means),
            "inference_warning": "Five seed clusters; interval is indicative and cell-specific.",
        })
    return output


def secondary_aggregator_effects(uplifts: list[dict]) -> list[dict]:
    warning = (
        "Secondary robust-aggregation arm is HFL-only and uses three seeds; report "
        "descriptively and do not use it for causal HFL-versus-VFL topology claims."
    )
    return [
        {**row, "inference_warning": warning}
        for row in uplifts if row["arm"] == "secondary"
    ]


def secondary_aggregator_summary(rows: list[dict]) -> list[dict]:
    fields = ("dataset", "fusion", "epsilon", "aggregator")
    grouped = defaultdict(list)
    for row in rows:
        grouped[tuple(row[name] for name in fields)].append(row)
    output = []
    for key in sorted(grouped, key=str):
        selected = grouped[key]
        output.append({
            **dict(zip(fields, key)),
            "mean_asr_uplift": float(np.mean([row["asr_uplift"] for row in selected])),
            "mean_clean_f1_delta": float(np.mean([row["clean_f1_delta"] for row in selected])),
            "seed_count": len({row["seed"] for row in selected}),
            "inference_warning": selected[0]["inference_warning"],
        })
    return output


def gate_report(rows: list[dict], uplifts: list[dict], effects: list[dict]) -> dict:
    seed_rows = [row for row in uplifts if row["seed"] == 42 and row["epsilon"] is None
                 and row["fusion"] == "intermediate"]
    surrogate = defaultdict(list)
    for row in rows:
        if row["config"]["poison_rate"] == 0.0:
            continue
        meta = row["attack"].get("artifact_metadata") or {}
        surrogate[row["config"]["dataset"]].append(float(meta.get("surrogate_validation_asr", 0.0)))
    artifact_ok = all(row["same_artifact_hash"] for row in effects)
    by_dataset = {}
    for dataset in sorted({row["dataset"] for row in seed_rows}):
        selected = [row for row in seed_rows if row["dataset"] == dataset]
        by_dataset[dataset] = {
            "surrogate_asr_ge_80": bool(surrogate[dataset] and max(surrogate[dataset]) >= 0.80),
            "transfer_asr_ge_50_any_topology": bool(
                selected and max(row["poisoned_asr_non_target"] for row in selected) >= 0.50
            ),
            "uplift_ge_20pp_any_topology": bool(
                selected and max(row["asr_uplift"] for row in selected) >= 0.20
            ),
            "clean_f1_drop_le_5pp_all_selected": bool(
                selected and min(row["clean_f1_delta"] for row in selected) >= -0.05
            ),
        }
    return {
        "artifact_hash_identical_in_topology_pairs": artifact_ok,
        "dataset_gates": by_dataset,
        "passed_all_observed": bool(by_dataset) and all(
            values["surrogate_asr_ge_80"]
            and values["transfer_asr_ge_50_any_topology"]
            and values["uplift_ge_20pp_any_topology"]
            and values["clean_f1_drop_le_5pp_all_selected"]
            for values in by_dataset.values()
        ) and artifact_ok,
    }


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
            f"Exp. 08 primary arm incomplete: {len(integrity['primary']['missing'])} missing jobs"
        )
    uplifts = uplift_rows(rows)
    effects = topology_effects(uplifts)
    secondary_rows = secondary_aggregator_effects(uplifts)
    report = {
        "integrity": integrity,
        "gate": gate_report(rows, uplifts, effects),
        "dataset_level_indicative": dataset_summary(effects),
        "condition_effects": condition_effects(effects),
        "uplifts": uplifts,
        "topology_effects": effects,
        "secondary_robust_aggregation": {
            "paired_rows": secondary_rows,
            "cell_summary": secondary_aggregator_summary(secondary_rows),
            "inference_warning": (
                "This HFL-only arm has three seeds and is not part of the causal "
                "HFL-versus-VFL topology comparison."
            ),
        },
    }
    output = args.results_dir / "analysis_exp08_timetrojan.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "integrity": integrity,
        "gate": report["gate"],
        "dataset_level_indicative": report["dataset_level_indicative"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
