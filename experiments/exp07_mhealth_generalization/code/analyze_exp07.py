#!/usr/bin/env python3
"""Validate Exp. 07 completeness and summarize five-seed cells."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from protocol import build_protocol


GROUP_FIELDS = (
    "arm",
    "topology",
    "epsilon",
    "client_count",
    "attack",
    "attack_ratio",
    "aggregator",
)
T_CRITICAL_95 = {2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776}


def load_results(results_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(results_dir.glob("*.json")):
        if path.name == "tuned_hyperparameters.json":
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("experiment") != "exp07_mhealth_generalization":
            continue
        rows.append(payload)
    return rows


def completeness(results: list[dict]) -> dict:
    expected = {config.config_id for config in build_protocol()}
    observed = {row["config"]["config_id"] for row in results}
    return {
        "expected": len(expected),
        "observed": len(observed),
        "missing": sorted(expected - observed),
        "unexpected": sorted(observed - expected),
    }


def summarize(results: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for result in results:
        config = result["config"]
        key = tuple(config[field] for field in GROUP_FIELDS)
        groups[key].append(result)

    summary = []
    for key, members in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        f1 = [float(member["final"]["f1_macro"]) for member in members]
        asr = [float(member["final"]["backdoor_asr"]) for member in members]
        mean = sum(f1) / len(f1)
        variance = sum((value - mean) ** 2 for value in f1) / max(1, len(f1) - 1)
        standard_error = math.sqrt(variance / len(f1))
        critical = T_CRITICAL_95.get(len(f1), 1.96)
        summary.append(
            {
                **dict(zip(GROUP_FIELDS, key, strict=True)),
                "seed_count": len(members),
                "f1_macro_mean": mean,
                "f1_macro_std": math.sqrt(variance),
                "f1_macro_ci95_low": mean - critical * standard_error,
                "f1_macro_ci95_high": mean + critical * standard_error,
                "backdoor_asr_mean": sum(asr) / len(asr),
            }
        )
    return summary


def tail_shape_flags(summary: list[dict]) -> list[dict]:
    """Flag utility decreases as epsilon relaxes; interpretation still needs CIs."""

    by_topology = defaultdict(list)
    for row in summary:
        if row["arm"] == "tail":
            by_topology[row["topology"]].append(row)
    flags = []
    for topology, rows in by_topology.items():
        rows.sort(key=lambda row: row["epsilon"])
        for previous, current in zip(rows, rows[1:]):
            if current["f1_macro_mean"] < previous["f1_macro_mean"]:
                flags.append(
                    {
                        "topology": topology,
                        "epsilon_from": previous["epsilon"],
                        "epsilon_to": current["epsilon"],
                        "kind": "f1_decreases_as_epsilon_increases",
                        "f1_delta": current["f1_macro_mean"] - previous["f1_macro_mean"],
                        "confidence_intervals_overlap": not (
                            current["f1_macro_ci95_low"] > previous["f1_macro_ci95_high"]
                        ),
                    }
                )
    return flags


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).parents[1] / "results")
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    results = load_results(args.results_dir)
    coverage = completeness(results)
    summary = summarize(results)
    write_csv(args.results_dir / "summary.csv", summary)
    report = {"completeness": coverage, "tail_shape_flags": tail_shape_flags(summary)}
    (args.results_dir / "analysis.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    incomplete = bool(coverage["missing"] or coverage["unexpected"])
    return 1 if incomplete and not args.allow_partial else 0


if __name__ == "__main__":
    raise SystemExit(main())
