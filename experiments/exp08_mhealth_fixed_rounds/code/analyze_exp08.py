#!/usr/bin/env python3
"""Check completeness and summarize Exp. 08 fixed-round results."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from protocol import build_protocol


def load_results(directory: Path) -> list[dict]:
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))]


def completeness(rows: list[dict]) -> dict:
    expected = {config.config_id for config in build_protocol()}
    observed = {row["config"]["config_id"] for row in rows}
    return {
        "expected": len(expected),
        "observed": len(observed & expected),
        "missing": sorted(expected - observed),
        "unexpected": sorted(observed - expected),
    }


def summarize(rows: list[dict]) -> list[dict]:
    grouped = defaultdict(list)
    for row in rows:
        config = row["config"]
        grouped[(config["topology"], config["fusion"], config["epsilon"])].append(
            row["best"]["f1_macro"]
        )
    return [
        {
            "topology": key[0],
            "fusion": key[1],
            "epsilon": key[2],
            "seed_count": len(values),
            "best_f1_mean": statistics.mean(values),
            "best_f1_stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        }
        for key, values in sorted(grouped.items())
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = {"completeness": completeness(load_results(args.results))}
    report["summary"] = summarize(load_results(args.results))
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
