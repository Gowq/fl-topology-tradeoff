#!/usr/bin/env python3
"""Check completeness and aggregate utility/ASR for Exp. 09."""

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
        key = (
            config["topology"], config["epsilon"], config["attack"],
            config["attack_ratio"], config["aggregator"],
        )
        grouped[key].append((row["final"]["f1_macro"], row["final"]["backdoor_asr"]))
    return [
        {
            "topology": key[0], "epsilon": key[1], "attack": key[2],
            "attack_ratio": key[3], "aggregator": key[4],
            "seed_count": len(values),
            "f1_mean": statistics.mean(value[0] for value in values),
            "asr_mean": statistics.mean(value[1] for value in values),
        }
        for key, values in sorted(grouped.items())
    ]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = load_results(args.results)
    rendered = json.dumps(
        {"completeness": completeness(rows), "summary": summarize(rows)}, indent=2
    ) + "\n"
    if args.output:
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
