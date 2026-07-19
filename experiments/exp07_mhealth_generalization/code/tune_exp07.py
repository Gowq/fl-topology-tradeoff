#!/usr/bin/env python3
"""Partitioned HFL/VFL tuning on subject 9; subject 10 is never loaded."""

from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from protocol import ExperimentConfig
from run_exp07 import (
    DEFAULT_HYPERPARAMETERS,
    ROOT,
    _torch,
    atomic_json,
    load_splits,
    run_hfl,
    run_vfl,
    set_seed,
)


TUNING_SEEDS = (42, 123, 456)
LEARNING_RATES = (0.001, 0.003, 0.01)
DROPOUTS = (0.1, 0.25)
HIDDEN_DIMS = (64, 128)


def tuning_jobs() -> list[dict]:
    jobs = []
    for topology, lr, dropout, hidden_dim, seed in itertools.product(
        ("hfl", "vfl"),
        LEARNING_RATES,
        DROPOUTS,
        HIDDEN_DIMS,
        TUNING_SEEDS,
    ):
        job_id = (
            f"{topology}__lr{str(lr).replace('.', 'p')}__"
            f"drop{str(dropout).replace('.', 'p')}__h{hidden_dim}__s{seed}"
        )
        jobs.append(
            {
                "job_id": job_id,
                "topology": topology,
                "lr": lr,
                "dropout": dropout,
                "hidden_dim": hidden_dim,
                "seed": seed,
            }
        )
    return jobs


def run_job(args, job: dict) -> dict:
    torch, _, _, _ = _torch()
    train, validation, _ = load_splits(
        args.data_root, args.window_size, args.stride, include_test=False
    )
    set_seed(job["seed"])
    config = ExperimentConfig("tuning", job["topology"], 0.0, job["seed"])
    params = DEFAULT_HYPERPARAMETERS | {
        "lr": job["lr"],
        "dropout": job["dropout"],
        "hidden_dim": job["hidden_dim"],
        "rounds": 1 if args.smoke_only else args.rounds,
    }
    runner = run_hfl if job["topology"] == "hfl" else run_vfl
    _, round_metrics, _, _ = runner(
        config,
        train,
        validation,
        validation,
        params,
        torch.device(args.device),
    )
    return {
        "schema_version": 1,
        **job,
        "rounds": params["rounds"],
        "selection_metric": "final validation F1-macro on subject 9",
        "test_subject_used": False,
        "final_f1_macro": round_metrics[-1]["f1_macro"],
    }


def aggregate_parts(parts_dir: Path, output: Path, allow_partial: bool = False) -> dict:
    jobs = tuning_jobs()
    expected = {job["job_id"] for job in jobs}
    rows = []
    for path in sorted(parts_dir.glob("tune__*.json")):
        rows.append(json.loads(path.read_text(encoding="utf-8")))
    observed = {row["job_id"] for row in rows}
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if (missing or unexpected) and not allow_partial:
        raise RuntimeError(
            f"Incomplete tuning parts: missing={len(missing)} unexpected={len(unexpected)}"
        )

    grouped = defaultdict(list)
    for row in rows:
        key = (row["topology"], row["lr"], row["dropout"], row["hidden_dim"])
        grouped[key].append(row)
    candidates = []
    for (topology, lr, dropout, hidden_dim), members in sorted(grouped.items()):
        seed_scores = {str(row["seed"]): row["final_f1_macro"] for row in members}
        candidates.append(
            {
                "topology": topology,
                "lr": lr,
                "dropout": dropout,
                "hidden_dim": hidden_dim,
                "seed_scores": seed_scores,
                "seed_count": len(seed_scores),
                "mean_f1_macro": float(np.mean(list(seed_scores.values()))),
            }
        )

    selected = {}
    for topology in ("hfl", "vfl"):
        eligible = [
            row
            for row in candidates
            if row["topology"] == topology
            and (allow_partial or row["seed_count"] == len(TUNING_SEEDS))
        ]
        if not eligible:
            raise RuntimeError(f"No complete tuning candidate for {topology}")
        best = max(
            eligible,
            key=lambda row: (
                row["mean_f1_macro"],
                -row["hidden_dim"],
                -row["dropout"],
            ),
        )
        selected[topology] = {
            "lr": best["lr"],
            "dropout": best["dropout"],
            "hidden_dim": best["hidden_dim"],
        }

    payload = {
        "schema_version": 2,
        "selection_metric": "mean final validation F1-macro on subject 9",
        "test_subject_used": False,
        "completeness": {
            "expected": len(expected),
            "observed": len(observed),
            "missing": missing,
            "unexpected": unexpected,
        },
        "selected": selected,
        "candidates": candidates,
    }
    atomic_json(output, payload)
    return payload


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/MHEALTHDATASET")
    parser.add_argument(
        "--parts-dir",
        type=Path,
        default=Path(__file__).parents[1] / "results/tuning_parts",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[1] / "results/tuned_hyperparameters.json",
    )
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--aggregate", action="store_true")
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    jobs = tuning_jobs()
    if args.list:
        print(json.dumps(jobs, indent=2))
        return 0
    if args.aggregate:
        payload = aggregate_parts(args.parts_dir, args.output, args.allow_partial)
        print(json.dumps(payload["selected"], indent=2))
        return 0
    if args.job_index is None or not 0 <= args.job_index < len(jobs):
        raise SystemExit(f"--job-index must be in [0, {len(jobs) - 1}], or use --aggregate")
    job = jobs[args.job_index]
    output_dir = args.parts_dir / ("smoke" if args.smoke_only else "")
    output = output_dir / f"tune__{job['job_id']}.json"
    if output.exists():
        print(f">>> Already complete: {output}")
        return 0
    print(f">>> Tuning job {args.job_index}/{len(jobs) - 1}: {job['job_id']}")
    atomic_json(output, run_job(args, job))
    print(f">>> Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
