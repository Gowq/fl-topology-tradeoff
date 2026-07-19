#!/usr/bin/env python3
"""Tune HFL and VFL separately on MHEALTH subject 9, never subject 10."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import numpy as np

from protocol import ExperimentConfig
from run_exp07 import (
    DEFAULT_HYPERPARAMETERS,
    ROOT,
    atomic_json,
    load_splits,
    run_hfl,
    run_vfl,
    set_seed,
    _torch,
)


TUNING_SEEDS = (42, 123, 456)
LEARNING_RATES = (0.001, 0.003, 0.01)
DROPOUTS = (0.1, 0.25)
HIDDEN_DIMS = (64, 128)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/MHEALTHDATASET")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[1] / "results/tuned_hyperparameters.json",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    torch, _, _, _ = _torch()
    train, validation, _ = load_splits(
        args.data_root, args.window_size, args.stride, include_test=False
    )
    device = torch.device(args.device)
    candidates = list(itertools.product(LEARNING_RATES, DROPOUTS, HIDDEN_DIMS))
    seeds = TUNING_SEEDS
    if args.smoke_only:
        candidates = candidates[:1]
        seeds = seeds[:1]
        args.rounds = 1

    scores = []
    for topology in ("hfl", "vfl"):
        runner = run_hfl if topology == "hfl" else run_vfl
        for lr, dropout, hidden_dim in candidates:
            seed_scores = []
            for seed in seeds:
                set_seed(seed)
                config = ExperimentConfig("tuning", topology, 0.0, seed)
                params = DEFAULT_HYPERPARAMETERS | {
                    "lr": lr,
                    "dropout": dropout,
                    "hidden_dim": hidden_dim,
                    "rounds": args.rounds,
                }
                # Validation subject 9 is the evaluation set during tuning.
                _, rounds, _, _ = runner(
                    config, train, validation, validation, params, device
                )
                seed_scores.append(rounds[-1]["f1_macro"])
            scores.append(
                {
                    "topology": topology,
                    "lr": lr,
                    "dropout": dropout,
                    "hidden_dim": hidden_dim,
                    "seeds": list(seeds),
                    "f1_macro_by_seed": seed_scores,
                    "mean_f1_macro": float(np.mean(seed_scores)),
                }
            )

    selected = {}
    for topology in ("hfl", "vfl"):
        best = max(
            (row for row in scores if row["topology"] == topology),
            key=lambda row: (row["mean_f1_macro"], -row["hidden_dim"], -row["dropout"]),
        )
        selected[topology] = {
            "lr": best["lr"],
            "dropout": best["dropout"],
            "hidden_dim": best["hidden_dim"],
        }
    atomic_json(
        args.output,
        {
            "schema_version": 1,
            "selection_metric": "final validation F1-macro on subject 9",
            "test_subject_used": False,
            "selected": selected,
            "candidates": scores,
        },
    )
    print(json.dumps(selected, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
