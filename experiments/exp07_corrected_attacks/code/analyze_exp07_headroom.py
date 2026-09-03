#!/usr/bin/env python3
"""Exploratory re-analysis of Exp. 07: is the cross-dataset magnitude gap a
dataset effect or floor compression?

The pre-registered endpoint D = F1_attacked - F1_clean is an absolute F1
difference, so it is bounded by how much utility the clean control holds above
the random-prediction floor. That headroom differs by an order of magnitude
between the two datasets, which makes the absolute topology effects
non-comparable across them. This script recomputes the same paired effect on a
headroom-normalised scale.

NOT pre-registered. Reported as exploratory.

Inference matches analyze_exp07.py: per-seed means first, then a bootstrap over
the five seed clusters (10,000 resamples, rng seed 7).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "results"
PARAMETER_ATTACKS = {"sign_flip", "scaling", "free_rider"}
# random-prediction floor = 1 / num_classes
FLOOR = {"opportunity": 1 / 18, "mhealth": 1 / 12}
BOOTSTRAP = 10_000


def load():
    clean, attacked = {}, []
    for path in RESULTS.glob("exp07v3__primary__*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = payload["config"]
        key = (cfg["dataset"], cfg["topology"], cfg["fusion"], cfg["epsilon"], cfg["seed"])
        if cfg["attack"] == "none":
            clean[key] = payload["final"]["f1_macro"]
        else:
            attacked.append((cfg, payload["final"]["f1_macro"]))
    return clean, attacked


def seed_clustered(values_by_seed: dict[int, list[float]], rng) -> tuple[float, list[float], int]:
    seed_means = np.array([np.mean(v) for v in values_by_seed.values()])
    boot = rng.choice(seed_means, size=(BOOTSTRAP, len(seed_means)), replace=True).mean(axis=1)
    return float(seed_means.mean()), [float(np.percentile(boot, 2.5)),
                                      float(np.percentile(boot, 97.5))], len(seed_means)


def main() -> None:
    clean, attacked = load()

    headroom = defaultdict(list)
    for (ds, topo, _f, _e, _s), f1 in clean.items():
        headroom[(ds, topo)].append(f1 - FLOOR[ds])
    at_or_below = {
        ds: sum(1 for (d, *_), f1 in clean.items() if d == ds and f1 <= FLOOR[ds])
        for ds in FLOOR
    }

    pairs = defaultdict(dict)
    for cfg, f1 in attacked:
        if cfg["attack"] not in PARAMETER_ATTACKS:
            continue
        key = (cfg["dataset"], cfg["topology"], cfg["fusion"], cfg["epsilon"], cfg["seed"])
        if key not in clean:
            continue
        cell = (cfg["dataset"], cfg["fusion"], cfg["epsilon"],
                cfg["attack"], cfg["attack_ratio"], cfg["seed"])
        pairs[cell][cfg["topology"]] = (f1 - clean[key], clean[key] - FLOOR[cfg["dataset"]])

    out = {"headroom": {}, "clean_cells_at_or_below_floor": at_or_below, "effects": {}}
    for key, vals in sorted(headroom.items()):
        out["headroom"]["/".join(key)] = {"mean": float(np.mean(vals)), "max": float(max(vals))}

    def collect(dataset, scale, min_headroom=None, no_dp_only=False):
        by_seed = defaultdict(list)
        for cell, arms in pairs.items():
            if cell[0] != dataset or "hfl" not in arms or "vfl" not in arms:
                continue
            if no_dp_only and cell[2] is not None:
                continue
            (d_h, h_h), (d_v, h_v) = arms["hfl"], arms["vfl"]
            if min_headroom is not None and not (h_h > min_headroom and h_v > min_headroom):
                continue
            by_seed[cell[5]].append(d_v - d_h if scale == "absolute" else d_v / h_v - d_h / h_h)
        return by_seed

    specs = [
        ("absolute_preregistered", "absolute", None, False),
        ("normalised_headroom_gt_0.05", "normalised", 0.05, False),
        ("normalised_no_dp_cells", "normalised", None, True),
        ("absolute_no_dp_cells", "absolute", None, True),
    ]
    for label, scale, thr, no_dp in specs:
        out["effects"][label] = {}
        for ds in ("mhealth", "opportunity"):
            by_seed = collect(ds, scale, thr, no_dp)
            if not by_seed:
                continue
            rng = np.random.default_rng(7)
            mean, ci, clusters = seed_clustered(by_seed, rng)
            out["effects"][label][ds] = {
                "mean": mean, "ci95": ci, "seed_clusters": clusters,
                "cells": sum(len(v) for v in by_seed.values()),
            }

    dest = RESULTS / "analysis_exp07_headroom_exploratory.json"
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
