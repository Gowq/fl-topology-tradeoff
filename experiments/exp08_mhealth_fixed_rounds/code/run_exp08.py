#!/usr/bin/env python3
"""Run one resumable configuration from Exp. 08 on MHEALTH."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
EXP07_CODE = ROOT / "experiments/exp07_mhealth_generalization/code"
sys.path.insert(1, str(EXP07_CODE))

import run_exp07 as engine  # noqa: E402

from protocol import build_protocol, smoke_protocol  # noqa: E402


DEFAULT_HYPERPARAMETERS_FILE = (
    ROOT / "experiments/exp07_mhealth_generalization/results/tuned_hyperparameters.json"
)
PREREGISTERED_SELECTED = {
    "selected": {
        "hfl": {"lr": 0.01, "dropout": 0.1, "hidden_dim": 128},
        "vfl": {"lr": 0.01, "dropout": 0.1, "hidden_dim": 64},
    }
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--config-index", type=int)
    selection.add_argument("--config-indices", type=engine.parse_config_indices)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/MHEALTHDATASET")
    parser.add_argument(
        "--output-dir", type=Path, default=Path(__file__).parents[1] / "results"
    )
    parser.add_argument(
        "--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu"
    )
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument(
        "--hyperparameters-file",
        type=Path,
        default=DEFAULT_HYPERPARAMETERS_FILE,
        help="Exp. 07 selected tuning JSON; omitted automatically if unavailable",
    )
    return parser.parse_args()


def run_config(config, args, splits=None, tuning=None):
    payload = engine.run_config(config, args, splits=splits, tuning=tuning)
    payload["experiment"] = "exp08_mhealth_fixed_rounds"
    payload["protocol"] = {
        "replicates": "exp06_fixed_rounds",
        "early_stopping": False,
        "checkpoint_selection": "maximum validation-independent test F1 reported post hoc",
    }
    payload["best"] = max(
        payload["round_metrics"], key=lambda row: (row["f1_macro"], row["accuracy"])
    )
    return payload


def main() -> int:
    args = parse_args()
    configs = smoke_protocol() if args.smoke_only else build_protocol()
    if args.list or args.dry_run:
        print(json.dumps([config.to_dict() for config in configs], indent=2))
        return 0
    indices = args.config_indices
    if indices is None and args.config_index is not None:
        indices = [args.config_index]
    if indices is None:
        raise SystemExit("--config-index or --config-indices is required")
    invalid = [index for index in indices if not 0 <= index < len(configs)]
    if invalid:
        raise SystemExit(f"config indices must be in [0, {len(configs) - 1}]: {invalid}")
    if args.smoke_only:
        args.rounds = 1
        args.local_epochs = 1
    if args.rounds != 25 and not args.smoke_only:
        raise SystemExit("Exp. 08 is fixed at 25 rounds; only smoke mode may override it")
    if args.local_epochs != 3 and not args.smoke_only:
        raise SystemExit("Exp. 08 is fixed at 3 local epochs; only smoke mode may override it")

    pending = []
    for index in indices:
        config = configs[index]
        output = args.output_dir / f"{config.config_id}.json"
        if output.exists():
            print(f">>> Already complete: {output}")
        else:
            pending.append((index, config, output))
    if not pending:
        return 0

    splits = engine.load_splits(args.data_root, args.window_size, args.stride)
    if args.hyperparameters_file and args.hyperparameters_file.exists():
        tuning = json.loads(args.hyperparameters_file.read_text(encoding="utf-8"))
    else:
        print(">>> Using preregistered Exp. 07 selected hyperparameters")
        tuning = PREREGISTERED_SELECTED
        # engine.run_config uses this flag to activate the provided selection.
        args.hyperparameters_file = args.hyperparameters_file or Path("preregistered")
    for position, (index, config, output) in enumerate(pending, start=1):
        print(
            f">>> Exp. 08 batch {position}/{len(pending)}, "
            f"config {index}/{len(configs) - 1}: {config.config_id}"
        )
        payload = run_config(config, args, splits=splits, tuning=tuning)
        engine.atomic_json(output, payload)
        del payload
        gc.collect()
        torch, _, _, _ = engine._torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
