#!/usr/bin/env python3
"""Run one resumable Exp. 09 attack/defense configuration on OPPORTUNITY."""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
EXP07_CODE = ROOT / "experiments/exp07_mhealth_generalization/code"
SHARED = ROOT / "experiments/shared/code"
sys.path.insert(1, str(EXP07_CODE))
sys.path.insert(1, str(SHARED))

import run_exp07 as engine  # noqa: E402
from experiment_validity import composed_epsilon, dp_plan_from_loaders, manual_dp_fusion_step, select_malicious_indices  # noqa: E402
from opportunity_data import (  # noqa: E402
    BACKDOOR_TARGET,
    ROOT_RUNS,
    ROOT_SUBJECT,
    TEST_RUNS,
    TEST_SUBJECT,
    TRAIN_RUNS,
    TRAIN_SUBJECTS,
    WindowSet,
    apply_sensor_trigger,
    concatenate,
    horizontal_clients,
    load_splits,
    trigger_scale,
)
from protocol import build_protocol, smoke_protocol  # noqa: E402


DELTA = 1e-5
MAX_GRAD_NORM = 5.0
DEFAULTS = {"lr": 0.01, "dropout": 0.5, "batch_size": 16, "local_epochs": 1, "rounds": 25}


def set_seed(seed: int) -> None:
    torch, _, _, _ = engine._torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dataset_class():
    torch, _, _, Dataset = engine._torch()

    class OpportunityDataset(Dataset):
        def __init__(
            self,
            windows: WindowSet,
            scale: np.ndarray,
            poison: bool = False,
            poisoned_subjects: set[int] | None = None,
            trigger_all_non_target: bool = False,
        ):
            self.windows = windows
            self.scale = scale
            self.poison = poison
            self.poisoned_subjects = poisoned_subjects or set()
            self.trigger_all_non_target = trigger_all_non_target

        def __len__(self):
            return len(self.windows)

        def __getitem__(self, index):
            label = int(self.windows.labels[index])
            subject = int(self.windows.subjects[index])
            values = {
                "body_sensors": self.windows.body[index],
                "object_sensors": self.windows.objects[index],
                "ambient_sensors": self.windows.ambient[index],
            }
            poison_train = (
                label == BACKDOOR_TARGET
                and (self.poison or subject in self.poisoned_subjects)
            )
            poison_test = self.trigger_all_non_target and label != BACKDOOR_TARGET
            if poison_train or poison_test:
                values = apply_sensor_trigger(values, self.scale)
            return (
                {name: torch.as_tensor(value, dtype=torch.float32) for name, value in values.items()},
                torch.tensor(label, dtype=torch.long),
            )

    return OpportunityDataset


def make_loader(windows, scale, batch_size, shuffle, **kwargs):
    _, _, DataLoader, _ = engine._torch()
    return DataLoader(
        dataset_class()(windows, scale, **kwargs), batch_size=batch_size,
        shuffle=shuffle, num_workers=0,
    )


def evaluate_backdoor(model, loader, device) -> float:
    torch, _, _, _ = engine._torch()
    model.eval()
    hits = total = 0
    with torch.no_grad():
        for values, target in loader:
            values, target = engine.to_device(values, target, device)
            predicted = model(values).argmax(dim=1)
            eligible = target != BACKDOOR_TARGET
            hits += int(((predicted == BACKDOOR_TARGET) & eligible).sum())
            total += int(eligible.sum())
    return float(hits / total) if total else 0.0


def build_model(topology, params, device):
    from models import OpportunityIntermediateModel

    return OpportunityIntermediateModel(
        topology=topology, dropout=params["dropout"]
    ).to(device)


def run_hfl(config, train_subjects, root_set, test_set, scale, params, device):
    from robust_aggregation import (
        apply_update,
        fedavg,
        fltrust,
        foolsgold,
        model_replacement,
        state_update,
    )

    clients = horizontal_clients(train_subjects)
    malicious = (
        select_malicious_indices(len(clients), config.attack_ratio, config.seed)
        if config.attack != "none" else set()
    )
    loaders = [
        make_loader(
            client, scale, params["batch_size"], True,
            poison=(config.attack == "sensor_backdoor" and index in malicious),
        )
        for index, client in enumerate(clients)
    ]
    clean_test = make_loader(test_set, scale, params["batch_size"], False)
    triggered_test = make_loader(
        test_set, scale, params["batch_size"], False, trigger_all_non_target=True
    )
    trusted_loader = make_loader(root_set, scale, params["batch_size"], True)
    model = build_model("hfl", params, device)
    use_dp = config.epsilon > 0
    sigma, sample_rate, steps_per_round = dp_plan_from_loaders(
        config.epsilon, loaders, params["batch_size"], params["rounds"],
        params["local_epochs"], DELTA,
    )
    history = None
    round_metrics = []
    for round_index in range(params["rounds"]):
        reference = copy.deepcopy(model.state_dict())
        updates, sizes, losses = [], [], []
        for index, loader in enumerate(loaders):
            local = copy.deepcopy(model).to(device)
            if use_dp:
                state, loss = engine.private_train(
                    local, loader, device, params["lr"], params["local_epochs"], sigma
                )
            else:
                loss = engine.plain_train(
                    local, loader, device, params["lr"], params["local_epochs"]
                )
                state = local.state_dict()
            update = state_update(state, reference)
            if config.attack == "model_replacement" and index in malicious:
                update = model_replacement(update, len(clients), len(malicious))
            updates.append(update)
            sizes.append(len(loader.dataset))
            losses.append(loss)

        weights = None
        if config.aggregator == "fedavg":
            aggregate = fedavg(updates, sizes)
        elif config.aggregator == "fltrust":
            root_update = engine.trusted_root_update(
                model, trusted_loader, device, params["lr"]
            )
            aggregate = fltrust(updates, root_update)
        elif config.aggregator == "foolsgold":
            aggregate, history, fg_weights = foolsgold(updates, history)
            weights = fg_weights.tolist()
        else:
            raise ValueError(f"Unsupported HFL aggregator: {config.aggregator}")
        model.load_state_dict(apply_update(reference, aggregate))
        metrics = engine.evaluate(model, clean_test, device)
        metrics.update({
            "round": round_index + 1,
            "loss": float(np.mean(losses)),
            "epsilon_spent": composed_epsilon(
                sigma, sample_rate, (round_index + 1) * steps_per_round, DELTA
            ) if use_dp else 0.0,
            "foolsgold_weights": weights,
        })
        round_metrics.append(metrics)
    return round_metrics, evaluate_backdoor(model, triggered_test, device), sorted(malicious)


def run_vfl(config, train_subjects, root_set, test_set, scale, params, device):
    del root_set
    torch, nn, _, _ = engine._torch()
    malicious_indices = select_malicious_indices(
        len(train_subjects), config.attack_ratio, config.seed
    )
    malicious_subjects = {TRAIN_SUBJECTS[index] for index in malicious_indices}
    train_set = concatenate(train_subjects)
    loader = make_loader(
        train_set, scale, params["batch_size"], True,
        poisoned_subjects=malicious_subjects,
    )
    clean_test = make_loader(test_set, scale, params["batch_size"], False)
    triggered_test = make_loader(
        test_set, scale, params["batch_size"], False, trigger_all_non_target=True
    )
    model = build_model("vfl", params, device)
    use_dp = config.epsilon > 0
    mechanisms = len(model.encoders) + 1
    sigma, sample_rate, steps_per_round = dp_plan_from_loaders(
        config.epsilon, [loader], params["batch_size"], params["rounds"],
        params["local_epochs"], DELTA, mechanisms_per_step=mechanisms,
    )
    private_modules = engine.make_vfl_private(model, sigma) if use_dp else []
    optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"], momentum=0.9)
    round_metrics = []
    for round_index in range(params["rounds"]):
        losses = []
        model.train()
        for _ in range(params["local_epochs"]):
            for values, target in loader:
                values, target = engine.to_device(values, target, device)
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(model(values), target)
                loss.backward()
                if use_dp:
                    for module in private_modules:
                        manual_dp_fusion_step(module, MAX_GRAD_NORM, sigma)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        metrics = engine.evaluate(model, clean_test, device)
        metrics.update({
            "round": round_index + 1,
            "loss": float(np.mean(losses)),
            "epsilon_spent": composed_epsilon(
                sigma, sample_rate, (round_index + 1) * steps_per_round, DELTA
            ) if use_dp else 0.0,
        })
        round_metrics.append(metrics)
    return round_metrics, evaluate_backdoor(model, triggered_test, device), sorted(malicious_subjects)


def run_config(config, args, splits=None):
    torch, _, _, _ = engine._torch()
    set_seed(config.seed)
    params = DEFAULTS | {
        "lr": args.lr, "dropout": args.dropout, "batch_size": args.batch_size,
        "local_epochs": args.local_epochs, "rounds": args.rounds,
    }
    if splits is None:
        splits = load_splits(args.data_root)
    train, root_set, test_set = splits
    scale = trigger_scale(train)
    device = torch.device(args.device)
    started = time.time()
    runner = run_hfl if config.topology == "hfl" else run_vfl
    rounds, asr, malicious = runner(
        config, train, root_set, test_set, scale, params, device
    )
    return {
        "schema_version": 1,
        "experiment": "exp09_opportunity_attack_defense",
        "config": config.to_dict(),
        "hyperparameters": params,
        "dataset": {
            "name": "OPPORTUNITY",
            "train": {"subjects": list(TRAIN_SUBJECTS), "runs": list(TRAIN_RUNS)},
            "trusted_root": {"subject": ROOT_SUBJECT, "runs": list(ROOT_RUNS)},
            "test": {"subject": TEST_SUBJECT, "runs": list(TEST_RUNS)},
            "window_size": 30, "stride": 15,
        },
        "backdoor": {
            "target_class": BACKDOOR_TARGET,
            "clean_label": True,
            "location": "body_lower channels 27:30, final 20% of window",
            "amplitude": "4x training-channel standard deviation",
        },
        "malicious_participants": malicious,
        "round_metrics": rounds,
        "final": {**rounds[-1], "backdoor_asr": asr},
        "duration_seconds": time.time() - started,
        "device": str(device),
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--config-index", type=int)
    selection.add_argument("--config-indices", type=engine.parse_config_indices)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parents[1] / "results")
    parser.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    parser.add_argument("--rounds", type=int, default=25)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--dropout", type=float, default=0.5)
    return parser.parse_args()


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
        args.rounds = args.local_epochs = 1

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
    splits = load_splits(args.data_root)
    for position, (index, config, output) in enumerate(pending, start=1):
        print(
            f">>> Exp. 09 batch {position}/{len(pending)}, "
            f"config {index}/{len(configs) - 1}: {config.config_id}"
        )
        payload = run_config(config, args, splits=splits)
        engine.atomic_json(output, payload)
        del payload
        gc.collect()
        torch, _, _, _ = engine._torch()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
