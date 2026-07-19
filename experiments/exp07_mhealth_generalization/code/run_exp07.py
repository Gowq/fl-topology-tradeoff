#!/usr/bin/env python3
"""Run one resumable configuration from the Exp. 07 MHEALTH protocol."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SHARED = ROOT / "experiments/shared/code"
sys.path.insert(0, str(SHARED))

from experiment_validity import (  # noqa: E402
    composed_epsilon,
    dp_plan_from_loaders,
    manual_dp_fusion_step,
    select_malicious_indices,
)
from mhealth_data import (  # noqa: E402
    MODALITY_COLUMNS,
    TEST_SUBJECTS,
    TRAIN_SUBJECTS,
    VALIDATION_SUBJECTS,
    WindowedSubject,
    apply_sensor_trigger,
    fit_normalization,
    load_subject,
    normalize_subject,
    select_client_subjects,
)
from protocol import ExperimentConfig, build_protocol, smoke_protocol  # noqa: E402


DEFAULT_HYPERPARAMETERS = {
    "lr": 0.003,
    "dropout": 0.25,
    "hidden_dim": 128,
    "batch_size": 64,
    "local_epochs": 1,
    "rounds": 25,
}
DELTA = 1e-5
MAX_GRAD_NORM = 5.0
BACKDOOR_TARGET = 0


def _torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError(
            "Exp. 07 requires the repository dependencies. Run `pip install -r requirements.txt`."
        ) from exc
    return torch, nn, DataLoader, Dataset


def set_seed(seed: int) -> None:
    torch, _, _, _ = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_splits(data_root: Path, window_size: int, stride: int, include_test: bool = True):
    train = [load_subject(data_root, subject, window_size, stride) for subject in TRAIN_SUBJECTS]
    validation = [load_subject(data_root, subject, window_size, stride) for subject in VALIDATION_SUBJECTS]
    test = (
        [load_subject(data_root, subject, window_size, stride) for subject in TEST_SUBJECTS]
        if include_test
        else []
    )
    stats = fit_normalization(train)
    return tuple(
        [normalize_subject(subject, stats) for subject in split]
        for split in (train, validation, test)
    )


def dataset_class():
    torch, _, _, Dataset = _torch()

    class MhealthDataset(Dataset):
        def __init__(
            self,
            subjects: Sequence[WindowedSubject],
            poisoned_subjects: Iterable[int] = (),
            trigger_all_non_target: bool = False,
        ):
            self.subjects = list(subjects)
            self.rows = [
                (subject_index, row)
                for subject_index, subject in enumerate(self.subjects)
                for row in range(len(subject))
            ]
            self.poisoned_subjects = set(poisoned_subjects)
            self.trigger_all_non_target = trigger_all_non_target

        def __len__(self):
            return len(self.rows)

        def __getitem__(self, index):
            subject_index, row = self.rows[index]
            subject = self.subjects[subject_index]
            label = int(subject.labels[row])
            values = {name: data[row] for name, data in subject.modalities.items()}
            poison_train = subject.subject_id in self.poisoned_subjects and label == BACKDOOR_TARGET
            poison_test = self.trigger_all_non_target and label != BACKDOOR_TARGET
            if poison_train or poison_test:
                values = apply_sensor_trigger(values)
            return (
                {name: torch.as_tensor(value, dtype=torch.float32) for name, value in values.items()},
                torch.tensor(label, dtype=torch.long),
            )

    return MhealthDataset


def make_loader(subjects, batch_size, shuffle, **dataset_kwargs):
    _, _, DataLoader, _ = _torch()
    return DataLoader(
        dataset_class()(subjects, **dataset_kwargs),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def to_device(values, target, device):
    return {name: tensor.to(device) for name, tensor in values.items()}, target.to(device)


def evaluate(model, loader, device) -> dict[str, float]:
    torch, _, _, _ = _torch()
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    predicted, expected = [], []
    with torch.no_grad():
        for values, target in loader:
            values, target = to_device(values, target, device)
            predicted.extend(model(values).argmax(dim=1).cpu().tolist())
            expected.extend(target.cpu().tolist())
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "f1_macro": float(f1_score(expected, predicted, average="macro", zero_division=0)),
    }


def evaluate_backdoor(model, loader, device) -> float:
    torch, _, _, _ = _torch()
    model.eval()
    hits = total = 0
    with torch.no_grad():
        for values, target in loader:
            values, target = to_device(values, target, device)
            predictions = model(values).argmax(dim=1)
            eligible = target != BACKDOOR_TARGET
            hits += int(((predictions == BACKDOOR_TARGET) & eligible).sum())
            total += int(eligible.sum())
    return float(hits / total) if total else 0.0


def plain_train(model, loader, device, lr, epochs):
    torch, nn, _, _ = _torch()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    losses = []
    model.train()
    for _ in range(epochs):
        for values, target in loader:
            values, target = to_device(values, target, device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(values), target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def private_train(model, loader, device, lr, epochs, noise_multiplier):
    torch, nn, _, _ = _torch()
    from opacus import PrivacyEngine

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    engine = PrivacyEngine()
    model, optimizer, private_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=MAX_GRAD_NORM,
    )
    losses = []
    for _ in range(epochs):
        for values, target in private_loader:
            values, target = to_device(values, target, device)
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(values), target)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    state = {
        name.removeprefix("_module."): value.detach().clone()
        for name, value in model.state_dict().items()
    }
    return state, float(np.mean(losses)) if losses else 0.0


def trusted_root_update(global_model, trusted_loader, device, lr):
    from robust_aggregation import state_update

    trusted = copy.deepcopy(global_model).to(device)
    plain_train(trusted, trusted_loader, device, lr, epochs=1)
    return state_update(trusted.state_dict(), global_model.state_dict())


def malicious_subject_ids(subjects, ratio, seed) -> set[int]:
    indices = select_malicious_indices(len(subjects), ratio, seed)
    return {subjects[index].subject_id for index in indices}


def run_hfl(config, train_subjects, validation_subjects, test_subjects, params, device):
    torch, _, _, _ = _torch()
    from models import MultimodalClassifier
    from robust_aggregation import (
        apply_update,
        fedavg,
        fltrust,
        foolsgold,
        model_replacement,
        state_update,
    )

    clients = select_client_subjects(train_subjects, config.client_count, config.seed)
    malicious = malicious_subject_ids(clients, config.attack_ratio, config.seed) if config.attack != "none" else set()
    loaders = [
        make_loader(
            [subject],
            params["batch_size"],
            True,
            poisoned_subjects=malicious if config.attack == "sensor_backdoor" else (),
        )
        for subject in clients
    ]
    clean_test = make_loader(test_subjects, params["batch_size"], False)
    triggered_test = make_loader(
        test_subjects, params["batch_size"], False, trigger_all_non_target=True
    )
    trusted_loader = make_loader(validation_subjects, params["batch_size"], True)
    model = MultimodalClassifier(
        topology="hfl",
        hidden_dim=params["hidden_dim"], dropout=params["dropout"]
    ).to(device)
    use_dp = config.epsilon > 0
    sigma, sample_rate, steps_per_round = dp_plan_from_loaders(
        config.epsilon,
        loaders,
        params["batch_size"],
        params["rounds"],
        params["local_epochs"],
        DELTA,
    )
    history = None
    round_metrics = []
    for round_index in range(params["rounds"]):
        updates, sizes, losses = [], [], []
        reference = copy.deepcopy(model.state_dict())
        for subject, loader in zip(clients, loaders, strict=True):
            local = copy.deepcopy(model).to(device)
            if use_dp:
                state, loss = private_train(
                    local, loader, device, params["lr"], params["local_epochs"], sigma
                )
            else:
                loss = plain_train(local, loader, device, params["lr"], params["local_epochs"])
                state = local.state_dict()
            update = state_update(state, reference)
            if config.attack == "model_replacement" and subject.subject_id in malicious:
                update = model_replacement(update, len(clients), len(malicious))
            updates.append(update)
            sizes.append(len(loader.dataset))
            losses.append(loss)

        if config.aggregator == "fedavg":
            aggregate = fedavg(updates, sizes)
            weights = None
        elif config.aggregator == "fltrust":
            root = trusted_root_update(model, trusted_loader, device, params["lr"])
            aggregate = fltrust(updates, root)
            weights = None
        elif config.aggregator == "foolsgold":
            aggregate, history, fg_weights = foolsgold(updates, history)
            weights = fg_weights.tolist()
        else:
            raise ValueError(f"Unsupported HFL aggregator: {config.aggregator}")
        model.load_state_dict(apply_update(reference, aggregate))
        metrics = evaluate(model, clean_test, device)
        metrics.update(
            {
                "round": round_index + 1,
                "loss": float(np.mean(losses)),
                "epsilon_spent": composed_epsilon(
                    sigma, sample_rate, (round_index + 1) * steps_per_round, DELTA
                ) if use_dp else 0.0,
                "foolsgold_weights": weights,
            }
        )
        round_metrics.append(metrics)
    return model, round_metrics, evaluate_backdoor(model, triggered_test, device), sorted(malicious)


def make_vfl_private(model, noise_multiplier):
    from opacus.grad_sample import GradSampleModule

    for name in list(model.encoders):
        model.encoders[name] = GradSampleModule(model.encoders[name])
    model.coordinator = GradSampleModule(model.coordinator)
    return [*model.encoders.values(), model.coordinator]


def run_vfl(config, train_subjects, validation_subjects, test_subjects, params, device):
    torch, nn, _, _ = _torch()
    from models import MultimodalClassifier

    del validation_subjects  # VFL has no FLTrust root dataset.
    malicious = malicious_subject_ids(train_subjects, config.attack_ratio, config.seed) if config.attack != "none" else set()
    loader = make_loader(
        train_subjects,
        params["batch_size"],
        True,
        poisoned_subjects=malicious if config.attack == "sensor_backdoor" else (),
    )
    clean_test = make_loader(test_subjects, params["batch_size"], False)
    triggered_test = make_loader(
        test_subjects, params["batch_size"], False, trigger_all_non_target=True
    )
    model = MultimodalClassifier(
        topology=config.topology,
        hidden_dim=params["hidden_dim"], dropout=params["dropout"]
    ).to(device)
    use_dp = config.epsilon > 0
    mechanism_count = len(model.encoders) + 1
    sigma, sample_rate, steps_per_round = dp_plan_from_loaders(
        config.epsilon,
        [loader],
        params["batch_size"],
        params["rounds"],
        params["local_epochs"],
        DELTA,
        mechanisms_per_step=mechanism_count,
    )
    private_modules = make_vfl_private(model, sigma) if use_dp else []
    optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"], momentum=0.9)
    round_metrics = []
    for round_index in range(params["rounds"]):
        losses = []
        model.train()
        for _ in range(params["local_epochs"]):
            for values, target in loader:
                values, target = to_device(values, target, device)
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
        metrics = evaluate(model, clean_test, device)
        metrics.update(
            {
                "round": round_index + 1,
                "loss": float(np.mean(losses)),
                "epsilon_spent": composed_epsilon(
                    sigma, sample_rate, (round_index + 1) * steps_per_round, DELTA
                ) if use_dp else 0.0,
            }
        )
        round_metrics.append(metrics)
    return model, round_metrics, evaluate_backdoor(model, triggered_test, device), sorted(malicious)


def run_config(config, args):
    torch, _, _, _ = _torch()
    set_seed(config.seed)
    selected = {}
    if args.hyperparameters_file:
        tuning = json.loads(args.hyperparameters_file.read_text(encoding="utf-8"))
        tuning_topology = "hfl" if config.topology == "hfl" else "vfl"
        selected = tuning["selected"][tuning_topology]
    explicit = {
        name: value
        for name, value in {
            "lr": args.lr,
            "dropout": args.dropout,
            "hidden_dim": args.hidden_dim,
        }.items()
        if value is not None
    }
    params = DEFAULT_HYPERPARAMETERS | selected | explicit | {
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
    }
    train, validation, test = load_splits(args.data_root, args.window_size, args.stride)
    device = torch.device(args.device)
    started = time.time()
    runner = run_hfl if config.topology == "hfl" else run_vfl
    _, rounds, backdoor_asr, malicious = runner(
        config, train, validation, test, params, device
    )
    return {
        "schema_version": 1,
        "experiment": "exp07_mhealth_generalization",
        "config": config.to_dict(),
        "hyperparameters": params,
        "dataset": {
            "name": "MHEALTH",
            "train_subjects": list(TRAIN_SUBJECTS),
            "validation_subjects": list(VALIDATION_SUBJECTS),
            "test_subjects": list(TEST_SUBJECTS),
            "window_size": args.window_size,
            "stride": args.stride,
            "modalities": list(MODALITY_COLUMNS),
        },
        "malicious_subjects": malicious,
        "round_metrics": rounds,
        "final": {**rounds[-1], "backdoor_asr": backdoor_asr},
        "duration_seconds": time.time() - started,
        "device": str(device),
    }


def atomic_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-index", type=int)
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/MHEALTHDATASET")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parents[1] / "results")
    parser.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    parser.add_argument("--rounds", type=int, default=DEFAULT_HYPERPARAMETERS["rounds"])
    parser.add_argument("--local-epochs", type=int, default=DEFAULT_HYPERPARAMETERS["local_epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULT_HYPERPARAMETERS["batch_size"])
    parser.add_argument("--lr", type=float)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument(
        "--hyperparameters-file",
        type=Path,
        help="JSON produced by tune_exp07.py; CLI values override selected values",
    )
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = smoke_protocol() if args.smoke_only else build_protocol()
    if args.list or args.dry_run:
        print(json.dumps([config.to_dict() for config in configs], indent=2))
        return 0
    if args.config_index is None:
        raise SystemExit("--config-index is required unless --list/--dry-run is used")
    if not 0 <= args.config_index < len(configs):
        raise SystemExit(f"--config-index must be in [0, {len(configs) - 1}]")
    if args.smoke_only:
        args.rounds = 1
        args.local_epochs = 1
    config = configs[args.config_index]
    output = args.output_dir / f"{config.config_id}.json"
    if output.exists():
        print(f">>> Already complete: {output}")
        return 0
    print(f">>> Exp. 07 config {args.config_index}/{len(configs) - 1}: {config.config_id}")
    atomic_json(output, run_config(config, args))
    print(f">>> Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
