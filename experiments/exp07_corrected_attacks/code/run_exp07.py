#!/usr/bin/env python3
"""Run one resumable cell of the corrected cross-dataset Exp. 07."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SHARED = ROOT / "experiments/shared/code"
sys.path.insert(0, str(SHARED))

from experiment_validity import composed_epsilon, dp_plan_from_loaders, manual_dp_fusion_step
from attacks import (
    attack_message,
    attack_update,
    corrupt_labels,
    label_poison_ids,
    malicious_indices,
)
from data import WindowSet, channel_counts, concatenate, load_dataset
from protocol import ExperimentConfig, build_protocol, smoke_protocol


DELTA = 1e-5
MAX_GRAD_NORM = 5.0
DEFAULTS = {
    "rounds": 25,
    "local_epochs": 3,
    "batch_size": 64,
    "lr": 0.01,
    "momentum": 0.9,
    "dropout": 0.1,
    "embedding_dim": 64,
    "hidden_dim": 128,
}


def _torch():
    try:
        import torch
        from torch import nn
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before running Exp. 07") from exc
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


def dataset_class():
    torch, _, _, Dataset = _torch()

    class FederatedDataset(Dataset):
        def __init__(self, windows: WindowSet):
            self.windows = windows

        def __len__(self):
            return len(self.windows)

        def __getitem__(self, index):
            return (
                {name: torch.as_tensor(values[index], dtype=torch.float32)
                 for name, values in self.windows.groups.items()},
                torch.tensor(int(self.windows.labels[index]), dtype=torch.long),
                torch.tensor(int(self.windows.example_ids[index]), dtype=torch.long),
            )

    return FederatedDataset


def make_loader(windows: WindowSet, batch_size: int, shuffle: bool):
    _, _, DataLoader, _ = _torch()
    return DataLoader(dataset_class()(windows), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0)


def to_device(values, labels, device):
    return {name: value.to(device) for name, value in values.items()}, labels.to(device)


def evaluate(model, loader, device) -> dict[str, float]:
    torch, _, _, _ = _torch()
    from sklearn.metrics import accuracy_score, f1_score

    predicted, expected = [], []
    model.eval()
    with torch.no_grad():
        for values, labels, _ in loader:
            values, labels = to_device(values, labels, device)
            predicted.extend(model(values).argmax(dim=1).cpu().tolist())
            expected.extend(labels.cpu().tolist())
    return {
        "accuracy": float(accuracy_score(expected, predicted)),
        "f1_macro": float(f1_score(expected, predicted, average="macro", zero_division=0)),
    }


def plain_local_train(model, loader, device, params, num_classes, poisoned_ids):
    torch, _, _, _ = _torch()
    optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"],
                                momentum=params["momentum"])
    losses = []
    model.train()
    for _ in range(params["local_epochs"]):
        for values, labels, example_ids in loader:
            values, labels = to_device(values, labels, device)
            if poisoned_ids:
                labels = corrupt_labels(labels, example_ids.numpy(), poisoned_ids, num_classes)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(values), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return copy.deepcopy(model.state_dict()), float(np.mean(losses))


def private_local_train(model, loader, device, params, num_classes, poisoned_ids, sigma):
    torch, _, _, _ = _torch()
    from opacus import PrivacyEngine

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"],
                                momentum=params["momentum"])
    engine = PrivacyEngine()
    private_model, optimizer, private_loader = engine.make_private(
        module=model, optimizer=optimizer, data_loader=loader,
        noise_multiplier=sigma, max_grad_norm=MAX_GRAD_NORM,
    )
    losses = []
    for _ in range(params["local_epochs"]):
        for values, labels, example_ids in private_loader:
            values, labels = to_device(values, labels, device)
            if poisoned_ids:
                labels = corrupt_labels(labels, example_ids.numpy(), poisoned_ids, num_classes)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(private_model(values), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    state = {name.removeprefix("_module."): value.detach().clone()
             for name, value in private_model.state_dict().items()}
    return state, float(np.mean(losses))


def make_model(clients, fusion, num_classes, params, device):
    from models import EightPartyFusion

    return EightPartyFusion(
        channel_counts(clients), fusion, num_classes,
        embedding_dim=params["embedding_dim"], hidden_dim=params["hidden_dim"],
        dropout=params["dropout"],
    ).to(device)


def run_hfl(config, clients, test, num_classes, params, device):
    torch, _, _, _ = _torch()
    from aggregation import aggregate, apply_update, state_update

    loaders = [make_loader(client, params["batch_size"], True) for client in clients]
    test_loader = make_loader(test, params["batch_size"], False)
    model = make_model(clients, config.fusion, num_classes, params, device)
    selected = malicious_indices(config.seed, config.attack_ratio) if not config.is_clean else ()
    poisoned = label_poison_ids(clients, selected) if config.attack == "label_flip" else frozenset()
    sigma, sample_rate, steps_per_round = dp_plan_from_loaders(
        config.epsilon or 0.0, loaders, params["batch_size"], params["rounds"],
        params["local_epochs"], DELTA, mechanisms_per_step=1,
    )
    rounds = []
    for round_index in range(params["rounds"]):
        reference = copy.deepcopy(model.state_dict())
        updates, sizes, losses = [], [], []
        for client_index, loader in enumerate(loaders):
            local = copy.deepcopy(model).to(device)
            is_parameter_attacker = client_index in selected and config.attack in {
                "sign_flip", "scaling", "free_rider"
            }
            if is_parameter_attacker and config.attack == "free_rider":
                update = {name: torch.zeros_like(value) for name, value in reference.items()}
                loss = 0.0
            else:
                train = private_local_train if config.epsilon is not None else plain_local_train
                arguments = (local, loader, device, params, num_classes, poisoned)
                state, loss = train(*arguments, sigma) if config.epsilon is not None else train(*arguments)
                update = state_update(state, reference)
                if is_parameter_attacker:
                    update = attack_update(update, config.attack)
            updates.append(update)
            sizes.append(len(loader.dataset))
            losses.append(loss)
        merged, metadata = aggregate(updates, sizes, config.aggregator, len(selected))
        model.load_state_dict(apply_update(reference, merged))
        metrics = evaluate(model, test_loader, device)
        metrics.update({
            "round": round_index + 1,
            "loss": float(np.mean(losses)),
            "epsilon_spent": composed_epsilon(
                sigma, sample_rate, (round_index + 1) * steps_per_round, DELTA
            ) if config.epsilon is not None else None,
            "aggregation": metadata,
        })
        rounds.append(metrics)
    return rounds, selected, poisoned, sigma, steps_per_round


def _wrap_vfl_modules(model, malicious, attack):
    from opacus.grad_sample import GradSampleModule

    modules = []
    for index, name in enumerate(list(model.branches)):
        if attack == "free_rider" and index in malicious:
            for parameter in model.branches[name].parameters():
                parameter.requires_grad_(False)
            continue
        model.branches[name] = GradSampleModule(model.branches[name])
        modules.append(model.branches[name])
    if any(parameter.requires_grad for parameter in model.coordinator.parameters()):
        model.coordinator = GradSampleModule(model.coordinator)
        modules.append(model.coordinator)
    return modules


def run_vfl(config, clients, test, num_classes, params, device):
    torch, _, _, _ = _torch()
    joined = concatenate(clients)
    loader = make_loader(joined, params["batch_size"], True)
    test_loader = make_loader(test, params["batch_size"], False)
    model = make_model(clients, config.fusion, num_classes, params, device)
    selected = malicious_indices(config.seed, config.attack_ratio) if not config.is_clean else ()
    poisoned = label_poison_ids(clients, selected) if config.attack == "label_flip" else frozenset()
    planned_mechanisms = 8 + int(config.fusion == "intermediate")
    sigma, sample_rate, steps_per_round = dp_plan_from_loaders(
        config.epsilon or 0.0, [loader], params["batch_size"], params["rounds"],
        params["local_epochs"], DELTA, mechanisms_per_step=planned_mechanisms,
    )
    private_modules = _wrap_vfl_modules(model, selected, config.attack) if config.epsilon is not None else []
    if config.epsilon is None and config.attack == "free_rider":
        for index, name in enumerate(model.branches):
            if index in selected:
                for parameter in model.branches[name].parameters():
                    parameter.requires_grad_(False)
    optimizer = torch.optim.SGD((parameter for parameter in model.parameters() if parameter.requires_grad),
                                lr=params["lr"], momentum=params["momentum"])
    rounds = []
    names = list(model.branches)
    for round_index in range(params["rounds"]):
        losses = []
        model.train()
        for _ in range(params["local_epochs"]):
            for values, labels, example_ids in loader:
                values, labels = to_device(values, labels, device)
                if poisoned:
                    labels = corrupt_labels(labels, example_ids.numpy(), poisoned, num_classes)
                optimizer.zero_grad()
                messages = model.messages(values)
                if config.attack in {"sign_flip", "scaling"}:
                    for index in selected:
                        messages[names[index]] = attack_message(messages[names[index]], config.attack)
                logits = model.fuse(messages)
                loss = torch.nn.functional.cross_entropy(logits, labels)
                loss.backward()
                if config.epsilon is not None:
                    for module in private_modules:
                        manual_dp_fusion_step(module, MAX_GRAD_NORM, sigma)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        [parameter for parameter in model.parameters() if parameter.requires_grad],
                        MAX_GRAD_NORM,
                    )
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        metrics = evaluate(model, test_loader, device)
        metrics.update({
            "round": round_index + 1,
            "loss": float(np.mean(losses)),
            "epsilon_spent": composed_epsilon(
                sigma, sample_rate, (round_index + 1) * steps_per_round, DELTA
            ) if config.epsilon is not None else None,
            "mechanisms_composed": planned_mechanisms,
        })
        rounds.append(metrics)
    return rounds, selected, poisoned, sigma, steps_per_round


def run_config(config: ExperimentConfig, args) -> dict:
    torch, _, _, _ = _torch()
    set_seed(config.seed)
    data_root = args.mhealth_root if config.dataset == "mhealth" else args.opportunity_root
    clients, test, num_classes = load_dataset(config.dataset, data_root)
    if len(clients) != config.participant_count:
        raise RuntimeError(f"{config.dataset} produced {len(clients)} HFL clients, expected 8")
    params = DEFAULTS | {
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
    }
    device = torch.device(args.device)
    started = time.time()
    runner = run_hfl if config.topology == "hfl" else run_vfl
    rounds, selected, poisoned, sigma, steps_per_round = runner(
        config, clients, test, num_classes, params, device
    )
    return {
        "schema_version": 2,
        "experiment": "exp07_corrected_cross_dataset_attacks",
        "config": config.to_dict(),
        "dataset": {"name": config.dataset, "num_classes": num_classes},
        "privacy": {
            "target_epsilon": config.epsilon,
            "delta": DELTA if config.epsilon is not None else None,
            "max_grad_norm": MAX_GRAD_NORM if config.epsilon is not None else None,
            "noise_multiplier": sigma if config.epsilon is not None else None,
            "composed_steps_per_round": steps_per_round if config.epsilon is not None else None,
            "accounting": "global-topology-aware-rdp" if config.epsilon is not None else "no-dp-control",
        },
        "attack_assignment": {
            "participant_indices": list(selected),
            "poisoned_example_count": len(poisoned),
        },
        "hyperparameters": params,
        "round_metrics": rounds,
        "final": rounds[-1],
        "duration_seconds": time.time() - started,
        "device": str(device),
    }


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-index", type=int)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    parser.add_argument("--rounds", type=int, default=DEFAULTS["rounds"])
    parser.add_argument("--local-epochs", type=int, default=DEFAULTS["local_epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--mhealth-root", type=Path, default=ROOT / "data/MHEALTHDATASET")
    parser.add_argument("--opportunity-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configs = smoke_protocol() if args.smoke_only else build_protocol()
    if args.list:
        print(json.dumps([config.to_dict() for config in configs], indent=2))
        return 0
    if args.config_index is None or not 0 <= args.config_index < len(configs):
        raise SystemExit(f"--config-index must be in [0, {len(configs) - 1}]")
    if args.smoke_only:
        args.rounds = 1
        args.local_epochs = 1
    config = configs[args.config_index]
    output = args.output_dir / f"{config.config_id}.json"
    if output.exists():
        print(f">>> Already complete: {output}")
        return 0
    print(f">>> Exp. 07 {args.config_index}/{len(configs) - 1}: {config.config_id}")
    atomic_json(output, run_config(config, args))
    print(f">>> Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
