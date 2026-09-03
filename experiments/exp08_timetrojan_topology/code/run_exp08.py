#!/usr/bin/env python3
"""Run one resumable cell of Exp. 08 TimeTrojan topology comparison."""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[3]
SHARED = ROOT / "experiments/shared/code"
sys.path.insert(0, str(SHARED))

from experiment_validity import composed_epsilon, dp_plan_from_loaders, manual_dp_fusion_step
from data import WindowSet, channel_counts, concatenate, load_dataset
from protocol import ArtifactSpec, ExperimentConfig, artifact_specs, build_protocol, smoke_protocol
from timetrojan import append_poison, generate_artifact, load_artifact


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
        from torch.utils.data import DataLoader, Dataset
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before running Exp. 08") from exc
    return torch, DataLoader, Dataset


def set_seed(seed: int) -> None:
    torch, _, _ = _torch()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def dataset_class():
    torch, _, Dataset = _torch()

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
    _, DataLoader, _ = _torch()
    return DataLoader(dataset_class()(windows), batch_size=batch_size, shuffle=shuffle,
                      num_workers=0)


def to_device(values, labels, device):
    return {name: value.to(device) for name, value in values.items()}, labels.to(device)


def evaluate(model, loader, device) -> dict[str, float]:
    torch, _, _ = _torch()
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


def evaluate_round(model, clean_loader, trigger_loader, target_class: int, device) -> dict[str, float]:
    clean = evaluate(model, clean_loader, device)
    trigger = evaluate(model, trigger_loader, device)
    return {
        "clean_accuracy": clean["accuracy"],
        "clean_f1_macro": clean["f1_macro"],
        "asr_non_target": trigger["accuracy"],
        "trigger_f1_macro": trigger["f1_macro"],
        "target_class": target_class,
    }


def plain_local_train(model, loader, device, params):
    torch, _, _ = _torch()
    optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"], momentum=params["momentum"])
    losses = []
    model.train()
    for _ in range(params["local_epochs"]):
        for values, labels, _ in loader:
            values, labels = to_device(values, labels, device)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(model(values), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    return copy.deepcopy(model.state_dict()), float(np.mean(losses)) if losses else 0.0


def private_local_train(model, loader, device, params, sigma):
    torch, _, _ = _torch()
    from opacus import PrivacyEngine

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"], momentum=params["momentum"])
    engine = PrivacyEngine()
    private_model, optimizer, private_loader = engine.make_private(
        module=model, optimizer=optimizer, data_loader=loader,
        noise_multiplier=sigma, max_grad_norm=MAX_GRAD_NORM,
    )
    losses = []
    for _ in range(params["local_epochs"]):
        for values, labels, _ in private_loader:
            values, labels = to_device(values, labels, device)
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(private_model(values), labels)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
    state = {name.removeprefix("_module."): value.detach().clone()
             for name, value in private_model.state_dict().items()}
    return state, float(np.mean(losses)) if losses else 0.0


def make_model(clients, fusion, num_classes, params, device):
    from models import EightPartyFusion

    return EightPartyFusion(
        channel_counts(clients), fusion, num_classes,
        embedding_dim=params["embedding_dim"], hidden_dim=params["hidden_dim"],
        dropout=params["dropout"],
    ).to(device)


def _artifact_path(args, config: ExperimentConfig) -> Path:
    spec = ArtifactSpec(config.dataset, config.seed, target_class=config.target_class)
    return args.artifact_dir / f"{spec.artifact_id}.npz"


def _prepare_clients(config, clients, test, args):
    artifact = load_artifact(_artifact_path(args, config), test)
    if artifact.metadata["target_class"] != config.target_class:
        raise RuntimeError("TimeTrojan artifact target class does not match config")
    if config.is_clean:
        return clients, artifact.triggered_test, artifact
    if artifact.metadata["poison_rate"] != config.poison_rate:
        raise RuntimeError("TimeTrojan artifact poison rate does not match config")
    return append_poison(clients, artifact), artifact.triggered_test, artifact


def _trusted_root_update(model, trusted_root, params, device) -> dict:
    from aggregation import state_update

    loader = make_loader(trusted_root, params["batch_size"], True)
    reference = copy.deepcopy(model.state_dict())
    local = copy.deepcopy(model).to(device)
    state, _ = plain_local_train(local, loader, device, params | {"local_epochs": 1})
    return state_update(state, reference)


def run_hfl(config, bundle, params, device, args):
    torch, _, _ = _torch()
    from aggregation import FoolsGoldState, aggregate, apply_update, state_update

    clients, triggered_test, artifact = _prepare_clients(config, bundle.clients, bundle.test, args)
    loaders = [make_loader(client, params["batch_size"], True) for client in clients]
    clean_loader = make_loader(bundle.test, params["batch_size"], False)
    trigger_loader = make_loader(triggered_test, params["batch_size"], False)
    model = make_model(clients, config.fusion, bundle.num_classes, params, device)
    sigma, sample_rate, steps_per_round = dp_plan_from_loaders(
        config.epsilon or 0.0, loaders, params["batch_size"], params["rounds"],
        params["local_epochs"], DELTA, mechanisms_per_step=1,
    )
    foolsgold_state = FoolsGoldState(len(loaders)) if config.aggregator == "foolsgold" else None
    rounds = []
    for round_index in range(params["rounds"]):
        reference = copy.deepcopy(model.state_dict())
        updates, sizes, losses = [], [], []
        for loader in loaders:
            local = copy.deepcopy(model).to(device)
            train = private_local_train if config.epsilon is not None else plain_local_train
            state, loss = (
                train(local, loader, device, params, sigma)
                if config.epsilon is not None else train(local, loader, device, params)
            )
            updates.append(state_update(state, reference))
            sizes.append(len(loader.dataset))
            losses.append(loss)
        root_update = (
            _trusted_root_update(model, bundle.trusted_root, params, device)
            if config.aggregator == "fltrust" else None
        )
        merged, aggregation_meta = aggregate(
            updates, sizes, config.aggregator,
            root_update=root_update, foolsgold_state=foolsgold_state,
        )
        model.load_state_dict(apply_update(reference, merged))
        metrics = evaluate_round(model, clean_loader, trigger_loader, config.target_class, device)
        metrics.update({
            "round": round_index + 1,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "epsilon_spent": composed_epsilon(
                sigma, sample_rate, (round_index + 1) * steps_per_round, DELTA
            ) if config.epsilon is not None else None,
            "aggregation": aggregation_meta,
        })
        rounds.append(metrics)
    return rounds, artifact, sigma, steps_per_round


def _wrap_vfl_modules(model):
    from opacus.grad_sample import GradSampleModule

    modules = []
    for name in list(model.branches):
        model.branches[name] = GradSampleModule(model.branches[name])
        modules.append(model.branches[name])
    if any(parameter.requires_grad for parameter in model.coordinator.parameters()):
        model.coordinator = GradSampleModule(model.coordinator)
        modules.append(model.coordinator)
    return modules


def run_vfl(config, bundle, params, device, args):
    torch, _, _ = _torch()
    clients, triggered_test, artifact = _prepare_clients(config, bundle.clients, bundle.test, args)
    joined = concatenate(clients)
    loader = make_loader(joined, params["batch_size"], True)
    clean_loader = make_loader(bundle.test, params["batch_size"], False)
    trigger_loader = make_loader(triggered_test, params["batch_size"], False)
    model = make_model(clients, config.fusion, bundle.num_classes, params, device)
    planned_mechanisms = 8 + int(config.fusion == "intermediate")
    sigma, sample_rate, steps_per_round = dp_plan_from_loaders(
        config.epsilon or 0.0, [loader], params["batch_size"], params["rounds"],
        params["local_epochs"], DELTA, mechanisms_per_step=planned_mechanisms,
    )
    private_modules = _wrap_vfl_modules(model) if config.epsilon is not None else []
    optimizer = torch.optim.SGD(model.parameters(), lr=params["lr"], momentum=params["momentum"])
    rounds = []
    for round_index in range(params["rounds"]):
        losses = []
        model.train()
        for _ in range(params["local_epochs"]):
            for values, labels, _ in loader:
                values, labels = to_device(values, labels, device)
                optimizer.zero_grad()
                loss = torch.nn.functional.cross_entropy(model(values), labels)
                loss.backward()
                if config.epsilon is not None:
                    for module in private_modules:
                        manual_dp_fusion_step(module, MAX_GRAD_NORM, sigma)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
                optimizer.step()
                losses.append(float(loss.detach().cpu()))
        metrics = evaluate_round(model, clean_loader, trigger_loader, config.target_class, device)
        metrics.update({
            "round": round_index + 1,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "epsilon_spent": composed_epsilon(
                sigma, sample_rate, (round_index + 1) * steps_per_round, DELTA
            ) if config.epsilon is not None else None,
            "mechanisms_composed": planned_mechanisms,
        })
        rounds.append(metrics)
    return rounds, artifact, sigma, steps_per_round


def run_config(config: ExperimentConfig, args) -> dict:
    torch, _, _ = _torch()
    set_seed(config.seed)
    data_root = args.mhealth_root if config.dataset == "mhealth" else args.opportunity_root
    bundle = load_dataset(config.dataset, data_root)
    if len(bundle.clients) != config.participant_count:
        raise RuntimeError(f"{config.dataset} produced {len(bundle.clients)} clients, expected 8")
    params = DEFAULTS | {
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
    }
    device = torch.device(args.device)
    started = time.time()
    runner = run_hfl if config.topology == "hfl" else run_vfl
    rounds, artifact, sigma, steps_per_round = runner(config, bundle, params, device, args)
    return {
        "schema_version": 1,
        "experiment": "exp08_timetrojan_topology",
        "config": config.to_dict(),
        "dataset": {
            "name": config.dataset,
            "num_classes": bundle.num_classes,
            "train_clients": len(bundle.clients),
            "test_examples": len(bundle.test),
            "validation_examples": len(bundle.validation),
            "trusted_root_examples": len(bundle.trusted_root),
            "split": bundle.split,
        },
        "privacy": {
            "target_epsilon": config.epsilon,
            "delta": DELTA if config.epsilon is not None else None,
            "max_grad_norm": MAX_GRAD_NORM if config.epsilon is not None else None,
            "noise_multiplier": sigma if config.epsilon is not None else None,
            "composed_steps_per_round": steps_per_round if config.epsilon is not None else None,
            "accounting": "global-topology-aware-rdp" if config.epsilon is not None else "no-dp-control",
        },
        "attack": {
            "enabled": not config.is_clean,
            "model": "upstream-TimeTrojan-FGSM-transfer" if not config.is_clean else "clean-control",
            "artifact_hash": artifact.artifact_hash if artifact is not None else None,
            "artifact_metadata": artifact.metadata if artifact is not None else None,
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


def prepare_artifact(spec: ArtifactSpec, args) -> dict:
    set_seed(spec.seed)
    data_root = args.mhealth_root if spec.dataset == "mhealth" else args.opportunity_root
    bundle = load_dataset(spec.dataset, data_root)
    path = args.artifact_dir / f"{spec.artifact_id}.npz"
    if path.exists() and not args.force:
        meta_path = path.with_suffix(".json")
        if meta_path.exists():
            return json.loads(meta_path.read_text(encoding="utf-8"))
        return {"artifact": str(path), "status": "exists"}
    return generate_artifact(
        dataset=spec.dataset,
        seed=spec.seed,
        poison_rate=spec.poison_rate,
        target_class=spec.target_class,
        clients=bundle.clients,
        validation=bundle.validation,
        test=bundle.test,
        num_classes=bundle.num_classes,
        output_path=path,
        device=args.device,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-index", type=int)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--list-artifacts", action="store_true")
    parser.add_argument("--prepare-artifact", action="store_true")
    parser.add_argument("--prepare-all-artifacts", action="store_true")
    parser.add_argument("--artifact-index", type=int)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--device", default="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else "cpu")
    parser.add_argument("--rounds", type=int, default=DEFAULTS["rounds"])
    parser.add_argument("--local-epochs", type=int, default=DEFAULTS["local_epochs"])
    parser.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--mhealth-root", type=Path, default=ROOT / "data/MHEALTHDATASET")
    parser.add_argument("--opportunity-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "results")
    parser.add_argument("--artifact-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "artifacts")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    artifacts = artifact_specs()
    configs = smoke_protocol() if args.smoke_only else build_protocol()
    if args.list:
        print(json.dumps([config.to_dict() for config in configs], indent=2))
        return 0
    if args.list_artifacts:
        print(json.dumps([asdict(spec) | {"artifact_id": spec.artifact_id} for spec in artifacts], indent=2))
        return 0
    if args.prepare_all_artifacts:
        for index, spec in enumerate(artifacts):
            print(f">>> Preparing artifact {index}/{len(artifacts) - 1}: {spec.artifact_id}")
            print(json.dumps(prepare_artifact(spec, args), indent=2))
        return 0
    if args.prepare_artifact:
        if args.artifact_index is None or not 0 <= args.artifact_index < len(artifacts):
            raise SystemExit(f"--artifact-index must be in [0, {len(artifacts) - 1}]")
        print(json.dumps(prepare_artifact(artifacts[args.artifact_index], args), indent=2))
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
    print(f">>> Exp. 08 {args.config_index}/{len(configs) - 1}: {config.config_id}")
    atomic_json(output, run_config(config, args))
    print(f">>> Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
