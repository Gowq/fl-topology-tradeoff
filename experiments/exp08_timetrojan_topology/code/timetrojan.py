"""TimeTrojan-FGSM artifact generation and reuse for Exp. 08."""

from __future__ import annotations

import hashlib
import json
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from data import WindowSet, concatenate, stack_groups, unstack_groups


K_POSITIONS = 5
ETA = 2.0
ITERATIONS = 10
SURROGATE_EPOCHS = 10
SURROGATE_BATCH_SIZE = 256
SURROGATE_LR = 0.003


@dataclass(frozen=True)
class Artifact:
    metadata: dict
    poisoned_train: WindowSet
    triggered_test: WindowSet
    triggered_validation: WindowSet
    artifact_hash: str
    poisoned_parent_ids: np.ndarray


def _torch():
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    return torch, DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    torch, _, _ = _torch()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _loader(values: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool):
    torch, DataLoader, TensorDataset = _torch()
    dataset = TensorDataset(
        torch.as_tensor(values, dtype=torch.float32),
        torch.as_tensor(labels, dtype=torch.long),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def _train_epoch(model, loader, device, lr: float) -> float:
    torch, _, _ = _torch()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses = []
    model.train()
    for values, labels in loader:
        values, labels = values.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(model(values), labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else 0.0


def _predict(model, values: np.ndarray, batch_size: int, device) -> np.ndarray:
    torch, _, _ = _torch()
    predictions = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(values), batch_size):
            batch = torch.as_tensor(values[start:start + batch_size], dtype=torch.float32).to(device)
            predictions.extend(model(batch).argmax(dim=1).cpu().numpy().tolist())
    return np.asarray(predictions, dtype=np.int64)


def _fgsm_trigger(model, values: np.ndarray, target_class: int, k: int, eta: float, device) -> np.ndarray:
    torch, _, _ = _torch()
    source = torch.as_tensor(values, dtype=torch.float32, device=device)
    output = source.clone().detach().requires_grad_(True)
    target = torch.full((len(values),), target_class, dtype=torch.long, device=device)
    target_loss = torch.nn.functional.cross_entropy(model(output), target)
    target_grad = torch.autograd.grad(target_loss, output, retain_graph=True)[0]

    other_grad = torch.zeros_like(output)
    num_classes = model.classifier.out_features
    for label in range(num_classes):
        if label == target_class:
            continue
        model.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(
            model(output), torch.full((len(values),), label, dtype=torch.long, device=device)
        )
        other_grad = other_grad + torch.autograd.grad(loss, output, retain_graph=True)[0]

    saliency = torch.where(
        (target_grad >= 0) & (other_grad <= 0),
        target_grad * torch.abs(other_grad),
        torch.zeros_like(target_grad),
    ).sum(dim=1)
    fallback = torch.abs(target_grad).sum(dim=1)
    empty = saliency.sum(dim=1) <= 0
    saliency[empty] = fallback[empty]
    topk = min(k, saliency.shape[1])
    indices = torch.topk(saliency, topk, dim=1).indices
    mask = torch.zeros_like(saliency)
    mask.scatter_(1, indices, 1.0)
    perturbed = output.detach() - eta * mask[:, None, :] * torch.sign(target_grad.detach())
    return perturbed.cpu().numpy().astype(np.float32)


def _select_poison_ids(train: WindowSet, target_class: int, poison_rate: float, seed: int) -> np.ndarray:
    candidates = np.flatnonzero(train.labels != target_class)
    count = max(1, int(round(len(train) * poison_rate)))
    count = min(count, len(candidates))
    rng = np.random.default_rng(seed + 80_008)
    return np.sort(rng.choice(candidates, size=count, replace=False))


def _make_triggered_windowset(template: WindowSet, values: np.ndarray, target_class: int,
                              example_ids: np.ndarray | None = None) -> WindowSet:
    return WindowSet(
        unstack_groups(values, template),
        np.full(len(values), target_class, dtype=np.int64),
        template.example_ids.copy() if example_ids is None else example_ids.astype(np.int64),
        template.owners.copy(),
    )


def _non_target_triggered(
    model, windows: WindowSet, target_class: int, device, batch_size: int
) -> WindowSet:
    values, _ = stack_groups(windows)
    mask = windows.labels != target_class
    triggered = _fgsm_trigger(model, values[mask], target_class, K_POSITIONS, ETA, device)
    selected = windows.take(np.flatnonzero(mask))
    return _make_triggered_windowset(selected, triggered, target_class)


def generate_artifact(
    *,
    dataset: str,
    seed: int,
    poison_rate: float,
    target_class: int,
    clients: list[WindowSet],
    validation: WindowSet,
    test: WindowSet,
    num_classes: int,
    output_path: Path,
    device: str,
) -> dict:
    torch, _, _ = _torch()
    from models import CentralSurrogate

    set_seed(seed)
    started = time.time()
    train = concatenate(clients)
    train_values, group_names = stack_groups(train)
    poison_indices = _select_poison_ids(train, target_class, poison_rate, seed)
    model = CentralSurrogate(train_values.shape[1], num_classes).to(torch.device(device))
    clean_loader = _loader(train_values, train.labels, SURROGATE_BATCH_SIZE, True)
    losses = []
    for _ in range(SURROGATE_EPOCHS):
        losses.append(_train_epoch(model, clean_loader, torch.device(device), SURROGATE_LR))

    poison_values = train_values[poison_indices]
    poison_labels = np.full(len(poison_indices), target_class, dtype=np.int64)
    for _ in range(ITERATIONS):
        poison_values = _fgsm_trigger(model, train_values[poison_indices], target_class,
                                      K_POSITIONS, ETA, torch.device(device))
        augmented_values = np.concatenate([train_values, poison_values])
        augmented_labels = np.concatenate([train.labels, poison_labels])
        loader = _loader(augmented_values, augmented_labels, SURROGATE_BATCH_SIZE, True)
        losses.append(_train_epoch(model, loader, torch.device(device), SURROGATE_LR))

    synthetic_ids = -(seed * 10**7 + np.arange(1, len(poison_indices) + 1, dtype=np.int64))
    poisoned_train = _make_triggered_windowset(
        train.take(poison_indices), poison_values, target_class, synthetic_ids
    )
    triggered_validation = _non_target_triggered(
        model, validation, target_class, torch.device(device), SURROGATE_BATCH_SIZE
    )
    triggered_test = _non_target_triggered(
        model, test, target_class, torch.device(device), SURROGATE_BATCH_SIZE
    )
    validation_predictions = _predict(
        model, stack_groups(triggered_validation)[0], SURROGATE_BATCH_SIZE, torch.device(device)
    )
    validation_asr = float(np.mean(validation_predictions == target_class)) if len(validation_predictions) else 0.0
    metadata = {
        "schema_version": 1,
        "attack": "TimeTrojan-FGSM-transfer",
        "dataset": dataset,
        "seed": seed,
        "target_class": target_class,
        "poison_rate": poison_rate,
        "dirty_label": True,
        "poisoned_train_count": int(len(poisoned_train)),
        "poisoned_parent_ids": [int(value) for value in train.example_ids[poison_indices]],
        "triggered_validation_count": int(len(triggered_validation)),
        "triggered_test_count": int(len(triggered_test)),
        "k_positions": K_POSITIONS,
        "eta": ETA,
        "iterations": ITERATIONS,
        "surrogate_epochs_clean": SURROGATE_EPOCHS,
        "surrogate_lr": SURROGATE_LR,
        "surrogate_validation_asr": validation_asr,
        "group_names": group_names,
        "elapsed_seconds": time.time() - started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(".npz.tmp")
    arrays = {"metadata": np.asarray(json.dumps(metadata))}
    for prefix, windows in (
        ("poisoned_train", poisoned_train),
        ("triggered_test", triggered_test),
        ("triggered_validation", triggered_validation),
    ):
        arrays[f"{prefix}__labels"] = windows.labels
        arrays[f"{prefix}__example_ids"] = windows.example_ids
        arrays[f"{prefix}__owners"] = windows.owners
        for name, values in windows.groups.items():
            arrays[f"{prefix}__group__{name}"] = values.astype(np.float32)
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, output_path)
    digest = sha256_file(output_path)
    metadata["artifact_hash"] = digest
    output_path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_artifact(path: Path, template: WindowSet) -> Artifact:
    if not path.exists():
        raise FileNotFoundError(f"missing TimeTrojan artifact: {path}")
    payload = np.load(path, allow_pickle=True)
    metadata = json.loads(str(payload["metadata"].item()))

    def read(prefix: str) -> WindowSet:
        return WindowSet(
            {name: payload[f"{prefix}__group__{name}"].astype(np.float32) for name in template.groups},
            payload[f"{prefix}__labels"].astype(np.int64),
            payload[f"{prefix}__example_ids"].astype(np.int64),
            payload[f"{prefix}__owners"].astype(np.int64),
        )

    return Artifact(metadata, read("poisoned_train"), read("triggered_test"),
                    read("triggered_validation"), sha256_file(path),
                    np.asarray(metadata["poisoned_parent_ids"], dtype=np.int64))


def append_poison(clients: list[WindowSet], artifact: Artifact) -> list[WindowSet]:
    by_parent = {
        int(parent_id): index for index, parent_id in enumerate(artifact.poisoned_parent_ids)
    }
    output = []
    for client in clients:
        indices = [by_parent[int(example_id)] for example_id in client.example_ids
                   if int(example_id) in by_parent]
        if not indices:
            output.append(client)
            continue
        poison_part = artifact.poisoned_train.take(indices)
        output.append(concatenate([client, poison_part]))
    return output
