"""Leakage-safe Exp. 08 dataset views with validation and FLTrust root splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


MHEALTH_GROUPS = {
    "chest_acc": ("chest", slice(0, 3)),
    "chest_ecg": ("chest", slice(3, 5)),
    "arm_acc": ("right_arm", slice(0, 3)),
    "arm_gyro": ("right_arm", slice(3, 6)),
    "arm_mag": ("right_arm", slice(6, 9)),
    "ankle_acc": ("left_ankle", slice(0, 3)),
    "ankle_gyro": ("left_ankle", slice(3, 6)),
    "ankle_mag": ("left_ankle", slice(6, 9)),
}
OPPORTUNITY_GROUPS = {
    "upper_a": ("body", slice(0, 8)),
    "upper_b": ("body", slice(8, 15)),
    "lower_a": ("body", slice(15, 23)),
    "lower_b": ("body", slice(23, 30)),
    "objects_a": ("objects", slice(0, 8)),
    "objects_b": ("objects", slice(8, 15)),
    "ambient_a": ("ambient", slice(0, 5)),
    "ambient_b": ("ambient", slice(5, 10)),
}


@dataclass(frozen=True)
class WindowSet:
    groups: dict[str, np.ndarray]
    labels: np.ndarray
    example_ids: np.ndarray
    owners: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)

    def take(self, indices: Sequence[int]) -> "WindowSet":
        indices = np.asarray(indices, dtype=int)
        return WindowSet(
            {name: values[indices] for name, values in self.groups.items()},
            self.labels[indices].copy(),
            self.example_ids[indices].copy(),
            self.owners[indices].copy(),
        )


@dataclass(frozen=True)
class DatasetBundle:
    clients: list[WindowSet]
    test: WindowSet
    validation: WindowSet
    trusted_root: WindowSet
    num_classes: int
    split: dict


MHEALTH_SPLIT = {
    "holdout": "cross-subject",
    "train_subjects": tuple(range(1, 9)),
    "validation_subjects": (9,),
    "trusted_root_subjects": (9,),
    "test_subjects": (10,),
}
OPPORTUNITY_SPLIT = {
    "holdout": "cross-subject",
    "train_subjects": (1, 3, 4),
    "validation_subjects": (3,),
    "trusted_root_subjects": (1,),
    "test_subjects": (2,),
    "train_runs": ("Drill", "ADL1", "ADL2"),
    "validation_runs": ("ADL3",),
    "trusted_root_runs": ("ADL3",),
    "test_runs": ("ADL4", "ADL5"),
    "hfl_chunks_per_subject": (3, 3, 2),
}


def concatenate(parts: Sequence[WindowSet]) -> WindowSet:
    if not parts:
        raise ValueError("at least one WindowSet is required")
    return WindowSet(
        {name: np.concatenate([part.groups[name] for part in parts]) for name in parts[0].groups},
        np.concatenate([part.labels for part in parts]),
        np.concatenate([part.example_ids for part in parts]),
        np.concatenate([part.owners for part in parts]),
    )


def channel_counts(parts: Sequence[WindowSet]) -> dict[str, int]:
    return {name: int(values.shape[1]) for name, values in parts[0].groups.items()}


def stack_groups(windows: WindowSet) -> tuple[np.ndarray, list[str]]:
    names = list(windows.groups)
    return np.concatenate([windows.groups[name] for name in names], axis=1), names


def unstack_groups(values: np.ndarray, template: WindowSet) -> dict[str, np.ndarray]:
    groups = {}
    cursor = 0
    for name, source in template.groups.items():
        width = source.shape[1]
        groups[name] = values[:, cursor:cursor + width, :].astype(np.float32)
        cursor += width
    return groups


def _stable_ids(dataset_code: int, owner: int, count: int) -> np.ndarray:
    return dataset_code * 10**9 + owner * 10**6 + np.arange(count, dtype=np.int64)


def _mhealth_subject(
    data_root: Path, subject_id: int, window: int = 128, stride: int = 64
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    candidates = (
        data_root / f"mHealth_subject{subject_id}.log",
        data_root / "MHEALTHDATASET" / f"mHealth_subject{subject_id}.log",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"MHEALTH subject {subject_id} not found below {data_root}")
    raw = np.loadtxt(path)
    labels = raw[:, 23].astype(int)
    physical = {"chest": slice(0, 5), "left_ankle": slice(5, 14), "right_arm": slice(14, 23)}
    windows = {name: [] for name in physical}
    encoded = []
    for start in range(0, len(raw) - window + 1, stride):
        stop = start + window
        values, counts = np.unique(labels[start:stop], return_counts=True)
        majority = int(values[np.argmax(counts)])
        if majority not in range(1, 13) or counts.max() / window < 0.8:
            continue
        for name, selection in physical.items():
            windows[name].append(raw[start:stop, selection].T.astype(np.float32))
        encoded.append(majority - 1)
    if not encoded:
        raise ValueError(f"MHEALTH subject {subject_id} has no stable windows")
    return {name: np.stack(values) for name, values in windows.items()}, np.asarray(encoded)


def _split_validation_and_root(subject: WindowSet) -> tuple[WindowSet, WindowSet]:
    validation_indices, root_indices = [], []
    for label in sorted(set(subject.labels.tolist())):
        indices = np.flatnonzero(subject.labels == label)
        if len(indices) < 6:
            midpoint = len(indices) // 2
            validation_indices.extend(indices[:midpoint])
            root_indices.extend(indices[midpoint:])
            continue
        midpoint = len(indices) // 2
        validation_indices.extend(indices[:max(0, midpoint - 2)])
        root_indices.extend(indices[min(len(indices), midpoint + 2):])
    return subject.take(sorted(validation_indices)), subject.take(sorted(root_indices))


def _mhealth(data_root: Path) -> DatasetBundle:
    train_raw = [_mhealth_subject(data_root, subject) for subject in range(1, 9)]
    validation_root_raw = _mhealth_subject(data_root, 9)
    test_raw = _mhealth_subject(data_root, 10)
    stats = {}
    for source in ("chest", "left_ankle", "right_arm"):
        joined = np.concatenate([modalities[source] for modalities, _ in train_raw])
        stats[source] = (
            joined.mean(axis=(0, 2), keepdims=True).astype(np.float32),
            np.maximum(joined.std(axis=(0, 2), keepdims=True), 1e-6).astype(np.float32),
        )

    def convert(raw_subject, subject_id: int) -> WindowSet:
        modalities, labels = raw_subject
        normalized = {
            name: ((values - stats[name][0]) / stats[name][1]).astype(np.float32)
            for name, values in modalities.items()
        }
        groups = {
            name: normalized[source][:, selection, :]
            for name, (source, selection) in MHEALTH_GROUPS.items()
        }
        count = len(labels)
        return WindowSet(groups, labels.copy(), _stable_ids(1, subject_id, count),
                         np.full(count, subject_id, dtype=np.int64))

    subject9 = convert(validation_root_raw, 9)
    validation, trusted_root = _split_validation_and_root(subject9)
    return DatasetBundle(
        [convert(subject, index) for index, subject in enumerate(train_raw, start=1)],
        convert(test_raw, 10),
        validation,
        trusted_root,
        12,
        dict(MHEALTH_SPLIT),
    )


def _opportunity_directory(data_root: Path) -> Path:
    candidates = (data_root / "OpportunityUCIDataset/dataset", data_root / "dataset", data_root)
    found = next((path for path in candidates if (path / "S1-Drill.dat").exists()), None)
    if found is None:
        raise FileNotFoundError(f"OPPORTUNITY files not found below {data_root}")
    return found


def _opportunity_subject(data_root: Path, subject: int, runs: Sequence[str]) -> WindowSet:
    import pandas as pd
    import sys

    shared = Path(__file__).resolve().parents[2] / "shared/code"
    if str(shared) not in sys.path:
        sys.path.insert(0, str(shared))
    from experiment_validity import OPPORTUNITY_LABEL_COLUMN, encode_opportunity_labels

    directory = _opportunity_directory(data_root)
    body, objects, ambient, labels = [], [], [], []
    window, stride = 30, 15
    for run in runs:
        frame = pd.read_csv(directory / f"S{subject}-{run}.dat", sep=r"\s+", header=None,
                            engine="python")
        values = frame.interpolate(method="linear", limit_direction="forward").fillna(0).to_numpy()
        encoded = encode_opportunity_labels(values[:, OPPORTUNITY_LABEL_COLUMN])
        for start in range(0, len(values) - window, stride):
            stop = start + window
            body.append(values[start:stop, 1:31].T.astype(np.float32))
            objects.append(values[start:stop, 50:65].T.astype(np.float32))
            ambient.append(values[start:stop, 100:110].T.astype(np.float32))
            labels.append(int(encoded[stop - 1]))
    raw = {"body": np.stack(body), "objects": np.stack(objects), "ambient": np.stack(ambient)}
    groups = {name: raw[source][:, selection, :] for name, (source, selection) in OPPORTUNITY_GROUPS.items()}
    count = len(labels)
    run_code = sum((index + 1) * sum(ord(char) for char in run)
                   for index, run in enumerate(runs)) % 1000
    owner = int(f"{subject}{run_code:03d}")
    return WindowSet(groups, np.asarray(labels, dtype=np.int64), _stable_ids(2, owner, count),
                     np.full(count, subject, dtype=np.int64))


def _normalize(train: list[WindowSet], *others: WindowSet) -> tuple[list[WindowSet], ...]:
    stats = {}
    for name in train[0].groups:
        joined = np.concatenate([part.groups[name] for part in train])
        mean = joined.mean(axis=(0, 2), keepdims=True).astype(np.float32)
        std = np.maximum(joined.std(axis=(0, 2), keepdims=True), 1e-6).astype(np.float32)
        stats[name] = (mean, std)

    def apply(part: WindowSet) -> WindowSet:
        return WindowSet(
            {name: ((values - stats[name][0]) / stats[name][1]).astype(np.float32)
             for name, values in part.groups.items()},
            part.labels.copy(), part.example_ids.copy(), part.owners.copy(),
        )

    return (list(map(apply, train)), *(apply(part) for part in others))


def _opportunity(data_root: Path) -> DatasetBundle:
    subjects = [
        _opportunity_subject(data_root, subject, OPPORTUNITY_SPLIT["train_runs"])
        for subject in OPPORTUNITY_SPLIT["train_subjects"]
    ]
    validation = _opportunity_subject(
        data_root,
        OPPORTUNITY_SPLIT["validation_subjects"][0],
        OPPORTUNITY_SPLIT["validation_runs"],
    )
    trusted_root = _opportunity_subject(
        data_root,
        OPPORTUNITY_SPLIT["trusted_root_subjects"][0],
        OPPORTUNITY_SPLIT["trusted_root_runs"],
    )
    test = _opportunity_subject(
        data_root,
        OPPORTUNITY_SPLIT["test_subjects"][0],
        OPPORTUNITY_SPLIT["test_runs"],
    )
    subjects, validation, trusted_root, test = _normalize(subjects, validation, trusted_root, test)
    clients = []
    for subject, chunks in zip(subjects, OPPORTUNITY_SPLIT["hfl_chunks_per_subject"]):
        clients.extend(
            subject.take(indices) for indices in np.array_split(np.arange(len(subject)), chunks)
        )
    return DatasetBundle(clients, test, validation, trusted_root, 18, dict(OPPORTUNITY_SPLIT))


def load_dataset(name: str, data_root: Path) -> DatasetBundle:
    if name == "mhealth":
        return _mhealth(data_root)
    if name == "opportunity":
        return _opportunity(data_root)
    raise ValueError(f"unsupported dataset: {name}")
