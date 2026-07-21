"""Leakage-safe, eight-participant views of MHEALTH and OPPORTUNITY."""

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
            self.labels[indices], self.example_ids[indices], self.owners[indices],
        )


def concatenate(parts: Sequence[WindowSet]) -> WindowSet:
    return WindowSet(
        {name: np.concatenate([part.groups[name] for part in parts]) for name in parts[0].groups},
        np.concatenate([part.labels for part in parts]),
        np.concatenate([part.example_ids for part in parts]),
        np.concatenate([part.owners for part in parts]),
    )


def _stable_ids(dataset_code: int, owner: int, count: int) -> np.ndarray:
    return dataset_code * 10**9 + owner * 10**6 + np.arange(count, dtype=np.int64)


def _mhealth_subject(data_root: Path, subject_id: int, window: int = 128,
                     stride: int = 64) -> tuple[dict[str, np.ndarray], np.ndarray]:
    candidates = (
        data_root / f"mHealth_subject{subject_id}.log",
        data_root / "MHEALTHDATASET" / f"mHealth_subject{subject_id}.log",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(f"MHEALTH subject {subject_id} not found below {data_root}")
    raw = np.loadtxt(path)
    labels = raw[:, 23].astype(int)
    physical = {"chest": slice(0, 5), "left_ankle": slice(5, 14),
                "right_arm": slice(14, 23)}
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


def _mhealth(data_root: Path) -> tuple[list[WindowSet], WindowSet, int]:
    train_raw = [_mhealth_subject(data_root, subject) for subject in range(1, 9)]
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

    return ([convert(subject, index) for index, subject in enumerate(train_raw, start=1)],
            convert(test_raw, 10), 12)


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
    return WindowSet(groups, np.asarray(labels, dtype=np.int64), _stable_ids(2, subject, count),
                     np.full(count, subject, dtype=np.int64))


def _normalize(train: list[WindowSet], test: WindowSet) -> tuple[list[WindowSet], WindowSet]:
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
            part.labels, part.example_ids, part.owners,
        )

    return [apply(part) for part in train], apply(test)


def _opportunity(data_root: Path) -> tuple[list[WindowSet], WindowSet, int]:
    subjects = [_opportunity_subject(data_root, subject, ("Drill", "ADL1", "ADL2"))
                for subject in range(1, 5)]
    test = _opportunity_subject(data_root, 2, ("ADL4", "ADL5"))
    subjects, test = _normalize(subjects, test)
    clients = []
    for subject in subjects:
        clients.extend(subject.take(indices) for indices in np.array_split(np.arange(len(subject)), 2))
    return clients, test, 18


def load_dataset(name: str, data_root: Path) -> tuple[list[WindowSet], WindowSet, int]:
    if name == "mhealth":
        return _mhealth(data_root)
    if name == "opportunity":
        return _opportunity(data_root)
    raise ValueError(f"unsupported dataset: {name}")


def channel_counts(parts: Sequence[WindowSet]) -> dict[str, int]:
    return {name: int(values.shape[1]) for name, values in parts[0].groups.items()}
