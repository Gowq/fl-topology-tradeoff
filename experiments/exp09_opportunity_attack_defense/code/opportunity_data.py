"""Leakage-safe OPPORTUNITY windows and attack views for Exp. 09."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd


ROOT_RUNS = ("ADL3",)
TRAIN_RUNS = ("Drill", "ADL1", "ADL2")
TEST_RUNS = ("ADL4", "ADL5")
TRAIN_SUBJECTS = (1, 2, 3, 4)
ROOT_SUBJECT = 1
TEST_SUBJECT = 2
WINDOW_SIZE = 30
STRIDE = 15
BACKDOOR_TARGET = 1  # first gesture class; class 0 is the dominant null class
TRIGGER_CHANNELS = (27, 28, 29)  # localized tail channels in body_lower
TRIGGER_FRACTION = 0.2
TRIGGER_STD_MULTIPLIER = 4.0


@dataclass(frozen=True)
class WindowSet:
    body: np.ndarray
    objects: np.ndarray
    ambient: np.ndarray
    labels: np.ndarray
    subjects: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)

    def take(self, indices: Sequence[int]) -> "WindowSet":
        index = np.asarray(indices, dtype=int)
        return WindowSet(
            self.body[index], self.objects[index], self.ambient[index],
            self.labels[index], self.subjects[index]
        )


def dataset_directory(data_root: Path) -> Path:
    candidates = (
        data_root / "OpportunityUCIDataset" / "dataset",
        data_root / "dataset",
        data_root,
    )
    directory = next((candidate for candidate in candidates if (candidate / "S1-Drill.dat").exists()), None)
    if directory is None:
        raise FileNotFoundError(
            f"OPPORTUNITY .dat files not found below {data_root}; run scripts/prepare_data.sh"
        )
    return directory


def _label_encoder():
    import sys

    shared = Path(__file__).resolve().parents[3] / "shared/code"
    if str(shared) not in sys.path:
        sys.path.insert(0, str(shared))
    from experiment_validity import OPPORTUNITY_LABEL_COLUMN, encode_opportunity_labels

    return OPPORTUNITY_LABEL_COLUMN, encode_opportunity_labels


def load_subject(data_root: Path, subject: int, runs: Sequence[str]) -> WindowSet:
    label_column, encode_labels = _label_encoder()
    directory = dataset_directory(data_root)
    parts = {"body": [], "objects": [], "ambient": [], "labels": []}
    for run in runs:
        path = directory / f"S{subject}-{run}.dat"
        if not path.exists():
            raise FileNotFoundError(f"Required OPPORTUNITY run is missing: {path}")
        frame = pd.read_csv(path, sep=r"\s+", header=None, engine="python")
        values = frame.interpolate(method="linear", limit_direction="forward").fillna(0).to_numpy()
        labels = encode_labels(values[:, label_column])
        # Preserve the original OPPORTUNITY artifact's window boundary exactly.
        for start in range(0, len(values) - WINDOW_SIZE, STRIDE):
            stop = start + WINDOW_SIZE
            parts["body"].append(values[start:stop, 1:31].T.astype(np.float32))
            parts["objects"].append(values[start:stop, 50:65].T.astype(np.float32))
            parts["ambient"].append(values[start:stop, 100:110].T.astype(np.float32))
            parts["labels"].append(int(labels[stop - 1]))
    count = len(parts["labels"])
    return WindowSet(
        np.stack(parts["body"]), np.stack(parts["objects"]), np.stack(parts["ambient"]),
        np.asarray(parts["labels"], dtype=np.int64), np.full(count, subject, dtype=np.int64)
    )


def concatenate(parts: Sequence[WindowSet]) -> WindowSet:
    return WindowSet(
        *(np.concatenate([getattr(part, field) for part in parts], axis=0)
          for field in ("body", "objects", "ambient", "labels", "subjects"))
    )


def load_splits(data_root: Path) -> tuple[list[WindowSet], WindowSet, WindowSet]:
    train = [load_subject(data_root, subject, TRAIN_RUNS) for subject in TRAIN_SUBJECTS]
    root = load_subject(data_root, ROOT_SUBJECT, ROOT_RUNS)
    test = load_subject(data_root, TEST_SUBJECT, TEST_RUNS)
    return train, root, test


def horizontal_clients(subjects: Sequence[WindowSet], clients_per_subject: int = 2) -> list[WindowSet]:
    clients = []
    for subject in subjects:
        for indices in np.array_split(np.arange(len(subject)), clients_per_subject):
            clients.append(subject.take(indices))
    return clients


def trigger_scale(subjects: Sequence[WindowSet]) -> np.ndarray:
    joined = np.concatenate([subject.body[:, TRIGGER_CHANNELS, :] for subject in subjects], axis=0)
    return np.maximum(joined.std(axis=(0, 2)), 1e-6).astype(np.float32)


def apply_sensor_trigger(
    modalities: Mapping[str, np.ndarray], scale: np.ndarray,
    fraction: float = TRIGGER_FRACTION,
) -> dict[str, np.ndarray]:
    triggered = {name: values.copy() for name, values in modalities.items()}
    width = max(1, int(triggered["body_sensors"].shape[-1] * fraction))
    triggered["body_sensors"][TRIGGER_CHANNELS, -width:] += (
        TRIGGER_STD_MULTIPLIER * scale[:, None]
    )
    return triggered
