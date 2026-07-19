"""MHEALTH loader and subject-independent partitions for Exp. 07."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np


ACTIVITY_IDS = tuple(range(1, 13))
ACTIVITY_TO_CLASS = {activity: index for index, activity in enumerate(ACTIVITY_IDS)}
TRAIN_SUBJECTS = tuple(range(1, 9))
VALIDATION_SUBJECTS = (9,)
TEST_SUBJECTS = (10,)

# The official MHEALTH files contain 23 synchronized sensor channels followed
# by the activity label. Device boundaries follow the dataset documentation.
MODALITY_COLUMNS = {
    "chest": np.arange(0, 5),
    "left_ankle": np.arange(5, 14),
    "right_arm": np.arange(14, 23),
}


@dataclass(frozen=True)
class WindowedSubject:
    subject_id: int
    modalities: Mapping[str, np.ndarray]
    labels: np.ndarray

    def __len__(self) -> int:
        return len(self.labels)


def window_subject(
    raw: np.ndarray,
    subject_id: int,
    window_size: int = 128,
    stride: int = 64,
) -> WindowedSubject:
    """Convert one subject log into fixed windows with stable majority labels."""

    if raw.ndim != 2 or raw.shape[1] < 24:
        raise ValueError(f"MHEALTH subject {subject_id} must have at least 24 columns")
    if window_size <= 0 or stride <= 0:
        raise ValueError("window_size and stride must be positive")

    labels = raw[:, 23].astype(int)
    if len(labels) < window_size:
        raise ValueError(f"MHEALTH subject {subject_id} has no complete windows")

    windows: dict[str, list[np.ndarray]] = {name: [] for name in MODALITY_COLUMNS}
    encoded_labels: list[int] = []
    for start in range(0, len(labels) - window_size + 1, stride):
        stop = start + window_size
        segment_labels = labels[start:stop]
        values, counts = np.unique(segment_labels, return_counts=True)
        majority = int(values[np.argmax(counts)])
        if majority not in ACTIVITY_TO_CLASS or counts.max() / window_size < 0.8:
            continue
        for name, columns in MODALITY_COLUMNS.items():
            windows[name].append(raw[start:stop, columns].T.astype(np.float32))
        encoded_labels.append(ACTIVITY_TO_CLASS[majority])

    if not encoded_labels:
        raise ValueError(f"MHEALTH subject {subject_id} has no stable activity windows")
    return WindowedSubject(
        subject_id=subject_id,
        modalities={name: np.stack(parts) for name, parts in windows.items()},
        labels=np.asarray(encoded_labels, dtype=np.int64),
    )


def load_subject(
    dataset_root: Path,
    subject_id: int,
    window_size: int = 128,
    stride: int = 64,
) -> WindowedSubject:
    candidates = (
        dataset_root / f"mHealth_subject{subject_id}.log",
        dataset_root / "MHEALTHDATASET" / f"mHealth_subject{subject_id}.log",
    )
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError(
            f"Missing mHealth_subject{subject_id}.log below {dataset_root}. "
            "Run scripts/prepare_data.sh before Exp. 07."
        )
    return window_subject(
        np.loadtxt(path), subject_id, window_size=window_size, stride=stride
    )


def fit_normalization(subjects: Sequence[WindowedSubject]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fit channel-wise statistics on training subjects only."""

    if not subjects:
        raise ValueError("At least one training subject is required")
    stats = {}
    for name in MODALITY_COLUMNS:
        joined = np.concatenate([subject.modalities[name] for subject in subjects], axis=0)
        mean = joined.mean(axis=(0, 2), keepdims=True)
        std = joined.std(axis=(0, 2), keepdims=True)
        stats[name] = (mean.astype(np.float32), np.maximum(std, 1e-6).astype(np.float32))
    return stats


def normalize_subject(
    subject: WindowedSubject,
    stats: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> WindowedSubject:
    return WindowedSubject(
        subject.subject_id,
        {
            name: ((values - stats[name][0]) / stats[name][1]).astype(np.float32)
            for name, values in subject.modalities.items()
        },
        subject.labels.copy(),
    )


def select_client_subjects(
    subjects: Sequence[WindowedSubject], client_count: int, seed: int
) -> list[WindowedSubject]:
    """Choose N real people as HFL clients; never create virtual subjects."""

    if client_count <= 0 or client_count > len(subjects):
        raise ValueError(
            f"client_count must be between 1 and {len(subjects)}, got {client_count}"
        )
    rng = np.random.default_rng(seed)
    # Prefixes of one seeded permutation make N={2,4,8} nested for paired scale
    # comparisons instead of changing the cohort composition at every N.
    indices = sorted(rng.permutation(len(subjects))[:client_count].tolist())
    return [subjects[index] for index in indices]


def apply_sensor_trigger(
    modalities: Mapping[str, np.ndarray],
    amplitude: float = 4.0,
    fraction: float = 0.2,
) -> dict[str, np.ndarray]:
    """Add the preregistered localized backdoor trigger to the ankle gyroscope."""

    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    triggered = {name: values.copy() for name, values in modalities.items()}
    ankle = triggered["left_ankle"]
    width = max(1, int(ankle.shape[-1] * fraction))
    ankle[..., 3:6, -width:] += amplitude
    return triggered


def iter_rows(subjects: Sequence[WindowedSubject]) -> Iterator[tuple[dict[str, np.ndarray], int, int]]:
    for subject in subjects:
        for row, label in enumerate(subject.labels):
            yield (
                {name: values[row] for name, values in subject.modalities.items()},
                int(label),
                subject.subject_id,
            )
