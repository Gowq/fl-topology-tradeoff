"""Topology-equivalent attack semantics for corrected Exp. 07."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


SCALE_FACTOR = 10.0
PARTICIPANT_COUNT = 8
SEED_ORDER = (42, 123, 456, 789, 2026)


def malicious_indices(seed: int, ratio: float, count: int = PARTICIPANT_COUNT) -> tuple[int, ...]:
    """Return preregistered, approximately balanced participant assignments.

    Each preregistered seed gets a distinct rotation offset, so the five seeds
    never reuse the same malicious set.
    """

    malicious_count = int(count * ratio)
    if malicious_count not in {2, 6} or count != PARTICIPANT_COUNT:
        raise ValueError("corrected Exp. 07 supports exactly 8 participants at 25% or 75%")
    try:
        seed_position = SEED_ORDER.index(seed)
    except ValueError:
        seed_position = int(np.random.default_rng(seed).integers(0, count))
    start = seed_position % count
    return tuple(sorted((start + offset) % count for offset in range(malicious_count)))


def label_poison_ids(clients: Sequence[object], selected: Sequence[int]) -> frozenset[int]:
    """Victim example IDs: every record owned by a malicious HFL participant.

    Both topologies consume this same set, so Label Flip corrupts identical
    examples regardless of how the data is partitioned.
    """

    return frozenset(
        int(example_id)
        for index in selected
        for example_id in clients[index].example_ids
    )


def corrupt_labels(labels, example_ids, poisoned: frozenset[int], num_classes: int):
    mask = np.fromiter((int(value) in poisoned for value in example_ids), dtype=bool)
    result = labels.clone()
    if mask.any():
        import torch

        torch_mask = torch.as_tensor(mask, dtype=torch.bool, device=labels.device)
        result[torch_mask] = (result[torch_mask] + 1) % num_classes
    return result


def attack_update(update: Mapping[str, object], attack: str) -> dict[str, object]:
    """Attack the HFL transmitted update, never the absolute model state."""

    if attack == "sign_flip":
        return {name: -value for name, value in update.items()}
    if attack == "scaling":
        return {name: SCALE_FACTOR * value for name, value in update.items()}
    if attack == "free_rider":
        return {name: value * 0 for name, value in update.items()}
    return {name: value for name, value in update.items()}


def attack_message(message, attack: str):
    """Attack the representation/logit transmitted by one VFL silo."""

    if attack == "sign_flip":
        return -message
    if attack == "scaling":
        return SCALE_FACTOR * message
    # A VFL free-rider freezes local training but still sends a valid message.
    return message

