"""Server-side attacks and modern HFL aggregators for Exp. 07."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


State = Mapping[str, torch.Tensor]


def state_update(state: State, reference: State) -> dict[str, torch.Tensor]:
    return {name: state[name].detach() - reference[name].detach() for name in reference}


def apply_update(reference: State, update: State) -> dict[str, torch.Tensor]:
    return {name: reference[name].detach() + update[name].to(reference[name].device) for name in reference}


def flatten_update(update: State) -> torch.Tensor:
    return torch.cat([value.detach().float().reshape(-1).cpu() for value in update.values()])


def weighted_update(updates: Sequence[State], weights: torch.Tensor) -> dict[str, torch.Tensor]:
    if not updates:
        raise ValueError("at least one client update is required")
    weights = weights.float()
    if weights.numel() != len(updates) or float(weights.sum()) <= 0:
        raise ValueError("aggregation weights must be positive and match updates")
    weights = weights / weights.sum()
    return {
        name: sum(
            (update[name].detach() * weights[index].to(update[name].device)
             for index, update in enumerate(updates)),
            start=torch.zeros_like(updates[0][name]),
        )
        for name in updates[0]
    }


def model_replacement(update: State, client_count: int, malicious_count: int) -> dict[str, torch.Tensor]:
    """Scale a malicious update so FedAvg replaces, rather than nudges, the model."""

    if client_count <= 0 or malicious_count <= 0:
        raise ValueError("client_count and malicious_count must be positive")
    factor = float(client_count) / float(malicious_count)
    return {name: value * factor for name, value in update.items()}


def fedavg(updates: Sequence[State], sample_counts: Sequence[int]) -> dict[str, torch.Tensor]:
    if len(sample_counts) != len(updates) or any(count <= 0 for count in sample_counts):
        raise ValueError("sample_counts must be positive and match updates")
    return weighted_update(updates, torch.tensor(sample_counts, dtype=torch.float32))


def fltrust(updates: Sequence[State], root_update: State) -> dict[str, torch.Tensor]:
    """FLTrust aggregation using a trusted server update as the trust anchor."""

    root = flatten_update(root_update)
    root_norm = torch.linalg.vector_norm(root).clamp_min(1e-12)
    trust = []
    normalized = []
    for update in updates:
        flat = flatten_update(update)
        norm = torch.linalg.vector_norm(flat).clamp_min(1e-12)
        trust.append(torch.relu(torch.dot(flat, root) / (norm * root_norm)))
        scale = float((root_norm / norm).item())
        normalized.append({name: value * scale for name, value in update.items()})
    weights = torch.stack(trust)
    if float(weights.sum()) <= 0:
        # A zero update is safer than falling back to untrusted FedAvg.
        return {name: torch.zeros_like(value) for name, value in root_update.items()}
    return weighted_update(normalized, weights)


def foolsgold_weights(histories: torch.Tensor) -> torch.Tensor:
    """Canonical cosine-similarity/pardoning/logit weighting from FoolsGold."""

    if histories.ndim != 2 or histories.shape[0] < 2:
        return torch.ones(histories.shape[0], dtype=torch.float32)
    normalized = histories / torch.linalg.vector_norm(histories, dim=1, keepdim=True).clamp_min(1e-12)
    cosine = normalized @ normalized.T
    cosine.fill_diagonal_(0.0)
    max_similarity = cosine.max(dim=1).values
    pardoned = cosine.clone()
    for i in range(len(histories)):
        for j in range(len(histories)):
            if max_similarity[i] < max_similarity[j] and max_similarity[j] > 0:
                pardoned[i, j] *= max_similarity[i] / max_similarity[j]
    weights = 1.0 - pardoned.max(dim=1).values
    if float(weights.max()) > 0:
        weights /= weights.max()
    weights = weights.clamp(1e-6, 1 - 1e-6)
    weights = (torch.log(weights / (1 - weights)) + 0.5).clamp(0, 1)
    return weights


def foolsgold(
    updates: Sequence[State], histories: torch.Tensor | None
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    current = torch.stack([flatten_update(update) for update in updates])
    histories = current if histories is None else histories + current
    weights = foolsgold_weights(histories)
    if float(weights.sum()) <= 0:
        weights = torch.ones_like(weights)
    return weighted_update(updates, weights), histories, weights
