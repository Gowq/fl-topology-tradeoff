"""HFL aggregators for Exp. 08, including FLTrust and FoolsGold."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch


State = Mapping[str, torch.Tensor]


def state_update(state: State, reference: State) -> dict[str, torch.Tensor]:
    return {name: state[name].detach() - reference[name].detach() for name in reference}


def apply_update(reference: State, update: State) -> dict[str, torch.Tensor]:
    return {name: reference[name].detach() + update[name].to(reference[name].device)
            for name in reference}


def flatten(update: State) -> torch.Tensor:
    return torch.cat([value.detach().float().reshape(-1).cpu() for value in update.values()])


def weighted_mean(updates: Sequence[State], weights: Sequence[float]) -> dict[str, torch.Tensor]:
    weights_tensor = torch.as_tensor(weights, dtype=torch.float32)
    if float(weights_tensor.sum()) <= 0:
        weights_tensor = torch.ones_like(weights_tensor)
    weights_tensor /= weights_tensor.sum()
    return {
        name: sum((update[name] * weights_tensor[index].to(update[name].device)
                   for index, update in enumerate(updates)), torch.zeros_like(updates[0][name]))
        for name in updates[0]
    }


def fltrust(updates: Sequence[State], root_update: State) -> tuple[dict[str, torch.Tensor], dict]:
    root_vector = flatten(root_update)
    root_norm = torch.norm(root_vector).clamp_min(1e-12)
    trusted_updates, trust_scores = [], []
    for update in updates:
        vector = flatten(update)
        vector_norm = torch.norm(vector).clamp_min(1e-12)
        score = torch.nn.functional.cosine_similarity(vector, root_vector, dim=0).clamp_min(0.0)
        scale = root_norm / vector_norm
        trusted_updates.append({name: value * scale.to(value.device) for name, value in update.items()})
        trust_scores.append(float(score))
    if sum(trust_scores) <= 0:
        return ({name: value.clone() for name, value in root_update.items()},
                {"trust_scores": trust_scores, "fallback": "root_update"})
    return weighted_mean(trusted_updates, trust_scores), {"trust_scores": trust_scores}


class FoolsGoldState:
    """Stateful FoolsGold history over client updates."""

    def __init__(self, client_count: int):
        self.history: torch.Tensor | None = None
        self.client_count = client_count

    def weights(self, updates: Sequence[State]) -> list[float]:
        vectors = torch.stack([flatten(update) for update in updates])
        self.history = vectors.clone() if self.history is None else self.history + vectors
        norms = torch.norm(self.history, dim=1, keepdim=True).clamp_min(1e-12)
        normalized = self.history / norms
        similarity = (normalized @ normalized.T).clamp(0, 1)
        similarity.fill_diagonal_(0)
        max_similarity = similarity.max(dim=1).values
        pardoned = similarity.clone()
        for i in range(self.client_count):
            for j in range(self.client_count):
                if i != j and max_similarity[i] < max_similarity[j] and max_similarity[j] > 0:
                    pardoned[i, j] *= max_similarity[i] / max_similarity[j]
        weights = 1.0 - pardoned.max(dim=1).values
        weights = weights.clamp(0, 1)
        if float(weights.max()) > 0:
            weights = weights / weights.max()
        weights = torch.log((weights / (1 - weights)).clamp(1e-6, 1e6)) + 0.5
        weights = weights.clamp(0, 1)
        if float(torch.sum(weights)) <= 0:
            weights = torch.ones_like(weights)
        return [float(value) for value in weights]


def aggregate(
    updates: Sequence[State],
    sample_counts: Sequence[int],
    method: str,
    *,
    root_update: State | None = None,
    foolsgold_state: FoolsGoldState | None = None,
) -> tuple[dict[str, torch.Tensor], dict]:
    if not updates:
        raise ValueError("at least one update is required")
    if method == "fedavg":
        return weighted_mean(updates, sample_counts), {"method": "fedavg"}
    if method == "fltrust":
        if root_update is None:
            raise ValueError("FLTrust requires a trusted root update")
        update, metadata = fltrust(updates, root_update)
        return update, {"method": "fltrust", **metadata}
    if method == "foolsgold":
        if foolsgold_state is None:
            foolsgold_state = FoolsGoldState(len(updates))
        weights = foolsgold_state.weights(updates)
        return weighted_mean(updates, weights), {
            "method": "foolsgold",
            "weights": weights,
            "mean_weight": float(np.mean(weights)),
        }
    raise ValueError(f"unsupported aggregator: {method}")
