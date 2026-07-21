"""HFL update aggregators used in corrected Exp. 07."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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
    normalized = torch.as_tensor(weights, dtype=torch.float32)
    normalized /= normalized.sum()
    return {
        name: sum((update[name] * normalized[index].to(update[name].device)
                   for index, update in enumerate(updates)), torch.zeros_like(updates[0][name]))
        for name in updates[0]
    }


def aggregate(updates: Sequence[State], sample_counts: Sequence[int], method: str,
              malicious_count: int) -> tuple[dict[str, torch.Tensor], dict]:
    if not updates:
        raise ValueError("at least one update is required")
    if method == "fedavg":
        return weighted_mean(updates, sample_counts), {"theoretical_condition_met": True}

    stacked = {name: torch.stack([update[name] for update in updates]) for name in updates[0]}
    if method == "median":
        return ({name: values.median(dim=0).values for name, values in stacked.items()},
                {"theoretical_condition_met": malicious_count < len(updates) / 2})
    if method == "trimmed_mean":
        trim = min(malicious_count, (len(updates) - 1) // 2)
        result = {}
        for name, values in stacked.items():
            ordered = values.sort(dim=0).values
            result[name] = ordered[trim:len(updates) - trim].mean(dim=0)
        return result, {"trim": trim, "theoretical_condition_met": 2 * malicious_count < len(updates)}
    if method == "krum":
        vectors = torch.stack([flatten(update) for update in updates])
        distances = torch.cdist(vectors, vectors).pow(2)
        neighbor_count = max(1, len(updates) - malicious_count - 2)
        scores = []
        for index in range(len(updates)):
            peers = torch.cat((distances[index, :index], distances[index, index + 1:]))
            scores.append(peers.topk(neighbor_count, largest=False).values.sum())
        selected = int(torch.stack(scores).argmin())
        return ({name: value.clone() for name, value in updates[selected].items()}, {
            "selected_client": selected,
            "neighbors": neighbor_count,
            "theoretical_condition_met": len(updates) >= 2 * malicious_count + 3,
        })
    raise ValueError(f"unsupported aggregator: {method}")

