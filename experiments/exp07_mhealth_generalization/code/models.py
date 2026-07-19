"""DP-compatible multimodal networks used by both Exp. 07 topologies."""

from __future__ import annotations

import torch
from torch import nn


INPUT_ORDER = ("chest", "left_ankle", "right_arm")
PHYSICAL_GROUPS = {
    "chest": tuple(range(0, 5)),
    "left_ankle": tuple(range(5, 14)),
    "right_arm": tuple(range(14, 23)),
}
CENTRALIZED_GROUPS = {"all_channels": tuple(range(23))}
RANDOM_GROUPS = {
    "random_1": (0, 3, 6, 9, 12, 15, 18, 21),
    "random_2": (1, 4, 7, 10, 13, 16, 19, 22),
    "random_3": (2, 5, 8, 11, 14, 17, 20),
}


def groups_for_topology(topology: str) -> dict[str, tuple[int, ...]]:
    if topology in {"hfl", "vfl"}:
        return PHYSICAL_GROUPS
    if topology == "centralized":
        return CENTRALIZED_GROUPS
    if topology == "vfl_random":
        return RANDOM_GROUPS
    prefix = "vfl_leave_"
    if topology.startswith(prefix):
        omitted = topology.removeprefix(prefix)
        if omitted not in PHYSICAL_GROUPS:
            raise ValueError(f"Unknown leave-one-device-out topology: {topology}")
        return {name: indices for name, indices in PHYSICAL_GROUPS.items() if name != omitted}
    raise ValueError(f"Unknown Exp. 07 topology: {topology}")


class SensorEncoder(nn.Module):
    def __init__(self, input_channels: int, embedding_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, padding=2),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, embedding_dim),
            nn.ReLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class MultimodalClassifier(nn.Module):
    """Intermediate-fusion model; encoders become VFL feature silos."""

    def __init__(
        self,
        topology: str = "vfl",
        embedding_dim: int = 64,
        hidden_dim: int = 128,
        dropout: float = 0.25,
        num_classes: int = 12,
    ):
        super().__init__()
        self.groups = groups_for_topology(topology)
        self.encoders = nn.ModuleDict(
            {
                name: SensorEncoder(len(indices), embedding_dim, dropout)
                for name, indices in self.groups.items()
            }
        )
        self.coordinator = nn.Sequential(
            nn.Linear(len(self.groups) * embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode(self, modalities: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        combined = torch.cat([modalities[name] for name in INPUT_ORDER], dim=1)
        return [
            self.encoders[name](combined[:, indices, :])
            for name, indices in self.groups.items()
        ]

    def forward(self, modalities: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.coordinator(torch.cat(self.encode(modalities), dim=1))


def model_kwargs(lr: float, dropout: float, hidden_dim: int) -> dict[str, float | int]:
    """Small explicit tuning grid; chosen parameters are persisted in results."""

    if lr <= 0:
        raise ValueError("learning rate must be positive")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in [0, 1)")
    if hidden_dim <= 0:
        raise ValueError("hidden_dim must be positive")
    return {"lr": float(lr), "dropout": float(dropout), "hidden_dim": int(hidden_dim)}
