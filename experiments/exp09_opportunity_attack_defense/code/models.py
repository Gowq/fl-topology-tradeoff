"""OPPORTUNITY intermediate-fusion networks for the Exp. 09 transfer."""

from __future__ import annotations

import torch
from torch import nn


VFL_GROUPS = {"body_upper": 15, "body_lower": 15, "objects": 15, "ambient": 10}
HFL_GROUPS = {"body": 30, "objects": 15, "ambient": 10}


class SiloEncoder(nn.Module):
    def __init__(self, channels: int, dropout: float = 0.15):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, 64, 5), nn.GroupNorm(8, 64), nn.ReLU(),
            nn.MaxPool1d(2), nn.Conv1d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64), nn.ReLU(), nn.Dropout(dropout), nn.Flatten(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class OpportunityIntermediateModel(nn.Module):
    def __init__(
        self, topology: str, dropout: float = 0.5, num_classes: int = 18
    ):
        super().__init__()
        if topology not in {"hfl", "vfl"}:
            raise ValueError(f"Unsupported topology: {topology}")
        self.topology = topology
        self.groups = HFL_GROUPS if topology == "hfl" else VFL_GROUPS
        self.encoders = nn.ModuleDict(
            {name: SiloEncoder(channels, dropout * 0.3) for name, channels in self.groups.items()}
        )
        self.coordinator = nn.Sequential(
            nn.Linear(64 * 13 * len(self.groups), 256), nn.ReLU(),
            nn.Dropout(dropout * 0.5), nn.Linear(256, num_classes),
        )

    def split(self, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        body = values["body_sensors"]
        if self.topology == "hfl":
            return {
                "body": body, "objects": values["object_sensors"],
                "ambient": values["ambient_sensors"],
            }
        return {
            "body_upper": body[:, :15, :], "body_lower": body[:, 15:, :],
            "objects": values["object_sensors"], "ambient": values["ambient_sensors"],
        }

    def encode(self, values: dict[str, torch.Tensor]) -> list[torch.Tensor]:
        split = self.split(values)
        return [self.encoders[name](split[name]) for name in self.groups]

    def forward(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.coordinator(torch.cat(self.encode(values), dim=1))
