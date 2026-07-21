"""Architecture-matched eight-branch models for HFL and VFL."""

from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, channels: int, output_dim: int, dropout: float):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, 32, 5, padding=2),
            nn.GroupNorm(4, 32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.network(values)


class EightPartyFusion(nn.Module):
    """Same model graph in both topologies; only ownership/training differs."""

    def __init__(self, channels: dict[str, int], fusion: str, num_classes: int,
                 embedding_dim: int = 64, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        if len(channels) != 8:
            raise ValueError(f"expected eight logical groups, got {len(channels)}")
        if fusion not in {"intermediate", "late"}:
            raise ValueError(f"unsupported fusion: {fusion}")
        self.fusion = fusion
        branch_output = embedding_dim if fusion == "intermediate" else num_classes
        self.branches = nn.ModuleDict({
            name: Encoder(count, branch_output, dropout) for name, count in channels.items()
        })
        self.coordinator = (
            nn.Sequential(
                nn.ReLU(),
                nn.Linear(embedding_dim * len(channels), hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, num_classes),
            )
            if fusion == "intermediate" else nn.Identity()
        )

    def messages(self, values: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: branch(values[name]) for name, branch in self.branches.items()}

    def fuse(self, messages: dict[str, torch.Tensor]) -> torch.Tensor:
        ordered = [messages[name] for name in self.branches]
        if self.fusion == "late":
            return torch.stack(ordered).mean(dim=0)
        return self.coordinator(torch.cat(ordered, dim=1))

    def forward(self, values: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.fuse(self.messages(values))

