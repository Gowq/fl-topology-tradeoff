"""Preregistered Exp. 08 fixed-round MHEALTH matrix."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product


SEEDS = (42, 123, 456)
EPSILONS = (100.0, 200.0)
TOPOLOGIES = ("hfl", "vfl")
FUSIONS = ("intermediate", "late")


@dataclass(frozen=True)
class ExperimentConfig:
    topology: str
    fusion: str
    epsilon: float
    seed: int
    arm: str = "fixed_rounds"
    client_count: int = 8
    attack: str = "none"
    attack_ratio: float = 0.0
    aggregator: str = "fedavg"

    @property
    def config_id(self) -> str:
        return (
            f"fixed_rounds__{self.topology}__{self.fusion}__"
            f"eps{int(self.epsilon)}__s{self.seed}"
        )

    def to_dict(self) -> dict:
        return {"config_id": self.config_id, **asdict(self)}


def build_protocol() -> list[ExperimentConfig]:
    configs = [
        ExperimentConfig(topology, fusion, epsilon, seed)
        for topology, fusion, epsilon, seed in product(
            TOPOLOGIES, FUSIONS, EPSILONS, SEEDS
        )
    ]
    if len({config.config_id for config in configs}) != len(configs):
        raise AssertionError("Exp. 08 generated duplicate configuration IDs")
    return configs


def smoke_protocol() -> list[ExperimentConfig]:
    return [
        ExperimentConfig(topology, fusion, 200.0, SEEDS[0], arm="smoke")
        for topology, fusion in product(TOPOLOGIES, FUSIONS)
    ]
