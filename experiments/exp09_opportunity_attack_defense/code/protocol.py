"""Preregistered transfer of the Exp. 07 robustness arm to OPPORTUNITY."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product


SEEDS = (42, 123, 456, 789, 2026)
EPSILONS = (0.0, 3.0, 20.0)
ATTACK_RATIOS = (0.25, 0.5)


@dataclass(frozen=True)
class ExperimentConfig:
    topology: str
    epsilon: float
    seed: int
    attack: str
    attack_ratio: float
    aggregator: str
    arm: str = "robustness_transfer"
    client_count: int = 8

    @property
    def config_id(self) -> str:
        eps = str(self.epsilon).replace(".", "p")
        ratio = str(self.attack_ratio).replace(".", "p")
        return (
            f"robustness_transfer__{self.topology}__eps{eps}__{self.attack}__"
            f"r{ratio}__{self.aggregator}__s{self.seed}"
        )

    def to_dict(self) -> dict:
        return {"config_id": self.config_id, **asdict(self)}


def build_protocol() -> list[ExperimentConfig]:
    hfl = [
        ExperimentConfig("hfl", epsilon, seed, attack, ratio, aggregator)
        for epsilon, seed, attack, ratio, aggregator in product(
            EPSILONS,
            SEEDS,
            ("model_replacement", "sensor_backdoor"),
            ATTACK_RATIOS,
            ("fedavg", "fltrust", "foolsgold"),
        )
    ]
    vfl = [
        ExperimentConfig(
            "vfl", epsilon, seed, "sensor_backdoor", ratio, "coordinator"
        )
        for epsilon, seed, ratio in product(EPSILONS, SEEDS, ATTACK_RATIOS)
    ]
    configs = hfl + vfl
    if len({config.config_id for config in configs}) != len(configs):
        raise AssertionError("Exp. 09 generated duplicate configuration IDs")
    return configs


def smoke_protocol() -> list[ExperimentConfig]:
    return [
        ExperimentConfig("hfl", 0.0, 42, "model_replacement", 0.5, "fltrust"),
        ExperimentConfig("hfl", 3.0, 42, "sensor_backdoor", 0.5, "foolsgold"),
        ExperimentConfig("vfl", 0.0, 42, "sensor_backdoor", 0.5, "coordinator"),
        ExperimentConfig("vfl", 3.0, 42, "sensor_backdoor", 0.5, "coordinator"),
    ]
