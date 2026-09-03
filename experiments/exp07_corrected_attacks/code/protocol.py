"""Preregistered matrix for the corrected Exp. 07 attack comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product


DATASETS = ("mhealth", "opportunity")
TOPOLOGIES = ("hfl", "vfl")
FUSIONS = ("intermediate", "late")
EPSILONS = (None, 20.0, 100.0)
ATTACKS = ("label_flip", "sign_flip", "scaling", "free_rider")
ATTACK_RATIOS = (0.25, 0.75)
PRIMARY_SEEDS = (42, 123, 456, 789, 2026)
SECONDARY_SEEDS = PRIMARY_SEEDS[:3]
SECONDARY_AGGREGATORS = ("krum", "trimmed_mean", "median")
PARTICIPANT_COUNT = 8
PROTOCOL_REVISION = "v3"


@dataclass(frozen=True)
class ExperimentConfig:
    arm: str
    dataset: str
    topology: str
    fusion: str
    epsilon: float | None
    attack: str
    attack_ratio: float
    aggregator: str
    seed: int
    participant_count: int = PARTICIPANT_COUNT

    @property
    def config_id(self) -> str:
        epsilon = "nodp" if self.epsilon is None else f"eps{int(self.epsilon)}"
        ratio = str(self.attack_ratio).replace(".", "p")
        return (
            f"exp07{PROTOCOL_REVISION}__{self.arm}__{self.dataset}__{self.topology}__{self.fusion}__"
            f"{epsilon}__{self.attack}__r{ratio}__{self.aggregator}__s{self.seed}"
        )

    @property
    def is_clean(self) -> bool:
        return self.attack == "none"

    def to_dict(self) -> dict:
        return {"config_id": self.config_id, **asdict(self)}


def _conditions() -> tuple[tuple[str, float], ...]:
    return (("none", 0.0),) + tuple(product(ATTACKS, ATTACK_RATIOS))


def build_protocol() -> list[ExperimentConfig]:
    """Build 2,052 independent jobs with clean controls in every comparison."""

    configs: list[ExperimentConfig] = []
    for dataset, topology, fusion, epsilon, condition, seed in product(
        DATASETS, TOPOLOGIES, FUSIONS, EPSILONS, _conditions(), PRIMARY_SEEDS
    ):
        attack, ratio = condition
        configs.append(
            ExperimentConfig(
                arm="primary",
                dataset=dataset,
                topology=topology,
                fusion=fusion,
                epsilon=epsilon,
                attack=attack,
                attack_ratio=ratio,
                aggregator="fedavg" if topology == "hfl" else "coordinator",
                seed=seed,
            )
        )

    for dataset, fusion, epsilon, condition, aggregator, seed in product(
        DATASETS,
        FUSIONS,
        EPSILONS,
        _conditions(),
        SECONDARY_AGGREGATORS,
        SECONDARY_SEEDS,
    ):
        attack, ratio = condition
        configs.append(
            ExperimentConfig(
                arm="secondary",
                dataset=dataset,
                topology="hfl",
                fusion=fusion,
                epsilon=epsilon,
                attack=attack,
                attack_ratio=ratio,
                aggregator=aggregator,
                seed=seed,
            )
        )

    ids = [config.config_id for config in configs]
    if len(ids) != 2052 or len(ids) != len(set(ids)):
        raise AssertionError(f"invalid Exp. 07 matrix: {len(ids)} jobs, {len(set(ids))} IDs")
    return configs


def execution_blocks(arm: str, block_size: int = 6) -> list[tuple[int, ...]]:
    """Return contiguous Grid blocks while preserving global config indices."""

    if arm not in {"primary", "secondary"}:
        raise ValueError(f"unsupported arm: {arm}")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    indices = [index for index, config in enumerate(build_protocol()) if config.arm == arm]
    return [tuple(indices[start:start + block_size])
            for start in range(0, len(indices), block_size)]


def smoke_protocol() -> list[ExperimentConfig]:
    """Exercise both datasets, topologies, DP, attacks, fusions and robust aggregation."""

    return [
        ExperimentConfig("primary", "mhealth", "hfl", "intermediate", None,
                         "label_flip", 0.25, "fedavg", 42),
        ExperimentConfig("primary", "mhealth", "vfl", "late", 20.0,
                         "sign_flip", 0.25, "coordinator", 42),
        ExperimentConfig("primary", "opportunity", "hfl", "late", 20.0,
                         "free_rider", 0.75, "fedavg", 42),
        ExperimentConfig("primary", "opportunity", "vfl", "intermediate", None,
                         "scaling", 0.75, "coordinator", 42),
        ExperimentConfig("secondary", "mhealth", "hfl", "intermediate", None,
                         "sign_flip", 0.25, "krum", 42),
    ]
