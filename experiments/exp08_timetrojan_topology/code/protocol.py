"""Preregistered matrix for Exp. 08 TimeTrojan topology comparison."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product


DATASETS = ("mhealth", "opportunity")
TOPOLOGIES = ("hfl", "vfl")
FUSIONS = ("intermediate", "late")
EPSILONS = (None, 20.0, 100.0)
POISON_RATES = (0.0, 0.05)
PRIMARY_SEEDS = (42, 123, 456, 789, 2026)
SECONDARY_SEEDS = PRIMARY_SEEDS[:3]
SECONDARY_AGGREGATORS = ("fltrust", "foolsgold")
PARTICIPANT_COUNT = 8
TARGET_CLASS = 1


@dataclass(frozen=True)
class ExperimentConfig:
    arm: str
    dataset: str
    topology: str
    fusion: str
    epsilon: float | None
    poison_rate: float
    aggregator: str
    seed: int
    participant_count: int = PARTICIPANT_COUNT
    target_class: int = TARGET_CLASS

    @property
    def config_id(self) -> str:
        epsilon = "nodp" if self.epsilon is None else f"eps{int(self.epsilon)}"
        poison = "clean" if self.poison_rate == 0 else f"poison{int(self.poison_rate * 100)}"
        return (
            f"{self.arm}__{self.dataset}__{self.topology}__{self.fusion}__"
            f"{epsilon}__{poison}__{self.aggregator}__s{self.seed}"
        )

    @property
    def is_clean(self) -> bool:
        return self.poison_rate == 0.0

    def to_dict(self) -> dict:
        return {"config_id": self.config_id, **asdict(self)}


@dataclass(frozen=True)
class ArtifactSpec:
    dataset: str
    seed: int
    poison_rate: float = 0.05
    target_class: int = TARGET_CLASS

    @property
    def artifact_id(self) -> str:
        return (
            f"timetrojan_fgsm__{self.dataset}__target{self.target_class}__"
            f"poison{int(self.poison_rate * 100)}__s{self.seed}"
        )


def build_protocol() -> list[ExperimentConfig]:
    """Build the 384-cell Exp. 08 matrix."""

    configs: list[ExperimentConfig] = []
    for dataset, topology, fusion, epsilon, poison_rate, seed in product(
        DATASETS, TOPOLOGIES, FUSIONS, EPSILONS, POISON_RATES, PRIMARY_SEEDS
    ):
        configs.append(
            ExperimentConfig(
                arm="primary",
                dataset=dataset,
                topology=topology,
                fusion=fusion,
                epsilon=epsilon,
                poison_rate=poison_rate,
                aggregator="fedavg" if topology == "hfl" else "coordinator",
                seed=seed,
            )
        )

    for dataset, fusion, epsilon, poison_rate, aggregator, seed in product(
        DATASETS, FUSIONS, EPSILONS, POISON_RATES, SECONDARY_AGGREGATORS, SECONDARY_SEEDS
    ):
        configs.append(
            ExperimentConfig(
                arm="secondary",
                dataset=dataset,
                topology="hfl",
                fusion=fusion,
                epsilon=epsilon,
                poison_rate=poison_rate,
                aggregator=aggregator,
                seed=seed,
            )
        )

    ids = [config.config_id for config in configs]
    if len(ids) != 384 or len(ids) != len(set(ids)):
        raise AssertionError(f"invalid Exp. 08 matrix: {len(ids)} jobs, {len(set(ids))} IDs")
    return configs


def artifact_specs() -> list[ArtifactSpec]:
    return [ArtifactSpec(dataset, seed) for dataset, seed in product(DATASETS, PRIMARY_SEEDS)]


def smoke_protocol() -> list[ExperimentConfig]:
    """Small stack check covering datasets, topologies, DP, fusions and robust arms."""

    return [
        ExperimentConfig("primary", "mhealth", "hfl", "intermediate", None, 0.0, "fedavg", 42),
        ExperimentConfig("primary", "mhealth", "vfl", "late", 20.0, 0.05, "coordinator", 42),
        ExperimentConfig("primary", "opportunity", "hfl", "late", None, 0.05, "fedavg", 42),
        ExperimentConfig("primary", "opportunity", "vfl", "intermediate", 100.0, 0.0, "coordinator", 42),
        ExperimentConfig("secondary", "mhealth", "hfl", "intermediate", None, 0.05, "fltrust", 42),
        ExperimentConfig("secondary", "opportunity", "hfl", "late", None, 0.05, "foolsgold", 42),
    ]
