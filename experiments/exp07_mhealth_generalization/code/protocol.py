"""Canonical MHEALTH configuration matrix for Exp. 07.

The experiment is split into focused arms instead of one confounded Cartesian
product.  Every configuration is deterministic and has a stable identifier so
Grid/Slurm jobs can be resumed without silently changing the protocol.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product
from typing import Iterable


SEEDS = (42, 123, 456, 789, 2026)
INTERMEDIATE_EPSILONS = (0.0, 0.25, 0.5, 0.75, 1.0, 3.0, 8.0, 20.0)
CLIENT_COUNTS = (2, 4, 8)
ATTACK_RATIOS = (0.25, 0.5)


@dataclass(frozen=True)
class ExperimentConfig:
    arm: str
    topology: str
    epsilon: float
    seed: int
    client_count: int = 8
    attack: str = "none"
    attack_ratio: float = 0.0
    aggregator: str = "fedavg"

    @property
    def config_id(self) -> str:
        eps = str(self.epsilon).replace(".", "p")
        ratio = str(self.attack_ratio).replace(".", "p")
        return (
            f"{self.arm}__{self.topology}__eps{eps}__n{self.client_count}__"
            f"{self.attack}__r{ratio}__{self.aggregator}__s{self.seed}"
        )

    def to_dict(self) -> dict:
        return {"config_id": self.config_id, **asdict(self)}


def _configs(**axes: Iterable) -> list[ExperimentConfig]:
    names = tuple(axes)
    return [
        ExperimentConfig(**dict(zip(names, values, strict=True)))
        for values in product(*(axes[name] for name in names))
    ]


def build_protocol() -> list[ExperimentConfig]:
    """Return the preregistered Exp. 07 matrix.

    Arms answer separate reviewer questions:
    - generalization: does the topology/DP curve replicate on MHEALTH?
    - scale: does HFL behavior persist as the number of subjects grows?
    - robustness: do modern attacks and modern aggregators alter the ranking?
    - tail: are apparent curve rebounds stable under denser sampling?
    """

    generalization = _configs(
        arm=("generalization",),
        topology=("hfl", "vfl"),
        epsilon=INTERMEDIATE_EPSILONS,
        seed=SEEDS,
    )

    scale = _configs(
        arm=("scale",),
        topology=("hfl",),
        epsilon=(0.0, 3.0, 20.0),
        seed=SEEDS,
        client_count=CLIENT_COUNTS,
        attack=("none", "model_replacement"),
        attack_ratio=(0.0, 0.5),
        aggregator=("fedavg",),
    )
    # Remove nonsensical cross-products (an attack with ratio zero or no attack
    # with a non-zero ratio) while retaining a transparent generated matrix.
    scale = [
        cfg
        for cfg in scale
        if (cfg.attack == "none") == (cfg.attack_ratio == 0.0)
    ]

    robustness = _configs(
        arm=("robustness",),
        topology=("hfl",),
        epsilon=(0.0, 3.0, 20.0),
        seed=SEEDS,
        client_count=(8,),
        attack=("model_replacement", "sensor_backdoor"),
        attack_ratio=ATTACK_RATIOS,
        aggregator=("fedavg", "fltrust", "foolsgold"),
    )
    # VFL has no homologous model updates to feed to HFL aggregators.  Its
    # coordinator is evaluated only against the cross-topology sensor backdoor.
    vfl_robustness = _configs(
        arm=("robustness",),
        topology=("vfl",),
        epsilon=(0.0, 3.0, 20.0),
        seed=SEEDS,
        client_count=(8,),
        attack=("sensor_backdoor",),
        attack_ratio=ATTACK_RATIOS,
        aggregator=("coordinator",),
    )

    tail = _configs(
        arm=("tail",),
        topology=("hfl", "vfl"),
        epsilon=(50.0, 100.0, 150.0, 200.0),
        seed=SEEDS,
    )

    redundancy = _configs(
        arm=("redundancy",),
        topology=(
            "centralized",
            "vfl",
            "vfl_random",
            "vfl_leave_chest",
            "vfl_leave_left_ankle",
            "vfl_leave_right_arm",
        ),
        epsilon=(0.0, 3.0, 20.0),
        seed=SEEDS,
        aggregator=("coordinator",),
    )

    configs = generalization + scale + robustness + vfl_robustness + tail + redundancy
    ids = [cfg.config_id for cfg in configs]
    if len(ids) != len(set(ids)):
        raise AssertionError("Exp. 07 protocol generated duplicate config IDs")
    return configs


def smoke_protocol() -> list[ExperimentConfig]:
    """Minimal topology + attack path for CPU/data-pipeline validation."""

    return [
        ExperimentConfig("smoke", "hfl", 0.0, SEEDS[0], client_count=2),
        ExperimentConfig(
            "smoke",
            "hfl",
            0.0,
            SEEDS[0],
            client_count=2,
            attack="model_replacement",
            attack_ratio=0.5,
            aggregator="fltrust",
        ),
        ExperimentConfig("smoke", "vfl", 0.0, SEEDS[0], client_count=2),
    ]
