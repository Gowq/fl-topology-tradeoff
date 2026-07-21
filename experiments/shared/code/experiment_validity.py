"""Shared validity helpers for the experiment suite.

The original scripts were standalone grid jobs.  This module centralizes fixes
that affect experimental validity so all jobs use the same semantics.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


OPPORTUNITY_LABEL_COLUMN = 249
OPPORTUNITY_RAW_LABELS = (
    0,
    404505,
    404508,
    404511,
    404516,
    404517,
    404519,
    404520,
    405506,
    406505,
    406508,
    406511,
    406516,
    406517,
    406519,
    406520,
    407521,
    408512,
)
OPPORTUNITY_NUM_CLASSES = len(OPPORTUNITY_RAW_LABELS)
_OPP_LABEL_MAP = {raw: idx for idx, raw in enumerate(OPPORTUNITY_RAW_LABELS)}


def encode_opportunity_labels(raw_labels: Iterable[float]) -> np.ndarray:
    """Map the selected OPPORTUNITY label column to contiguous class ids.

    The experiments use column 249 (ML_Both_Arms).  Across the
    committed OPPORTUNITY files, this column has 18 gesture/activity codes.
    Mapping those raw codes to 0..17 preserves the paper's 18-class HAR task
    without the invalid clipping used by the original scripts.
    """

    labels = np.asarray(raw_labels, dtype=int)
    unknown = sorted(set(labels.tolist()) - set(OPPORTUNITY_RAW_LABELS))
    if unknown:
        raise ValueError(
            f"Unexpected OPPORTUNITY labels {unknown}; expected {OPPORTUNITY_RAW_LABELS}"
        )
    return np.asarray([_OPP_LABEL_MAP[int(label)] for label in labels], dtype=int)


def opportunity_num_classes() -> int:
    return OPPORTUNITY_NUM_CLASSES


def participant_sample_rate(loaders: Sequence[object], batch_size: int) -> float:
    """Return the conservative max sample rate across client/silo loaders."""

    rates = []
    for loader in loaders:
        dataset_len = len(loader.dataset)
        if dataset_len <= 0:
            continue
        rates.append(min(1.0, float(batch_size) / float(dataset_len)))
    if not rates:
        raise ValueError("Cannot compute sample rate from empty loaders")
    return max(rates)


def calibrated_noise_multiplier(
    target_epsilon: float,
    sample_rate: float,
    epochs: int | None,
    delta: float,
    steps: int | None = None,
) -> float:
    """Calibrate Gaussian noise for a target epsilon over the full training run."""

    if target_epsilon <= 0:
        return 0.0
    if sample_rate <= 0 or sample_rate > 1:
        raise ValueError(f"Invalid sample_rate={sample_rate}")
    if steps is not None and steps <= 0:
        raise ValueError(f"Invalid steps={steps}")
    if steps is None and (epochs is None or epochs <= 0):
        raise ValueError(f"Invalid epochs={epochs}")

    try:
        from opacus.accountants.utils import get_noise_multiplier

        schedule = (
            {"steps": int(steps)}
            if steps is not None
            else {"epochs": int(epochs)}
        )
        return float(
            get_noise_multiplier(
                target_epsilon=float(target_epsilon),
                target_delta=float(delta),
                sample_rate=float(sample_rate),
                accountant="rdp",
                **schedule,
            )
        )
    except Exception as exc:  # pragma: no cover - only used when Opacus API changes
        raise RuntimeError(
            "Could not calibrate DP noise with Opacus. "
            "Do not fall back to 1/epsilon because it is not a privacy accountant."
        ) from exc


def dp_plan_from_loaders(
    target_epsilon: float,
    loaders: Sequence[object],
    batch_size: int,
    total_rounds: int,
    local_epochs: int,
    delta: float,
    mechanisms_per_step: int = 1,
) -> tuple[float, float, int]:
    """Return noise multiplier, sample rate, and composition-adjusted steps/round.

    `mechanisms_per_step` is the number of independent DP-SGD mechanisms a single
    participant's data passes through in one training step. For HFL this is 1
    (one model per client). For VFL a person's features are split across the silo
    encoders and the fusion head, so the person is touched by `(num_silos + 1)`
    mechanisms per step; under person-level accounting these compose
    sequentially. We therefore calibrate the noise for the full composed budget
    and inflate the reported steps by the same factor, so the composed epsilon
    over *all* mechanisms meets the target — making VFL and HFL epsilons an
    apples-to-apples comparison rather than per-mechanism figures.
    """

    if target_epsilon <= 0:
        return 0.0, 0.0, 0
    mechanisms_per_step = max(1, int(mechanisms_per_step))
    sample_rate = participant_sample_rate(loaders, batch_size)
    steps_per_round = max(len(loader) for loader in loaders) * int(local_epochs)
    composed_steps_per_round = steps_per_round * mechanisms_per_step
    noise_multiplier = calibrated_noise_multiplier(
        target_epsilon=target_epsilon,
        sample_rate=sample_rate,
        epochs=None,
        delta=delta,
        steps=int(total_rounds) * composed_steps_per_round,
    )
    return noise_multiplier, sample_rate, composed_steps_per_round


def vfl_mechanisms_per_step(loaders: Sequence[object], fusion_head) -> int:
    """Return DP mechanisms touched by one VFL participant per optimizer step."""

    fusion_has_params = any(p.requires_grad for p in fusion_head.parameters())
    return len(loaders) + int(fusion_has_params)


def composed_epsilon(
    noise_multiplier: float,
    sample_rate: float,
    steps: int,
    delta: float,
) -> float:
    """Compute composed epsilon for repeated DP-SGD steps."""

    if noise_multiplier <= 0 or steps <= 0:
        return 0.0
    try:
        from opacus.accountants import RDPAccountant

        accountant = RDPAccountant()
        for _ in range(int(steps)):
            accountant.step(noise_multiplier=float(noise_multiplier), sample_rate=float(sample_rate))
        return float(accountant.get_epsilon(delta=float(delta)))
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Could not compose epsilon with Opacus RDPAccountant") from exc


def select_malicious_indices(num_participants: int, attack_ratio: float, seed: int) -> set[int]:
    """Sample malicious participants reproducibly instead of always taking prefix IDs."""

    if num_participants <= 0:
        return set()
    count = int(num_participants * attack_ratio)
    count = max(0, min(count, num_participants))
    if count == 0:
        return set()
    rng = np.random.default_rng(int(seed))
    return set(int(i) for i in rng.choice(num_participants, size=count, replace=False))


def label_flip_ratio_for_vfl(attack_type: str, attack_ratio: float) -> float:
    """Keep VFL Label Flip explicit as label corruption, not malicious-silo count."""

    return float(attack_ratio) if attack_type == "label_flip" else 0.0


def attach_dp_to_fusion_head(fusion_head):
    """Wrap a fusion head in opacus.GradSampleModule so per-sample grads are
    captured on each backward().

    Necessary because the VFL fusion head consumes ``list[Tensor]`` (embeddings
    from multiple silos), which is incompatible with PrivacyEngine.make_private
    (it expects a DataLoader-driven single-tensor input).  GradSampleModule only
    installs hooks on inner layers (Linear/Conv/...), so list inputs at the
    outer forward are fine.

    Returns the wrapped module; replace the original reference with this.
    """

    from opacus.grad_sample import GradSampleModule

    return GradSampleModule(fusion_head)


def manual_dp_fusion_step(fusion_head, max_grad_norm: float, noise_multiplier: float):
    """Apply one DP-SGD-style step on a GradSampleModule-wrapped fusion head.

    Call **after** ``loss.backward()`` and **before** ``optimizer.step()``:

        loss.backward()
        manual_dp_fusion_step(fusion_head, max_grad_norm=5.0, noise_multiplier=sigma)
        optimizer.step()
        optimizer.zero_grad()

    Computes joint per-sample L2 norms across all parameters, clips each sample
    to ``max_grad_norm``, sums the clipped per-sample grads, adds Gaussian
    noise with standard deviation ``noise_multiplier * max_grad_norm``, and
    sets ``p.grad`` to the noisy mean.  Clears ``p.grad_sample`` afterward.

    This is the standard DP-SGD update (Abadi et al. 2016), executed manually
    because PrivacyEngine.make_private cannot wrap the VFL fusion head.
    Privacy accounting must still be done externally via RDPAccountant or
    ``composed_epsilon`` using the same ``noise_multiplier`` and the loader's
    sample rate.
    """

    import torch

    params = [
        p for p in fusion_head.parameters()
        if p.requires_grad and getattr(p, "grad_sample", None) is not None
    ]
    if not params:
        raise RuntimeError(
            "manual_dp_fusion_step requires a GradSampleModule-wrapped fusion "
            "head; no parameter has grad_sample after backward(). Wrap with "
            "attach_dp_to_fusion_head before training."
        )

    device = params[0].device
    batch_size = params[0].grad_sample.shape[0]

    per_sample_sq = torch.zeros(batch_size, device=device)
    for p in params:
        per_sample_sq += p.grad_sample.reshape(batch_size, -1).pow(2).sum(dim=1)
    per_sample_norms = per_sample_sq.sqrt()
    clip_factor = torch.clamp(max_grad_norm / (per_sample_norms + 1e-6), max=1.0)

    for p in params:
        gs = p.grad_sample
        view_shape = [batch_size] + [1] * (gs.dim() - 1)
        clipped = gs * clip_factor.view(view_shape)
        summed = clipped.sum(dim=0)
        noise = torch.randn_like(summed) * (noise_multiplier * max_grad_norm)
        p.grad = (summed + noise) / batch_size
        p.grad_sample = None
