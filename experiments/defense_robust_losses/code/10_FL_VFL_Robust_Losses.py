#!/usr/bin/env python3
"""Experiment 10: noise-robust loss defenses for VFL under calibrated DP.

Exp09 showed that sample-selection / consistency defenses (small_loss,
leave_one_silo, fusion_consistency) do NOT rescue VFL from untargeted
label_flip — the F1 stays at the OPPORTUNITY floor. Exp10 explores an
orthogonal family: noise-robust loss functions that make training intrinsically
tolerant to the symmetric label noise that untargeted label_flip induces,
without trying to detect which samples are corrupted.

Defenses (all applied at the fusion-head loss, DP path unchanged):
  - gce             Generalized Cross Entropy (Zhang & Sabuncu, NeurIPS 2018)
  - sce             Symmetric Cross Entropy (Wang et al., ICCV 2019)
  - label_smoothing soft targets regularizer
  - bootstrap       soft bootstrapping (Reed et al., 2015)
  - none            plain CE baseline

This is a PRELIMINARY exploration. Per policy, smoke runs on Pegasus only
(via Docker); Grid is reserved for the definitive run. Do not run locally.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
EXPERIMENTS_DIR = HERE.parents[1]
SHARED_CODE = EXPERIMENTS_DIR / "shared" / "code"
EXP08_CODE = EXPERIMENTS_DIR / "exp03_attacks_aggregation" / "code"
if str(SHARED_CODE) not in sys.path:
    sys.path.insert(0, str(SHARED_CODE))
if str(EXP08_CODE) not in sys.path:
    sys.path.insert(0, str(EXP08_CODE))

from timeout_utils import format_timeout_duration


def _load_exp08_vfl_module():
    path = EXP08_CODE / "08_FL_RobustAgg_DP_Frontier_part3.py"
    spec = importlib.util.spec_from_file_location("exp08_vfl_part3", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Exp08 VFL module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


p8 = _load_exp08_vfl_module()


class AttackConfig:
    LABEL_FLIP = p8.AttackConfig.LABEL_FLIP
    SIGN_FLIP = p8.AttackConfig.SIGN_FLIP
    SCALING = p8.AttackConfig.SCALING
    FREE_RIDER = p8.AttackConfig.FREE_RIDER
    TARGETED_LABEL_FLIP = "targeted_label_flip"
    NONE = p8.AttackConfig.NONE

    # Exp10 centers on the label_flip failure mode. targeted_label_flip is kept
    # for an Exp09 cross-reference and sign_flip as a parameter-space shield
    # sanity check (a good loss must not destroy the Exp08 shield).
    ATTACK_TYPES = [
        LABEL_FLIP,
        TARGETED_LABEL_FLIP,
        SIGN_FLIP,
    ]


class DefenseConfig:
    NONE = "none"
    GCE = "gce"                      # Generalized CE (Zhang & Sabuncu 2018)
    MAE = "mae"                      # Mean Absolute Error loss (robust extreme)
    SCE = "sce"                      # Symmetric CE (Wang et al. 2019)
    NCE_RCE = "nce_rce"              # Active-Passive Loss / APL (Ma et al. 2020)
    LABEL_SMOOTHING = "label_smoothing"
    BOOTSTRAP = "bootstrap"          # soft bootstrapping (Reed et al. 2015)

    # Breadth-first sweep: distinct noise-robust training objectives, all cheap
    # and applied at the same fusion-head loss point. The most promising one
    # gets a deeper Grid run later.
    DEFENSES = [NONE, GCE, MAE, SCE, NCE_RCE, LABEL_SMOOTHING, BOOTSTRAP]


# Pegasus smoke matrix: focus on the label_flip failure mode, keep
# targeted_label_flip for an Exp09 cross-reference and sign_flip as a
# shield-preservation sanity check. VFL Intermediate, eps=100, 1 seed.
# A Grid definitive run would later expand fusions/epsilons/seeds.
FUSION_MODES = ["Intermediate"]
DP_EPSILONS = [100.0]
ATTACK_RATIOS = [0.50, 0.75]
SEEDS = [42, 123, 456]
SMOKE_SEEDS = [42]

TARGET_SOURCE_CLASS = int(os.environ.get("EXP10_TARGET_SOURCE_CLASS", "1"))
TARGET_CLASS = int(os.environ.get("EXP10_TARGET_CLASS", "2"))

# Robust-loss hyperparameters (literature defaults; overridable for tuning).
GCE_Q = float(os.environ.get("EXP10_GCE_Q", "0.7"))
SCE_ALPHA = float(os.environ.get("EXP10_SCE_ALPHA", "0.1"))
SCE_BETA = float(os.environ.get("EXP10_SCE_BETA", "1.0"))
SCE_RCE_CLAMP = float(os.environ.get("EXP10_SCE_RCE_CLAMP", "1e-4"))
LABEL_SMOOTHING_EPS = float(os.environ.get("EXP10_LABEL_SMOOTHING", "0.1"))
BOOTSTRAP_BETA = float(os.environ.get("EXP10_BOOTSTRAP_BETA", "0.95"))


@dataclass(frozen=True)
class SemanticDefense:
    name: str
    keep_rate: float = 1.0


def set_seed(seed: int) -> None:
    p8.set_seed(seed)


def defense_for(name: str, attack_ratio: float) -> SemanticDefense:
    # Robust-loss defenses carry no per-sample selection state; hyperparameters
    # live in module constants so they can be swept without touching this map.
    return SemanticDefense(name=name)


def apply_targeted_label_flip(
    labels: torch.Tensor,
    flip_ratio: float,
    source_class: int = TARGET_SOURCE_CLASS,
    target_class: int = TARGET_CLASS,
) -> torch.Tensor:
    """Flip a fraction of source-class labels to a fixed target class.

    Batches without source-class examples are left unchanged. This keeps
    targeted_label_flip semantically distinct from generic label_flip and makes
    target_attack_success interpretable as source->target coercion.
    """

    flipped = labels.clone()
    if flip_ratio <= 0:
        return flipped

    source_mask = labels == int(source_class)
    candidate_indices = torch.nonzero(source_mask, as_tuple=False).flatten()
    if candidate_indices.numel() == 0:
        return flipped

    n_to_flip = max(1, int(candidate_indices.numel() * float(flip_ratio)))
    n_to_flip = min(n_to_flip, candidate_indices.numel())
    perm = torch.randperm(candidate_indices.numel(), device=labels.device)[:n_to_flip]
    flip_indices = candidate_indices[perm]
    flipped[flip_indices] = int(target_class)
    return flipped


def apply_semantic_label_attack(labels: torch.Tensor, attack_type: str, attack_ratio: float) -> torch.Tensor:
    if attack_type == AttackConfig.LABEL_FLIP:
        return p8.apply_label_flipping(labels, p8.NUM_CLASSES, flip_ratio=attack_ratio).to(labels.device)
    if attack_type == AttackConfig.TARGETED_LABEL_FLIP:
        return apply_targeted_label_flip(labels, attack_ratio)
    return labels


def label_corruption_ratio(attack_type: str, attack_ratio: float) -> float:
    if attack_type in (AttackConfig.LABEL_FLIP, AttackConfig.TARGETED_LABEL_FLIP):
        return float(attack_ratio)
    return 0.0


# --------------------------------------------------------------------------
# Noise-robust loss functions. Each reduces to a mean over per-sample losses,
# so Opacus per-sample gradients and the manual DP fusion step are unchanged.
# --------------------------------------------------------------------------

def gce_loss(logits: torch.Tensor, labels: torch.Tensor, q: float = GCE_Q) -> torch.Tensor:
    """Generalized Cross Entropy: L = (1 - p_y^q) / q. Interpolates CE (q->0)
    and MAE (q=1); robust to symmetric label noise."""
    probs = F.softmax(logits, dim=1)
    p_y = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp_min(1e-7)
    return ((1.0 - p_y.pow(q)) / q).mean()


def mae_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mean Absolute Error on the softmax simplex: sum_k |p_k - y_k| = 2(1 - p_y).
    The robust extreme (bounded loss), but slow to converge."""
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(labels, probs.size(1)).float()
    return (probs - one_hot).abs().sum(dim=1).mean()


def sce_loss(logits: torch.Tensor, labels: torch.Tensor,
             alpha: float = SCE_ALPHA, beta: float = SCE_BETA) -> torch.Tensor:
    """Symmetric Cross Entropy: alpha*CE + beta*RCE. RCE = -sum p * log(y),
    with the one-hot zeros clamped (acts as the paper's constant A)."""
    num_classes = logits.size(1)
    ce = F.cross_entropy(logits, labels)
    probs = F.softmax(logits, dim=1).clamp(1e-7, 1.0)
    label_dist = F.one_hot(labels, num_classes).float().clamp_min(SCE_RCE_CLAMP)
    rce = -(probs * torch.log(label_dist)).sum(dim=1).mean()
    return alpha * ce + beta * rce


def nce_rce_loss(logits: torch.Tensor, labels: torch.Tensor,
                 alpha: float = SCE_ALPHA, beta: float = SCE_BETA) -> torch.Tensor:
    """Active-Passive Loss (Ma et al. 2020): Normalized CE + Reverse CE.
    NCE = CE / sum_k CE_k normalizes the active term; RCE is the passive,
    bounded term. Combination is provably robust and trains faster than MAE."""
    num_classes = logits.size(1)
    log_probs = F.log_softmax(logits, dim=1)
    # Normalized CE: per-sample CE divided by the sum of CE over all classes.
    ce_y = -log_probs.gather(1, labels.unsqueeze(1)).squeeze(1)
    nce = (ce_y / (-log_probs.sum(dim=1)).clamp_min(1e-7)).mean()
    probs = log_probs.exp().clamp(1e-7, 1.0)
    label_dist = F.one_hot(labels, num_classes).float().clamp_min(SCE_RCE_CLAMP)
    rce = -(probs * torch.log(label_dist)).sum(dim=1).mean()
    return alpha * nce + beta * rce


def label_smoothing_loss(logits: torch.Tensor, labels: torch.Tensor,
                         eps: float = LABEL_SMOOTHING_EPS) -> torch.Tensor:
    """Plain CE with label smoothing — soft targets dilute the impact of a
    flipped hard label."""
    return F.cross_entropy(logits, labels, label_smoothing=eps)


def bootstrap_soft_loss(logits: torch.Tensor, labels: torch.Tensor,
                        beta: float = BOOTSTRAP_BETA) -> torch.Tensor:
    """Soft bootstrapping (Reed et al. 2015): blend the given label with the
    model's own prediction, target = beta*y + (1-beta)*p_hat."""
    num_classes = logits.size(1)
    log_probs = F.log_softmax(logits, dim=1)
    probs = log_probs.exp().detach()
    one_hot = F.one_hot(labels, num_classes).float()
    targets = beta * one_hot + (1.0 - beta) * probs
    return -(targets * log_probs).sum(dim=1).mean()


ROBUST_LOSSES = {
    DefenseConfig.GCE: gce_loss,
    DefenseConfig.MAE: mae_loss,
    DefenseConfig.SCE: sce_loss,
    DefenseConfig.NCE_RCE: nce_rce_loss,
    DefenseConfig.LABEL_SMOOTHING: label_smoothing_loss,
    DefenseConfig.BOOTSTRAP: bootstrap_soft_loss,
}


class SemanticDefenseCoordinator:
    def __init__(
        self,
        silos,
        fusion_head,
        device,
        params,
        defense: SemanticDefense,
        use_label_attack: bool = False,
        attack_type: str = AttackConfig.NONE,
        attack_ratio: float = 0.5,
    ):
        self.silos = silos
        self.fusion_head = fusion_head.to(device)
        self.device = device
        self.params = params
        self.defense = defense
        self.use_label_attack = use_label_attack
        self.attack_type = attack_type
        self.attack_ratio = attack_ratio

        if list(fusion_head.parameters()):
            self.optimizer = torch.optim.SGD(fusion_head.parameters(), lr=params["lr"], momentum=params["momentum"])
        else:
            self.optimizer = None
        self.fusion_head_dp_enabled = False
        self.fusion_head_dp_noise_multiplier = 0.0

    def make_fusion_head_private(self, noise_multiplier: float) -> None:
        if self.optimizer is None:
            return
        self.fusion_head = p8.attach_dp_to_fusion_head(self.fusion_head)
        self.fusion_head_dp_enabled = True
        self.fusion_head_dp_noise_multiplier = noise_multiplier

    def _loss_for_batch(self, embeddings: list[torch.Tensor], labels: torch.Tensor) -> torch.Tensor:
        output = self.fusion_head(embeddings)
        loss_fn = ROBUST_LOSSES.get(self.defense.name)
        if loss_fn is not None:
            return loss_fn(output, labels)
        return F.cross_entropy(output, labels)

    def train_round(self, silo_loaders, label_loader, epochs: int = 1) -> float:
        self.fusion_head.train()
        for silo in self.silos.values():
            silo.encoder.train()

        total_loss = 0.0
        num_batches = 0

        for _epoch in range(epochs):
            silo_iters = {name: iter(loader) for name, loader in silo_loaders.items()}
            label_iter = iter(label_loader)

            try:
                while True:
                    silo_batches = {name: next(silo_iters[name])[0].to(self.device) for name in self.silos.keys()}
                    labels = next(label_iter)[0].to(self.device)

                    if self.use_label_attack:
                        labels = apply_semantic_label_attack(labels, self.attack_type, self.attack_ratio)

                    if self.optimizer:
                        self.optimizer.zero_grad()
                    for silo in self.silos.values():
                        silo.optimizer.zero_grad()

                    embeddings = [self.silos[name].forward(batch) for name, batch in silo_batches.items()]
                    loss = self._loss_for_batch(embeddings, labels)
                    loss.backward()

                    if self.optimizer:
                        if self.fusion_head_dp_enabled:
                            p8.manual_dp_fusion_step(
                                self.fusion_head,
                                max_grad_norm=5.0,
                                noise_multiplier=self.fusion_head_dp_noise_multiplier,
                            )
                        else:
                            torch.nn.utils.clip_grad_norm_(self.fusion_head.parameters(), max_norm=5.0)

                    for silo in self.silos.values():
                        if not silo.privacy_engine:
                            torch.nn.utils.clip_grad_norm_(silo.encoder.parameters(), max_norm=5.0)

                    if self.optimizer:
                        self.optimizer.step()
                    for silo in self.silos.values():
                        silo.optimizer.step()

                    total_loss += float(loss.item())
                    num_batches += 1
            except StopIteration:
                pass

        return total_loss / max(num_batches, 1)

    def evaluate(self, test_silo_loaders, test_label_loader):
        self.fusion_head.eval()
        for silo in self.silos.values():
            silo.encoder.eval()

        all_preds = []
        all_targets = []
        silo_iters = {name: iter(loader) for name, loader in test_silo_loaders.items()}
        label_iter = iter(test_label_loader)

        with torch.no_grad():
            try:
                while True:
                    silo_batches = {name: next(silo_iters[name])[0].to(self.device) for name in self.silos.keys()}
                    labels = next(label_iter)[0].to(self.device)
                    embeddings = [self.silos[name].forward(silo_batches[name]) for name in self.silos.keys()]
                    output = self.fusion_head(embeddings)
                    all_preds.extend(output.argmax(dim=1).cpu().numpy())
                    all_targets.extend(labels.cpu().numpy())
            except StopIteration:
                pass

        acc = p8.accuracy_score(all_targets, all_preds)
        f1 = p8.f1_score(all_targets, all_preds, average="macro", zero_division=0)
        target_metrics = targeted_attack_metrics(all_targets, all_preds)
        return acc, f1, target_metrics


def targeted_attack_metrics(targets, preds) -> dict[str, float | int | None]:
    targets_np = np.asarray(targets)
    preds_np = np.asarray(preds)
    source_mask = targets_np == TARGET_SOURCE_CLASS
    if source_mask.sum() == 0:
        return {
            "target_source_class": TARGET_SOURCE_CLASS,
            "target_class": TARGET_CLASS,
            "target_source_count": 0,
            "target_attack_success": None,
            "target_source_recall": None,
        }
    source_preds = preds_np[source_mask]
    return {
        "target_source_class": TARGET_SOURCE_CLASS,
        "target_class": TARGET_CLASS,
        "target_source_count": int(source_mask.sum()),
        "target_attack_success": float((source_preds == TARGET_CLASS).mean()),
        "target_source_recall": float((source_preds == TARGET_SOURCE_CLASS).mean()),
    }


def build_silos(silo_loaders, params, attack_type: str, attack_ratio: float, seed: int):
    silo_names = list(silo_loaders.keys())
    malicious_ids = p8.select_malicious_indices(len(silo_names), attack_ratio, seed)
    malicious_names = [silo_names[i] for i in malicious_ids]
    silo_channels = {"body_upper": 15, "body_lower": 15, "objects": 15, "ambient": 10}

    silos = {}
    for idx, (name, channels) in enumerate(silo_channels.items()):
        is_malicious = name in malicious_names
        silos[name] = p8.MaliciousVerticalSilo(
            idx,
            name,
            channels,
            p8.DEVICE,
            params,
            is_malicious=is_malicious,
            attack_type=attack_type if is_malicious else AttackConfig.NONE,
        )
    return silos, malicious_names


def run_vertical_fl_semantic_defense(
    fusion_mode: str,
    dp_epsilon: float,
    attack_type: str,
    attack_ratio: float,
    defense_name: str,
    seed: int,
    silo_loaders,
    label_loader,
    test_silo_loaders,
    test_label_loader,
    num_rounds: int = p8.NUM_ROUNDS,
    epochs: int = p8.LOCAL_EPOCHS,
):
    set_seed(seed)
    params = p8.DATASET_PARAMS[p8.DATASET_OPPORTUNITY].copy()
    params["lr"] = 0.01 if dp_epsilon > 0 else 0.001
    if fusion_mode == "Intermediate":
        params["lr"] *= 0.3

    if silo_loaders is None:
        raise RuntimeError("Data Partition Failed")

    use_dp = dp_epsilon > 0
    noise_multiplier, dp_sample_rate, dp_steps_per_round = p8.dp_plan_from_loaders(
        dp_epsilon,
        list(silo_loaders.values()),
        params["batch_size"],
        num_rounds,
        epochs,
        p8.DP_DELTA,
        mechanisms_per_step=len(silo_loaders) + 1,
    )

    silos, malicious_names = build_silos(silo_loaders, params, attack_type, attack_ratio, seed)
    if use_dp:
        for silo in silos.values():
            silo.make_private(silo_loaders[silo.silo_name], noise_multiplier=noise_multiplier)

    if fusion_mode == "Intermediate":
        fusion_head = p8.IntermediateFusionVFL(dropout_p=params["dropout"])
    elif fusion_mode == "Late":
        fusion_head = p8.LateFusionVFL(dropout_p=params["dropout"])
    else:
        raise ValueError(f"Unsupported fusion_mode={fusion_mode}")

    corruption_ratio = label_corruption_ratio(attack_type, attack_ratio)
    defense = defense_for(defense_name, attack_ratio)
    coordinator = SemanticDefenseCoordinator(
        silos,
        fusion_head,
        p8.DEVICE,
        params,
        defense=defense,
        use_label_attack=corruption_ratio > 0,
        attack_type=attack_type,
        attack_ratio=attack_ratio,
    )
    if use_dp:
        coordinator.make_fusion_head_private(noise_multiplier)

    round_metrics = []
    best_f1 = 0.0
    best_checkpoint = None
    rounds_without_improvement = 0
    prev_loss = 0.0
    loss_stagnation_count = 0

    for round_idx in range(num_rounds):
        loss = coordinator.train_round(silo_loaders, label_loader, epochs=epochs)

        if loss > p8.EARLY_STOP_LOSS_THRESHOLD:
            print(f" [EARLY STOP: Loss={loss:.2e}] ", end="")
            break

        if round_idx > 0:
            loss_change_pct = abs(loss - prev_loss) / max(prev_loss, 1e-6)
            if loss_change_pct < p8.LOSS_STAGNATION_THRESHOLD:
                loss_stagnation_count += 1
                if loss_stagnation_count >= p8.LOSS_STAGNATION_PATIENCE:
                    print(" [EARLY STOP: Loss stagnation] ", end="")
                    break
            else:
                loss_stagnation_count = 0
        prev_loss = loss

        gc.collect()
        torch.cuda.empty_cache()

        acc, f1, target_metrics = coordinator.evaluate(test_silo_loaders, test_label_loader)
        spent_eps = p8.composed_epsilon(
            noise_multiplier,
            dp_sample_rate,
            (round_idx + 1) * dp_steps_per_round,
            p8.DP_DELTA,
        ) if use_dp else 0.0
        metrics = {
            "round": round_idx + 1,
            "loss": loss,
            "acc": acc,
            "f1": f1,
            "epsilon": spent_eps,
            "fusion_mode": fusion_mode,
            "topology": "VFL",
            "aggregation": "coordinator",
            "attack_type": attack_type,
            "attack_ratio": attack_ratio,
            "defense": defense_name,
            "defense_keep_rate": defense.keep_rate,
            "num_malicious": len(malicious_names),
            "malicious_silos": malicious_names,
            "label_corruption_ratio": corruption_ratio,
            **target_metrics,
        }
        round_metrics.append(metrics)

        if f1 > best_f1:
            best_f1 = f1
            best_checkpoint = {
                "fusion_head": {k: v.cpu() for k, v in coordinator.fusion_head.state_dict().items()},
                "silos": {
                    name: {k: v.cpu() for k, v in silo.encoder.state_dict().items()}
                    for name, silo in coordinator.silos.items()
                },
            }
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1

        if rounds_without_improvement >= p8.EARLY_STOP_PATIENCE and round_idx >= p8.WARMUP_ROUNDS:
            print(" [EARLY STOP: No improvement] ", end="")
            break

        if (round_idx + 1) % 5 == 0:
            print(f" (R{round_idx+1}: F1={f1:.4f}) ", end="", flush=True)
        else:
            print(".", end="", flush=True)

    if best_checkpoint is not None and round_metrics:
        coordinator.fusion_head.load_state_dict(best_checkpoint["fusion_head"])
        for name, silo in coordinator.silos.items():
            raw_sd = best_checkpoint["silos"][name]
            clean_sd = {k.replace("_module.", ""): v for k, v in raw_sd.items()}
            try:
                silo.encoder.load_state_dict(clean_sd)
            except RuntimeError:
                silo.encoder.load_state_dict(raw_sd)
        acc, f1, target_metrics = coordinator.evaluate(test_silo_loaders, test_label_loader)
        spent_eps = p8.composed_epsilon(
            noise_multiplier,
            dp_sample_rate,
            len(round_metrics) * dp_steps_per_round,
            p8.DP_DELTA,
        ) if use_dp else 0.0
        round_metrics.append({
            **round_metrics[-1],
            "round": len(round_metrics) + 1,
            "loss": round_metrics[-1]["loss"],
            "acc": acc,
            "f1": f1,
            "epsilon": spent_eps,
            "note": "best_checkpoint",
            **target_metrics,
        })

    cleanup_training_objects(coordinator, silos, fusion_head, best_checkpoint)
    return round_metrics


def cleanup_training_objects(coordinator, silos, fusion_head, best_checkpoint) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    for silo in silos.values():
        for param in silo.encoder.parameters():
            if hasattr(param, "grad_sample"):
                del param.grad_sample
            param.grad = None
        if silo.privacy_engine is not None:
            silo.privacy_engine = None
        silo.encoder = silo.encoder.cpu()
        silo.encoder = None
        del silo.optimizer
    del coordinator
    del silos
    del fusion_head
    del best_checkpoint
    gc.collect()
    torch.cuda.empty_cache()


def build_experiments():
    experiments = []
    for fusion in FUSION_MODES:
        for attack_type in AttackConfig.ATTACK_TYPES:
            for attack_ratio in ATTACK_RATIOS:
                for dp_epsilon in DP_EPSILONS:
                    for defense in DefenseConfig.DEFENSES:
                        experiments.append({
                            "fusion": fusion,
                            "attack_type": attack_type,
                            "attack_ratio": attack_ratio,
                            "dp_epsilon": dp_epsilon,
                            "defense": defense,
                        })
    return experiments


def _config_worker(exp_dict, seeds, result_path, batch_size):
    sl, ll, tsl, tll = p8.partition_opportunity_vertical(batch_size)
    if sl is None:
        with open(result_path, "w") as f:
            json.dump([[] for _ in seeds], f)
        return

    seed_results = []
    for seed in seeds:
        try:
            res = run_vertical_fl_semantic_defense(
                exp_dict["fusion"],
                exp_dict["dp_epsilon"],
                exp_dict["attack_type"],
                exp_dict["attack_ratio"],
                exp_dict["defense"],
                seed,
                sl,
                ll,
                tsl,
                tll,
            )
            seed_results.append(res if res else [])
        except Exception as exc:
            print(f"  [subprocess error seed={seed}]: {exc}", flush=True)
            seed_results.append([])
        gc.collect()
        torch.cuda.empty_cache()

    with open(result_path, "w") as f:
        json.dump(seed_results, f)
    gc.collect()
    torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-seeds", type=int, default=1, choices=[1, 3],
                        help="Use 1 seed for Pegasus technical smoke, 3 for expanded run.")
    parser.add_argument("--max-configs", type=int, default=None,
                        help="Optional prefix limit for debugging on Pegasus only.")
    args = parser.parse_args()

    seeds = SMOKE_SEEDS if args.smoke_seeds == 1 else SEEDS
    experiments = build_experiments()
    if args.max_configs is not None:
        experiments = experiments[:args.max_configs]

    # Results dir is overridable so the Grid definitive run does not collide with
    # the Pegasus/Grid smoke artifacts (which live in results/).
    results_dir = os.environ.get("EXP10_RESULTS_DIR", "results")
    os.makedirs(results_dir, exist_ok=True)
    outfile = os.path.join(results_dir, "exp10_vfl_robust_losses.json")
    partial_path = os.path.join(results_dir, "partial_results_exp10_vfl_robust_losses.json")

    def config_sig(cfg):
        return (cfg["fusion"], cfg["attack_type"], cfg["attack_ratio"],
                cfg["dp_epsilon"], cfg["defense"])

    def is_complete(entry):
        runs = entry.get("runs", [])
        return len(runs) == len(seeds) and all(r for r in runs)

    # Resume: reuse fully-complete configs from a prior (possibly requeued) run.
    # A config is only reused if its seed count matches the current run, so a
    # 1-seed smoke partial does not satisfy a 3-seed definitive run.
    done_by_sig = {}
    if os.path.exists(partial_path):
        try:
            with open(partial_path) as f:
                prev = json.load(f)
            for entry in prev:
                if is_complete(entry):
                    done_by_sig[config_sig(entry["config"])] = entry
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            print(f"[resume] ignoring unreadable partial ({exc})", flush=True)

    all_results = []
    todo = []
    for exp in experiments:
        prev_entry = done_by_sig.get(config_sig(exp))
        if prev_entry is not None:
            all_results.append(prev_entry)
        else:
            todo.append(exp)

    print(f"Starting Exp10 VFL Robust Losses on {os.uname().nodename}")
    print(f"Device: {p8.DEVICE}")
    print(f"Results dir: {results_dir}")
    print(f"Configs: {len(experiments)} total | "
          f"{len(all_results)} resumed | {len(todo)} to run | Seeds/config: {len(seeds)}")
    print(f"Timeout protection: {format_timeout_duration(7200)} per seed\n")

    batch_size = p8.DATASET_PARAMS[p8.DATASET_OPPORTUNITY]["batch_size"]
    spawn_ctx = p8.mp.get_context("spawn")
    had_failures = False

    # Persist resumed configs immediately so a crash before the first new config
    # finishes does not lose the resume baseline.
    with open(partial_path, "w") as f:
        json.dump(all_results, f)

    for idx, exp in enumerate(todo, start=1):
        print(
            f"\n[{idx}/{len(todo)}] VFL {exp['fusion']} eps={exp['dp_epsilon']} "
            f"Attack={exp['attack_type']} Ratio={exp['attack_ratio']} Defense={exp['defense']}",
            flush=True,
        )
        tmp_path = f"/tmp/exp10_{idx}_{os.getpid()}.json"
        subprocess_timeout = len(seeds) * 7200 + 300
        proc = spawn_ctx.Process(target=_config_worker, args=(exp, seeds, tmp_path, batch_size))
        proc.start()
        for seed in seeds:
            print(f"  Seed {seed}", end=" ", flush=True)
        proc.join(timeout=subprocess_timeout)

        if proc.is_alive():
            print(f"\n  [SUBPROCESS TIMEOUT after {subprocess_timeout}s]", flush=True)
            proc.terminate()
            proc.join()
            seed_results = [[] for _ in seeds]
        elif os.path.exists(tmp_path):
            with open(tmp_path) as f:
                seed_results = json.load(f)
            os.unlink(tmp_path)
            for seed_result in seed_results:
                if seed_result:
                    print(f" F1={seed_result[-1]['f1']:.4f}", end="", flush=True)
            print()
        else:
            print("\n  [SUBPROCESS FAILED — no result file]", flush=True)
            seed_results = [[] for _ in seeds]

        if any(not seed_result for seed_result in seed_results):
            had_failures = True

        all_results.append({
            "config": {
                **exp,
                "topology": "VFL",
                "aggregation": "coordinator",
            },
            "runs": seed_results,
        })

        with open(partial_path, "w") as f:
            json.dump(all_results, f)

        gc.collect()
        torch.cuda.empty_cache()

    with open(outfile, "w") as f:
        json.dump(all_results, f)
    print(f"\nExp10 done. Results saved to {outfile}")
    if had_failures:
        print("Exp10 run failed: at least one config/seed produced no metrics.", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
