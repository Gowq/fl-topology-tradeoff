#!/usr/bin/env python3
# 08_FL_RobustAgg_DP_Frontier_part3.py
# PART 3: VFL Coordinator + DP — Experiment 08 (Frontier regime)
# Tests: OPPORTUNITY Vertical (Intermediate/Late) with attacks + DP (ε ∈ {20.0, 100.0})
# Aggregation: Coordinator (passive defense via compartmentalization)
# Based on: exp03a part3 (já bug-free) com ε re-calibrado do Exp02 frontier.
#
# Grid: 1 agg × 2 fusion × 4 attack × 4 ratio × 2 ε = 64 configs
# Sub-parts: a (0-15), b (16-31), c (32-47), d (48-63)
# Output: results/exp08_robustagg_dp_frontier_part3{a/b/c/d}.json

import os
import sys
import gc
import copy
import json
import numpy as np
import pandas as pd
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from opacus import PrivacyEngine
from sklearn.metrics import f1_score, accuracy_score
import warnings

import multiprocessing as mp

from pathlib import Path
SHARED_CODE = Path(__file__).resolve().parents[2] / "shared" / "code"
if str(SHARED_CODE) not in sys.path:
    sys.path.insert(0, str(SHARED_CODE))

from timeout_utils import run_with_timeout, format_timeout_duration

# =============================================================================
# ATTACK UTILITIES
# =============================================================================

class AttackConfig:
    LABEL_FLIP = "label_flip"
    SIGN_FLIP = "sign_flip"
    SCALING = "scaling"
    FREE_RIDER = "free_rider"
    NONE = "none"

    ATTACK_TYPES = [LABEL_FLIP, SIGN_FLIP, SCALING, FREE_RIDER]
    VFL_ATTACK_RATIOS = [0.10, 0.25, 0.50, 0.75]

def apply_label_flipping(labels, num_classes, flip_ratio=1.0, strategy="random"):
    flipped = labels.clone()
    n_to_flip = int(len(labels) * flip_ratio)
    flip_indices = np.random.choice(len(labels), n_to_flip, replace=False)
    for idx in flip_indices:
        original = flipped[idx].item()
        flipped[idx] = np.random.choice([c for c in range(num_classes) if c != original])
    return flipped


warnings.filterwarnings('ignore')

from experiment_validity import (
    OPPORTUNITY_LABEL_COLUMN,
    encode_opportunity_labels,
    opportunity_num_classes,
    participant_sample_rate,
    calibrated_noise_multiplier,
    composed_epsilon,
    dp_plan_from_loaders,
    vfl_mechanisms_per_step,
    select_malicious_indices,
    label_flip_ratio_for_vfl,
    attach_dp_to_fusion_head,
    manual_dp_fusion_step,
)

# --- CONFIGURATION ---
NUM_ROUNDS = 25
LOCAL_EPOCHS = 3
DATA_ROOT = "./data"
NUM_SEEDS = 3
SEEDS = [42, 123, 456]

NUM_SILOS_OPP = 4

DATASET_PARAMS = {
    "OPPORTUNITY": {
        "lr": 0.01,
        "momentum": 0.9,
        "batch_size": 16,
        "dropout": 0.5
    }
}

DATASET_OPPORTUNITY = "OPPORTUNITY"
NUM_WORKERS = 0
DP_DELTA = 1e-5
NUM_CLASSES = opportunity_num_classes()

EARLY_STOP_LOSS_THRESHOLD = 1e6
EARLY_STOP_PATIENCE = 3
LOSS_STAGNATION_THRESHOLD = 0.01
LOSS_STAGNATION_PATIENCE = 3
WARMUP_ROUNDS = 5

DP_EPSILONS = [20.0, 100.0]
ATTACK_TYPES = AttackConfig.ATTACK_TYPES
ATTACK_RATIOS = AttackConfig.VFL_ATTACK_RATIOS
FUSION_MODES = ['Intermediate', 'Late']

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# --- DATASET UTILS ---
def download_opportunity(root):
    extract_path = os.path.join(root, "OpportunityUCIDataset")
    if os.path.exists(extract_path):
        return
    zip_path = os.path.join(root, "OpportunityUCIDataset.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root)


def load_opportunity_subject(root, subject_id, runs=['Drill', 'ADL1', 'ADL2']):
    base = os.path.join(root, "OpportunityUCIDataset", "dataset")
    all_body, all_obj, all_amb, all_labels = [], [], [], []

    for run in runs:
        filename = f"S{subject_id}-{run}.dat"
        filepath = os.path.join(base, filename)
        if not os.path.exists(filepath):
            continue

        try:
            df = pd.read_csv(filepath, sep='\\s+', header=None, engine='python')
            df = df.interpolate(method='linear', limit_direction='forward').fillna(0)
            data = df.values
        except:
            continue

        body = data[:, 1:31]
        obj = data[:, 50:65]
        amb = data[:, 100:110]
        labels = encode_opportunity_labels(data[:, OPPORTUNITY_LABEL_COLUMN] if data.shape[1] > OPPORTUNITY_LABEL_COLUMN else np.zeros(len(data)))

        window_size, step = 30, 15
        for i in range(0, len(data) - window_size, step):
            all_body.append(body[i:i+window_size].T)
            all_obj.append(obj[i:i+window_size].T)
            all_amb.append(amb[i:i+window_size].T)
            all_labels.append(labels[i+window_size-1])

    if len(all_labels) == 0:
        return None

    return (
        torch.tensor(np.array(all_body), dtype=torch.float32),
        torch.tensor(np.array(all_obj), dtype=torch.float32),
        torch.tensor(np.array(all_amb), dtype=torch.float32),
        torch.tensor(np.array(all_labels), dtype=torch.long)
    )


def partition_opportunity_vertical(batch_size):
    download_opportunity(DATA_ROOT)
    all_body, all_obj, all_amb, all_labels = [], [], [], []
    for subject_id in range(1, NUM_SILOS_OPP + 1):
        try:
            res = load_opportunity_subject(DATA_ROOT, subject_id)
            if res:
                b, o, a, y = res
                all_body.append(b)
                all_obj.append(o)
                all_amb.append(a)
                all_labels.append(y)
        except RuntimeError:
            continue

    if not all_body:
        return None, None, None, None

    body = torch.cat(all_body)
    obj = torch.cat(all_obj)
    amb = torch.cat(all_amb)
    labels = torch.cat(all_labels)

    num_samples = len(labels)
    shuffle_indices = torch.randperm(num_samples)
    body = body[shuffle_indices]
    obj = obj[shuffle_indices]
    amb = amb[shuffle_indices]
    labels = labels[shuffle_indices]

    silo_data = {
        "body_upper": body[:, :15, :],
        "body_lower": body[:, 15:, :],
        "objects": obj,
        "ambient": amb
    }
    silo_loaders = {
        name: DataLoader(TensorDataset(feats), batch_size=batch_size, shuffle=False,
                         num_workers=NUM_WORKERS, pin_memory=False)
        for name, feats in silo_data.items()
    }
    label_loader = DataLoader(TensorDataset(labels), batch_size=batch_size, shuffle=False)

    res_test = load_opportunity_subject(DATA_ROOT, 2, runs=["ADL4", "ADL5"])
    if res_test:
        b_t, o_t, a_t, y_t = res_test
        test_silo_data = {
            "body_upper": b_t[:, :15, :],
            "body_lower": b_t[:, 15:, :],
            "objects": o_t,
            "ambient": a_t
        }
        test_silo_loaders = {
            n: DataLoader(TensorDataset(f), batch_size=batch_size, shuffle=False)
            for n, f in test_silo_data.items()
        }
        test_label_loader = DataLoader(TensorDataset(y_t), batch_size=batch_size, shuffle=False)
        return silo_loaders, label_loader, test_silo_loaders, test_label_loader
    return None, None, None, None


# --- MODELS ---
class DeepConvEncoder(nn.Module):
    def __init__(self, in_channels, dropout_p=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 5),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )

    def forward(self, x):
        return self.conv(x)


class SiloEncoder(nn.Module):
    def __init__(self, in_channels, dropout_p=0.5):
        super().__init__()
        self.encoder = DeepConvEncoder(in_channels, dropout_p)

    def forward(self, x):
        return self.encoder(x)


class IntermediateFusionVFL(nn.Module):
    def __init__(self, num_silos=4, embedding_size_per_silo=832, num_classes=18, dropout_p=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_silos * embedding_size_per_silo, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, embeddings_list):
        return self.classifier(torch.cat([emb.flatten(1) for emb in embeddings_list], dim=1))


class LateFusionVFL(nn.Module):
    def __init__(self, num_silos=4, embedding_size_per_silo=832, num_classes=18, dropout_p=0.5):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(embedding_size_per_silo, num_classes)
            ) for _ in range(num_silos)
        ])

    def forward(self, embeddings_list):
        logits = [head(emb) for head, emb in zip(self.heads, embeddings_list)]
        return sum(logits) / len(logits)


# --- MALICIOUS VERTICAL SILO (with DP) ---
class MaliciousVerticalSilo:
    def __init__(self, silo_id, silo_name, in_channels, device, params,
                 is_malicious=False, attack_type=AttackConfig.NONE):
        self.silo_id = silo_id
        self.silo_name = silo_name
        self.device = device
        self.params = params
        self.is_malicious = is_malicious
        self.attack_type = attack_type

        self.encoder = SiloEncoder(in_channels, params['dropout']).to(device)
        self.optimizer = optim.SGD(self.encoder.parameters(), lr=params['lr'], momentum=params['momentum'])
        self.privacy_engine = None

    def make_private(self, data_loader, noise_multiplier):
        self.encoder.train()
        self.privacy_engine = PrivacyEngine()
        self.encoder, self.optimizer, _ = self.privacy_engine.make_private(
            module=self.encoder,
            optimizer=self.optimizer,
            data_loader=data_loader,
            noise_multiplier=noise_multiplier,
            max_grad_norm=5.0
        )

    def get_epsilon(self):
        return self.privacy_engine.get_epsilon(delta=DP_DELTA) if self.privacy_engine else 0.0

    def forward(self, data):
        embedding = self.encoder(data)

        if self.is_malicious and self.attack_type == AttackConfig.SIGN_FLIP:
            embedding = -embedding
        elif self.is_malicious and self.attack_type == AttackConfig.SCALING:
            embedding = embedding * 10.0
        elif self.is_malicious and self.attack_type == AttackConfig.FREE_RIDER:
            embedding = embedding * 0.0

        return embedding


class MaliciousVerticalCoordinator:
    def __init__(self, silos, fusion_head, device, params,
                 use_label_attack=False, attack_type=AttackConfig.NONE, attack_ratio=0.5):
        self.silos = silos
        self.fusion_head = fusion_head.to(device)
        self.device = device
        self.params = params
        self.use_label_attack = use_label_attack
        self.attack_type = attack_type
        self.attack_ratio = attack_ratio

        if list(fusion_head.parameters()):
            self.optimizer = optim.SGD(fusion_head.parameters(), lr=params['lr'], momentum=params['momentum'])
        else:
            self.optimizer = None
        self.fusion_head_dp_enabled = False
        self.fusion_head_dp_noise_multiplier = 0.0

    def make_fusion_head_private(self, label_loader, noise_multiplier):
        # Manual DP-SGD: wrap with GradSampleModule so per-sample grads are
        # captured on backward(); clip + Gaussian noise applied in train_round
        # via manual_dp_fusion_step. PrivacyEngine.make_private cannot wrap a
        # module whose forward consumes list[Tensor] (silo embeddings).
        if self.optimizer is None:
            return
        self.fusion_head = attach_dp_to_fusion_head(self.fusion_head)
        self.fusion_head_dp_enabled = True
        self.fusion_head_dp_noise_multiplier = noise_multiplier


    def train_round(self, silo_loaders, label_loader, epochs=1):
        self.fusion_head.train()
        for silo in self.silos.values():
            silo.encoder.train()

        total_loss = 0
        num_batches = 0

        for epoch in range(epochs):
            silo_iters = {name: iter(loader) for name, loader in silo_loaders.items()}
            label_iter = iter(label_loader)

            try:
                while True:
                    silo_batches = {name: next(silo_iters[name])[0].to(self.device) for name in self.silos.keys()}
                    labels = next(label_iter)[0].to(self.device)

                    if self.use_label_attack and self.attack_type == AttackConfig.LABEL_FLIP:
                        labels = apply_label_flipping(labels, NUM_CLASSES, flip_ratio=self.attack_ratio)
                        labels = labels.to(self.device)

                    if self.optimizer:
                        self.optimizer.zero_grad()
                    for silo in self.silos.values():
                        silo.optimizer.zero_grad()

                    embeddings = [self.silos[name].forward(batch) for name, batch in silo_batches.items()]
                    output = self.fusion_head(embeddings)
                    loss = F.cross_entropy(output, labels)
                    loss.backward()

                    # Fusion head update: manual DP-SGD if privatized, else plain clipping.
                    if self.optimizer:
                        if self.fusion_head_dp_enabled:
                            manual_dp_fusion_step(
                                self.fusion_head,
                                max_grad_norm=5.0,
                                noise_multiplier=self.fusion_head_dp_noise_multiplier,
                            )
                        else:
                            torch.nn.utils.clip_grad_norm_(self.fusion_head.parameters(), max_norm=5.0)
                    # Only clip silo gradients if NOT using DP (Opacus handles clipping internally)
                    for silo in self.silos.values():
                        if not silo.privacy_engine:
                            torch.nn.utils.clip_grad_norm_(silo.encoder.parameters(), max_norm=5.0)

                    if self.optimizer:
                        self.optimizer.step()
                    for silo in self.silos.values():
                        silo.optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1
            except StopIteration:
                pass

        return total_loss / max(num_batches, 1)

    def evaluate(self, test_silo_loaders, test_label_loader):
        self.fusion_head.eval()
        for silo in self.silos.values():
            silo.encoder.eval()

        all_preds, all_targets = [], []
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

        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        return acc, f1


# --- MAIN VFL RUNNER ---
def run_vertical_fl_robust(fusion_mode, dp_epsilon, attack_type, attack_ratio, seed,
                            silo_loaders, label_loader, test_silo_loaders, test_label_loader,
                            num_rounds=NUM_ROUNDS, epochs=LOCAL_EPOCHS):
    set_seed(seed)
    params = DATASET_PARAMS[DATASET_OPPORTUNITY].copy()

    if dp_epsilon > 0:
        params['lr'] = 0.01
    else:
        params['lr'] = 0.001

    if fusion_mode == 'Intermediate':
        params['lr'] = params['lr'] * 0.3

    use_dp = dp_epsilon > 0

    if silo_loaders is None:
        raise RuntimeError("Data Partition Failed")

    if fusion_mode == 'Intermediate':
        fusion_head = IntermediateFusionVFL(dropout_p=params['dropout'])
    elif fusion_mode == 'Late':
        fusion_head = LateFusionVFL(dropout_p=params['dropout'])
    else:
        raise ValueError(f"Unsupported fusion_mode={fusion_mode}")

    noise_multiplier, dp_sample_rate, dp_steps_per_round = dp_plan_from_loaders(
        dp_epsilon, list(silo_loaders.values()), params['batch_size'], num_rounds, epochs, DP_DELTA,
        mechanisms_per_step=vfl_mechanisms_per_step(silo_loaders, fusion_head),
    )

    silo_names = list(silo_loaders.keys())
    malicious_ids = select_malicious_indices(len(silo_names), attack_ratio, seed)
    malicious_names = [silo_names[i] for i in malicious_ids]

    silo_channels = {'body_upper': 15, 'body_lower': 15, 'objects': 15, 'ambient': 10}
    silos = {}
    for i, (name, ch) in enumerate(silo_channels.items()):
        is_mal = name in malicious_names
        silos[name] = MaliciousVerticalSilo(
            i, name, ch, DEVICE, params,
            is_malicious=is_mal,
            attack_type=attack_type if is_mal else AttackConfig.NONE
        )

    # Apply DP to each silo encoder
    if use_dp:
        for silo in silos.values():
            silo.make_private(silo_loaders[silo.silo_name], noise_multiplier=noise_multiplier)

    label_corruption_ratio = label_flip_ratio_for_vfl(attack_type, attack_ratio)
    use_label_attack = label_corruption_ratio > 0

    coordinator = MaliciousVerticalCoordinator(
        silos, fusion_head, DEVICE, params,
        use_label_attack=use_label_attack,
        attack_type=attack_type,
        attack_ratio=attack_ratio
    )
    if use_dp:
        fusion_has_params = any(p.requires_grad for p in fusion_head.parameters())
        coordinator.make_fusion_head_private(label_loader, noise_multiplier)
        assert (not fusion_has_params) or coordinator.fusion_head_dp_enabled, (
            "fusion head has trainable parameters but DP was not enabled for it"
        )

    round_metrics = []
    best_f1 = 0.0
    best_checkpoint = None
    rounds_without_improvement = 0
    prev_loss = 0.0
    loss_stagnation_count = 0

    for round_idx in range(num_rounds):
        loss = coordinator.train_round(silo_loaders, label_loader, epochs=epochs)

        if loss > EARLY_STOP_LOSS_THRESHOLD:
            print(f" [EARLY STOP: Loss={loss:.2e}] ", end='')
            break

        if round_idx > 0:
            loss_change_pct = abs(loss - prev_loss) / max(prev_loss, 1e-6)
            if loss_change_pct < LOSS_STAGNATION_THRESHOLD:
                loss_stagnation_count += 1
                if loss_stagnation_count >= LOSS_STAGNATION_PATIENCE:
                    print(f" [EARLY STOP: Loss stagnation] ", end='')
                    break
            else:
                loss_stagnation_count = 0
        prev_loss = loss

        gc.collect()
        torch.cuda.empty_cache()

        acc, f1 = coordinator.evaluate(test_silo_loaders, test_label_loader)
        spent_eps = composed_epsilon(
            noise_multiplier, dp_sample_rate, (round_idx + 1) * dp_steps_per_round, DP_DELTA
        ) if use_dp else 0.0
        round_metrics.append({
            'round': round_idx + 1,
            'loss': loss,
            'acc': acc,
            'f1': f1,
            'epsilon': spent_eps,
            'fusion_mode': fusion_mode,
            'topology': 'VFL',
            'aggregation': 'coordinator',
            'attack_type': attack_type,
            'attack_ratio': attack_ratio,
            'num_malicious': len(malicious_names),
            'malicious_silos': malicious_names
        })

        if f1 > best_f1:
            best_f1 = f1
            # Store on CPU to avoid accumulating VRAM across rounds
            best_checkpoint = {
                'fusion_head': {k: v.cpu() for k, v in coordinator.fusion_head.state_dict().items()},
                'silos': {name: {k: v.cpu() for k, v in silo.encoder.state_dict().items()}
                          for name, silo in coordinator.silos.items()}
            }
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1

        if rounds_without_improvement >= EARLY_STOP_PATIENCE and round_idx >= WARMUP_ROUNDS:
            print(f" [EARLY STOP: No improvement] ", end='')
            break

        if (round_idx + 1) % 5 == 0:
            print(f" (R{round_idx+1}: F1={f1:.4f}) ", end='', flush=True)
        else:
            print('.', end='', flush=True)

    # Restore best checkpoint
    if best_checkpoint is not None:
        coordinator.fusion_head.load_state_dict(best_checkpoint['fusion_head'])
        for name, silo in coordinator.silos.items():
            # Strip Opacus _module prefix if present
            raw_sd = best_checkpoint['silos'][name]
            clean_sd = {k.replace('_module.', ''): v for k, v in raw_sd.items()}
            try:
                silo.encoder.load_state_dict(clean_sd)
            except RuntimeError:
                silo.encoder.load_state_dict(raw_sd)
        acc, f1 = coordinator.evaluate(test_silo_loaders, test_label_loader)
        spent_eps = composed_epsilon(
            noise_multiplier, dp_sample_rate, len(round_metrics) * dp_steps_per_round, DP_DELTA
        ) if use_dp else 0.0
        round_metrics.append({
            'round': len(round_metrics) + 1,
            'loss': round_metrics[-1]['loss'] if round_metrics else 0,
            'acc': acc,
            'f1': f1,
            'epsilon': spent_eps,
            'fusion_mode': fusion_mode,
            'topology': 'VFL',
            'aggregation': 'coordinator',
            'attack_type': attack_type,
            'attack_ratio': attack_ratio,
            'num_malicious': len(malicious_names),
            'malicious_silos': malicious_names,
            'label_corruption_ratio': label_corruption_ratio,
            'note': 'best_checkpoint'
        })

    # Explicit cleanup: break Opacus reference cycles + clear grad_sample buffers
    torch.cuda.synchronize()
    for silo in silos.values():
        # Clear Opacus per-sample gradient buffers (not freed by zero_grad)
        for param in silo.encoder.parameters():
            if hasattr(param, 'grad_sample'):
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
    gc.collect()
    torch.cuda.empty_cache()

    return round_metrics


# --- SUBPROCESS WORKER (memory isolation per config) ---
def _config_worker(exp_dict, seeds, result_path, batch_size):
    """Run one config (all seeds) in an isolated subprocess — exits cleanly to reclaim RAM."""
    import gc, json, os, torch
    sl, ll, tsl, tll = partition_opportunity_vertical(batch_size)
    if sl is None:
        with open(result_path, 'w') as f:
            json.dump([[] for _ in seeds], f)
        return
    seed_results = []
    for seed in seeds:
        try:
            res = run_vertical_fl_robust(
                exp_dict['fusion'], exp_dict['dp_epsilon'],
                exp_dict['attack_type'], exp_dict['attack_ratio'],
                seed, sl, ll, tsl, tll
            )
            seed_results.append(res if res else [])
        except Exception as e:
            print(f"  [subprocess error seed={seed}]: {e}", flush=True)
            seed_results.append([])
        gc.collect()
        torch.cuda.empty_cache()
    with open(result_path, 'w') as f:
        json.dump(seed_results, f)
    gc.collect()
    torch.cuda.empty_cache()


# --- SMOKE TEST ---
def run_smoke_test():
    print("\n" + "=" * 60)
    print(">> PRE-FLIGHT SMOKE TEST (1 Round, 1 Epoch) - VFL Coordinator + DP")
    print("=" * 60)

    configs = [
        {'fusion': 'Intermediate', 'epsilon': 3.0, 'attack_type': AttackConfig.LABEL_FLIP, 'attack_ratio': 0.25},
        {'fusion': 'Late', 'epsilon': 8.0, 'attack_type': AttackConfig.SCALING, 'attack_ratio': 0.25},
        {'fusion': 'Intermediate', 'epsilon': 3.0, 'attack_type': AttackConfig.FREE_RIDER, 'attack_ratio': 0.25},
    ]

    smoke_loaders = partition_opportunity_vertical(DATASET_PARAMS[DATASET_OPPORTUNITY]['batch_size'])
    silo_loaders_s, label_loader_s, test_silo_loaders_s, test_label_loader_s = smoke_loaders

    for cfg in configs:
        print(f"Testing: VFL {cfg['fusion']} Eps={cfg['epsilon']} Attack={cfg['attack_type']} Ratio={cfg['attack_ratio']}", end=' ')
        try:
            run_vertical_fl_robust(
                cfg['fusion'], cfg['epsilon'], cfg['attack_type'], cfg['attack_ratio'], 42,
                silo_loaders_s, label_loader_s, test_silo_loaders_s, test_label_loader_s,
                num_rounds=1, epochs=1
            )
            print(" -> OK")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f" -> FAIL: {e}")
            sys.exit(1)
    del smoke_loaders, silo_loaders_s, label_loader_s, test_silo_loaders_s, test_label_loader_s
    gc.collect()
    torch.cuda.empty_cache()
    print(">> SMOKE TEST PASSED\n")


# --- MAIN ---
if __name__ == "__main__":
    print(f"Starting VFL Coordinator+DP Grid Job (Exp03a Part3) on {os.uname().nodename}")
    print(f"Device: {DEVICE}")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, default='all',
                        choices=['a', 'b', 'c', 'd', 'all'],
                        help='Sub-part: a (0-15), b (16-31), c (32-47), d (48-63), all (0-63)')
    parser.add_argument('--smoke-only', action='store_true',
                        help='Run smoke test only and exit')
    args = parser.parse_args()

    if args.smoke_only:
        run_smoke_test()
        sys.exit(0)

    if args.part in ('a', 'all'):
        run_smoke_test()

    # Build grid: 2 fusions × 4 attacks × 4 ratios × 2 epsilons = 64 configs
    experiments = []
    for fusion in FUSION_MODES:
        for attack_type in ATTACK_TYPES:
            for attack_ratio in ATTACK_RATIOS:
                for dp_epsilon in DP_EPSILONS:
                    experiments.append({
                        'fusion': fusion,
                        'attack_type': attack_type,
                        'attack_ratio': attack_ratio,
                        'dp_epsilon': dp_epsilon,
                    })

    part_slices = {'a': (0, 16), 'b': (16, 32), 'c': (32, 48), 'd': (48, 64), 'all': (0, 64)}
    start, end = part_slices[args.part]
    experiments = experiments[start:end]
    part_label = f"3{args.part.upper()}" if args.part != 'all' else "3"

    os.makedirs("results", exist_ok=True)

    total_configs = 64
    print(f"\n=== PART {part_label}: VFL Coordinator+DP v25 - Running {len(experiments)} configs ({start+1}-{end}) ===")
    print(f"Timeout protection: {format_timeout_duration(7200)} per config\n")

    _batch_size = DATASET_PARAMS[DATASET_OPPORTUNITY]['batch_size']
    _spawn_ctx = mp.get_context('spawn')

    all_results = []
    for i, exp in enumerate(experiments):
        global_idx = start + i + 1
        print(f"\n[{global_idx}/{total_configs}] VFL {exp['fusion']} eps={exp['dp_epsilon']} "
              f"Attack={exp['attack_type']} Ratio={exp['attack_ratio']}", flush=True)

        tmp_path = f'/tmp/exp08_p3{args.part}_{i}_{os.getpid()}.json'
        # 3 seeds × 2h timeout each + 5 min margin
        subprocess_timeout = len(SEEDS) * 7200 + 300

        p = _spawn_ctx.Process(
            target=_config_worker,
            args=(exp, SEEDS, tmp_path, _batch_size)
        )
        p.start()
        for seed in SEEDS:
            print(f"  Seed {seed}", end=' ', flush=True)
        p.join(timeout=subprocess_timeout)

        if p.is_alive():
            print(f"\n  [SUBPROCESS TIMEOUT after {subprocess_timeout}s]", flush=True)
            p.terminate()
            p.join()
            seed_results = [[] for _ in SEEDS]
        elif os.path.exists(tmp_path):
            with open(tmp_path) as f:
                seed_results = json.load(f)
            os.unlink(tmp_path)
            for j, sr in enumerate(seed_results):
                if sr:
                    print(f" F1={sr[-1]['f1']:.4f}", end='', flush=True)
            print()
        else:
            print(f"\n  [SUBPROCESS FAILED — no result file]", flush=True)
            seed_results = [[] for _ in SEEDS]

        exp_data = {
            'config': {
                **exp,
                'topology': 'VFL',
                'aggregation': 'coordinator'
            },
            'runs': seed_results
        }
        all_results.append(exp_data)

        with open(f'results/partial_results_exp08_p3{args.part}.json', 'w') as f:
            json.dump(all_results, f)

        gc.collect()
        torch.cuda.empty_cache()

    outfile = f'results/exp08_robustagg_dp_frontier_part3{args.part}.json'
    with open(outfile, 'w') as f:
        json.dump(all_results, f)
    print(f"\nPART {part_label} (VFL Coordinator+DP) DONE. Results saved to {outfile}")
