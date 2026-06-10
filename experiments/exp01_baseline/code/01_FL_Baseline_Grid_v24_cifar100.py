#!/usr/bin/env python3
# 01_FL_Baseline_Grid_v24_cifar100.py
# CIFAR-100 replication of Exp01 (Botacin checklist item #13)
# Includes: CIFAR-100 Horizontal Cross-Device only (Eps=[0,1,5,8])
# Goal: compare DP-induced F1 drop vs CIFAR-10 baseline to disentangle
#       privacy cost from task difficulty.
# Features:
# - Same FL setup as part1 (NUM_ROUNDS=25, LOCAL_EPOCHS=3, 3 seeds, 10 clients IID)
# - GroupNorm (DP-compatible), Opacus DP
# - Output classifier: 100 classes (vs 10 in CIFAR-10 script)

import os
import sys
import gc
import copy
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from opacus import PrivacyEngine
from sklearn.metrics import f1_score, accuracy_score
import warnings

# Timeout protection for Grid experiments
from pathlib import Path
SHARED_CODE = Path(__file__).resolve().parents[2] / "shared" / "code"
if str(SHARED_CODE) not in sys.path:
    sys.path.insert(0, str(SHARED_CODE))

from timeout_utils import run_with_timeout, format_timeout_duration

warnings.filterwarnings('ignore')

from experiment_validity import (
    OPPORTUNITY_LABEL_COLUMN,
    encode_opportunity_labels,
    opportunity_num_classes,
    participant_sample_rate,
    calibrated_noise_multiplier,
    composed_epsilon,
    dp_plan_from_loaders,
    select_malicious_indices,
    label_flip_ratio_for_vfl,
)

# --- CONFIGURATION ---
NUM_ROUNDS = 25
LOCAL_EPOCHS = 3
DATA_ROOT = "./data"
NUM_SEEDS = 3
SEEDS = [42, 123, 456]

# Client Configuration
NUM_CLIENTS_CIFAR = 10

# Hyperparameters (kept identical to CIFAR-10 part1 for direct comparability)
DATASET_PARAMS = {
    "CIFAR100": {
        "lr": 0.01,
        "momentum": 0.9,
        "batch_size": 16,
        "dropout": 0.25
    }
}

DATASET_CIFAR100 = "CIFAR100"
NUM_CLASSES_CIFAR100 = 100
NUM_WORKERS = 4
DP_DELTA = 1e-5

# Early Stopping Configuration (mirrors part1)
EARLY_STOP_LOSS_THRESHOLD = 1e6
EARLY_STOP_PATIENCE = 3
WARMUP_ROUNDS = 5
DP_EPSILONS = [0.0, 1.0, 5.0, 8.0]

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

# --- PARTITIONING ---
def partition_cifar100_iid(num_clients, batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    try:
        trainset = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=True, download=False, transform=transform)
        testset = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=False, download=False, transform=transform)
    except:
        trainset = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR100(root=DATA_ROOT, train=False, download=True, transform=transform)

    total_samples = len(trainset)
    indices = np.random.permutation(total_samples)
    client_indices = np.array_split(indices, num_clients)

    client_loaders = [DataLoader(Subset(trainset, idx), batch_size=batch_size, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=False) for idx in client_indices]
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    return client_loaders, test_loader

# --- MODELS ---
# Same architecture as part1 SimpleCNN_GN, but final FC sized for 100 classes.
class SimpleCNN_GN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES_CIFAR100, dropout_p=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1); self.gn1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1); self.gn2 = nn.GroupNorm(8, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1); self.gn3 = nn.GroupNorm(8, 64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1); self.gn4 = nn.GroupNorm(8, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64*8*8, 256); self.fc2 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, x):
        if isinstance(x, dict): x = x['image']
        x = self.pool1(F.relu(self.gn2(self.conv2(F.relu(self.gn1(self.conv1(x)))))))
        x = self.pool2(F.relu(self.gn4(self.conv4(F.relu(self.gn3(self.conv3(x)))))))
        return self.fc2(self.dropout(F.relu(self.fc1(torch.flatten(x, 1)))))

# --- CLASSES FOR FL ---
class FederatedClient:
    def __init__(self, client_id, train_loader, device, params):
        self.client_id = client_id; self.train_loader = train_loader; self.device = device; self.params = params
        self.model = None; self.privacy_engine = None

    def train(self, global_model, local_epochs, use_dp=False, noise_multiplier=1.0):
        self.model = copy.deepcopy(global_model).to(self.device); self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.params['lr'], momentum=self.params['momentum'])
        if use_dp:
            self.privacy_engine = PrivacyEngine()
            self.model, optimizer, train_loader_private = self.privacy_engine.make_private(
                module=self.model, optimizer=optimizer, data_loader=self.train_loader,
                noise_multiplier=noise_multiplier, max_grad_norm=5.0)
            loader = train_loader_private
        else:
            loader = self.train_loader

        epoch_loss = 0
        for epoch in range(local_epochs):
            for data, target in loader:
                target = target.to(self.device)
                data = {k: v.to(self.device) for k, v in data.items()} if isinstance(data, dict) else data.to(self.device)
                optimizer.zero_grad()
                loss = F.cross_entropy(self.model(data), target)
                loss.backward()
                if not use_dp:  # Manual clipping when not using DP (DP applies its own)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += loss.item()

        epsilon = self.privacy_engine.get_epsilon(delta=DP_DELTA) if use_dp else 0.0
        state_dict = self.model.state_dict()
        if use_dp: state_dict = {k.replace('_module.', ''): v for k, v in state_dict.items()}
        return state_dict, len(self.train_loader.dataset), epoch_loss / local_epochs, epsilon

def fedavg_aggregate(client_weights, client_sizes):
    total_size = sum(client_sizes); aggregated = {k: torch.zeros_like(v) for k, v in client_weights[0].items()}
    for cw, csize in zip(client_weights, client_sizes):
        for k in aggregated.keys(): aggregated[k] += cw[k] * (csize / total_size)
    return aggregated

def evaluate(model, loader, device):
    model.eval(); all_preds, all_targets = [], []
    with torch.no_grad():
        for data, target in loader:
            target = target.to(device)
            data = {k: v.to(device) for k, v in data.items()} if isinstance(data, dict) else data.to(device)
            all_preds.extend(model(data).argmax(dim=1).cpu().numpy()); all_targets.extend(target.cpu().numpy())
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return acc, f1, all_preds

# --- RUNNERS ---
def run_horizontal_fl(dataset_name, dp_epsilon, seed, num_rounds=NUM_ROUNDS, epochs=LOCAL_EPOCHS):
    set_seed(seed)
    params = DATASET_PARAMS[dataset_name].copy()

    client_loaders, test_loader = partition_cifar100_iid(NUM_CLIENTS_CIFAR, params['batch_size'])
    use_dp = dp_epsilon > 0
    noise_multiplier, dp_sample_rate, dp_steps_per_round = dp_plan_from_loaders(
        dp_epsilon, client_loaders, params['batch_size'], num_rounds, epochs, DP_DELTA
    )
    global_model = SimpleCNN_GN(num_classes=NUM_CLASSES_CIFAR100, dropout_p=params['dropout']).to(DEVICE)

    clients = [FederatedClient(i, loader, DEVICE, params) for i, loader in enumerate(client_loaders)]
    round_metrics = []

    best_f1 = 0.0
    best_model_state = None
    rounds_without_improvement = 0

    for round_idx in range(num_rounds):
        client_weights, client_sizes, client_losses, client_eps = [], [], [], []
        for client in clients:
            w, s, l, e = client.train(global_model, epochs, use_dp=use_dp, noise_multiplier=noise_multiplier)
            client_weights.append(w); client_sizes.append(s); client_losses.append(l); client_eps.append(e)

        avg_loss = np.mean(client_losses)

        if avg_loss > EARLY_STOP_LOSS_THRESHOLD:
            print(f" [EARLY STOP: Loss={avg_loss:.2e} > threshold] ", end='')
            break

        global_model.load_state_dict(fedavg_aggregate(client_weights, client_sizes))
        gc.collect(); torch.cuda.empty_cache()

        acc, f1, preds = evaluate(global_model, test_loader, DEVICE)
        epsilon_spent = composed_epsilon(
            noise_multiplier, dp_sample_rate, (round_idx + 1) * dp_steps_per_round, DP_DELTA
        ) if use_dp else 0.0
        round_metrics.append({'round': round_idx+1, 'loss': avg_loss, 'acc': acc, 'f1': f1, 'epsilon': epsilon_spent})

        if f1 > best_f1:
            best_f1 = f1
            best_model_state = copy.deepcopy(global_model.state_dict())
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1

        if rounds_without_improvement >= EARLY_STOP_PATIENCE and round_idx >= 10:
            print(f" [EARLY STOP: No improvement for {EARLY_STOP_PATIENCE} rounds] ", end='')
            break

        if (round_idx + 1) % 5 == 0: print(f" (Loss: {avg_loss:.3f}) ", end='', flush=True)
        else: print('.', end='', flush=True)

    if best_model_state is not None:
        global_model.load_state_dict(best_model_state)
        acc, f1, preds = evaluate(global_model, test_loader, DEVICE)
        round_metrics.append({'round': len(round_metrics)+1,
                             'loss': round_metrics[-1]['loss'] if round_metrics else 0,
                             'acc': acc, 'f1': f1,
                             'epsilon': round_metrics[-1]['epsilon'] if round_metrics else 0,
                             'note': 'best_checkpoint'})

    return round_metrics

# --- MAIN CONTROL ---
def run_smoke_test():
    print("\n" + "="*60)
    print(">> PRE-FLIGHT SMOKE TEST (1 Round, 1 Epoch)")
    print("="*60)

    configs = [
        {'dataset': DATASET_CIFAR100, 'epsilon': 1.0},
    ]

    for cfg in configs:
        print(f"Testing: {cfg['dataset']} Eps={cfg['epsilon']}", end=' ')
        try:
            run_horizontal_fl(cfg['dataset'], cfg['epsilon'], 42, 1, 1)
            print(" -> OK")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f" -> FAIL: {e}")
            sys.exit(1)
    print(">> SMOKE TEST PASSED\n")

if __name__ == "__main__":
    print(f"Starting Grid Job (CIFAR-100 replication) on {os.uname().nodename}")
    print(f"Device: {DEVICE}")

    # 1. Smoke Test
    run_smoke_test()

    # 2. Main Experiments: CIFAR-100 Horizontal x 4 epsilons x 3 seeds = 12 runs
    experiments = [{'dataset': DATASET_CIFAR100, 'topo': 'horizontal', 'fusion': 'Horizontal', 'eps': eps}
                   for eps in DP_EPSILONS]

    print(f"\n=== CIFAR-100 Horizontal Cross-Device (Eps={DP_EPSILONS}) v24 - "
          f"{len(experiments)} configs x {NUM_SEEDS} seeds ===")
    print(f"⏱️  Timeout protection: {format_timeout_duration(7200)} per config\n")

    all_results = []
    for i, exp in enumerate(experiments):
        print(f"\n[{i+1}/{len(experiments)}] {exp['dataset']} {exp['fusion']} Eps={exp['eps']}")

        seed_results = []
        for seed in SEEDS:
            print(f"  Seed {seed}", end=' ', flush=True)

            def run_single_experiment():
                return run_horizontal_fl(exp['dataset'], exp['eps'], seed)

            success, res = run_with_timeout(
                run_single_experiment,
                timeout_seconds=7200,  # 2 hours
                on_timeout=lambda: print(f"\n  ⚠️  TIMEOUT after 2h - skipping to next config")
            )

            if not success:
                print(f" [TIMEOUT]")
                seed_results.append([])
                continue

            seed_results.append(res)
            if res:
                final_f1 = res[-1]['f1']
                print(f" F1={final_f1:.4f}")
            else:
                print(f" [FAILED: No rounds completed]")

        exp_data = {'config': exp, 'runs': seed_results}
        all_results.append(exp_data)

        os.makedirs("results", exist_ok=True)
        with open('results/partial_results_cifar100.json', 'w') as f: json.dump(all_results, f)

    os.makedirs("results", exist_ok=True)
    with open('results/exp01_v24_cifar100_results.json', 'w') as f: json.dump(all_results, f)
    print("\nCIFAR-100 RUN DONE. Results saved to results/exp01_v24_cifar100_results.json")
