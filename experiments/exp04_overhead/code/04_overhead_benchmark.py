#!/usr/bin/env python3
# 04_overhead_benchmark.py
# Exp04 — Operational Overhead Benchmark: HFL vs VFL × Intermediate/Late × ε={0, 3.0}
# Measures per-round wall_time_s, gpu_mem_allocated_mb, gpu_mem_peak_mb, num_params.
# Fixed 10 rounds, no early stop, no attack, single seed (42), OPPORTUNITY only.
# Output: results/exp04_overhead_benchmark.csv

import os
import sys
import gc
import copy
import time
import csv
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

warnings.filterwarnings('ignore')

from pathlib import Path
SHARED_CODE = Path(__file__).resolve().parents[2] / "shared" / "code"
if str(SHARED_CODE) not in sys.path:
    sys.path.insert(0, str(SHARED_CODE))
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
    attach_dp_to_fusion_head,
    manual_dp_fusion_step,
)

# --- CONFIG ---
NUM_ROUNDS    = 10
LOCAL_EPOCHS  = 3
DATA_ROOT     = "./data"
SEED          = 42
NUM_CLIENTS   = 12
DP_DELTA      = 1e-5
NUM_WORKERS   = 0

DATASET_PARAMS = {
    "lr": 0.01, "momentum": 0.9, "batch_size": 16, "dropout": 0.5
}

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

CONFIGS = [
    {"topology": "HFL", "fusion": "Intermediate", "epsilon": 0.0},
    {"topology": "HFL", "fusion": "Intermediate", "epsilon": 3.0},
    {"topology": "HFL", "fusion": "Late",         "epsilon": 0.0},
    {"topology": "HFL", "fusion": "Late",         "epsilon": 3.0},
    {"topology": "VFL", "fusion": "Intermediate", "epsilon": 0.0},
    {"topology": "VFL", "fusion": "Intermediate", "epsilon": 3.0},
    {"topology": "VFL", "fusion": "Late",         "epsilon": 0.0},
    {"topology": "VFL", "fusion": "Late",         "epsilon": 3.0},
]


# --- REPRODUCIBILITY ---
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True


# --- DATA ---
def download_opportunity(root):
    extract_path = os.path.join(root, "OpportunityUCIDataset")
    if os.path.exists(extract_path):
        return
    zip_path = os.path.join(root, "OpportunityUCIDataset.zip")
    if os.path.exists(zip_path):
        print("Extracting OpportunityUCIDataset.zip ...")
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall(root)
    else:
        raise FileNotFoundError(f"OpportunityUCIDataset.zip not found at {zip_path}")


def load_opportunity_subject(root, subject_id, runs=('Drill', 'ADL1', 'ADL2')):
    base = os.path.join(root, "OpportunityUCIDataset", "dataset")
    all_body, all_obj, all_amb, all_labels = [], [], [], []
    for run in runs:
        filepath = os.path.join(base, f"S{subject_id}-{run}.dat")
        if not os.path.exists(filepath):
            continue
        try:
            df = pd.read_csv(filepath, sep=r'\s+', header=None, engine='python')
            df = df.interpolate(method='linear', limit_direction='forward').fillna(0)
            data = df.values
        except Exception:
            continue
        body   = data[:, 1:31]
        obj    = data[:, 50:65]
        amb    = data[:, 100:110]
        labels = encode_opportunity_labels(data[:, OPPORTUNITY_LABEL_COLUMN] if data.shape[1] > OPPORTUNITY_LABEL_COLUMN else np.zeros(len(data)))
        window_size, step = 30, 15
        for i in range(0, len(data) - window_size, step):
            all_body.append(body[i:i+window_size].T)
            all_obj.append(obj[i:i+window_size].T)
            all_amb.append(amb[i:i+window_size].T)
            all_labels.append(int(labels[i + window_size // 2]))
    if not all_body:
        return None
    return (np.array(all_body, dtype=np.float32),
            np.array(all_obj,  dtype=np.float32),
            np.array(all_amb,  dtype=np.float32),
            np.array(all_labels, dtype=np.int64))


class OppHorizontalDataset(Dataset):
    def __init__(self, body, obj, amb, labels):
        self.body   = torch.from_numpy(body)
        self.obj    = torch.from_numpy(obj)
        self.amb    = torch.from_numpy(amb)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'body_sensors':   self.body[idx],
            'object_sensors': self.obj[idx],
            'ambient_sensors': self.amb[idx],
        }, self.labels[idx]


def partition_opportunity_horizontal(num_clients, batch_size):
    download_opportunity(DATA_ROOT)
    all_body, all_obj, all_amb, all_labels = [], [], [], []
    for sid in range(1, 5):
        r = load_opportunity_subject(DATA_ROOT, sid)
        if r:
            all_body.append(r[0]); all_obj.append(r[1])
            all_amb.append(r[2]); all_labels.append(r[3])
    if not all_body:
        raise RuntimeError("No OPPORTUNITY data loaded")
    body   = np.concatenate(all_body)
    obj    = np.concatenate(all_obj)
    amb    = np.concatenate(all_amb)
    labels = np.concatenate(all_labels)
    split  = int(len(labels) * 0.8)
    train_ds = OppHorizontalDataset(body[:split], obj[:split], amb[:split], labels[:split])
    test_ds  = OppHorizontalDataset(body[split:], obj[split:], amb[split:], labels[split:])
    n = len(train_ds)
    sizes = [n // num_clients] * num_clients
    sizes[-1] += n - sum(sizes)
    offsets = [0] + list(np.cumsum(sizes[:-1]))
    client_loaders = [
        DataLoader(
            torch.utils.data.Subset(train_ds, range(offsets[i], offsets[i] + sizes[i])),
            batch_size=batch_size, shuffle=True, num_workers=NUM_WORKERS, drop_last=True
        )
        for i in range(num_clients)
    ]
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    return client_loaders, test_loader


class OppVerticalSiloDataset(Dataset):
    def __init__(self, sensor_data, labels):
        self.data   = torch.from_numpy(sensor_data)
        self.labels = torch.from_numpy(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


def partition_opportunity_vertical(batch_size):
    download_opportunity(DATA_ROOT)
    all_upper, all_lower, all_obj, all_amb, all_labels = [], [], [], [], []
    for sid in range(1, 5):
        r = load_opportunity_subject(DATA_ROOT, sid)
        if r:
            body, obj, amb, labels = r
            all_upper.append(body[:, :15, :])
            all_lower.append(body[:, 15:, :])
            all_obj.append(obj)
            all_amb.append(amb)
            all_labels.append(labels)
    if not all_upper:
        raise RuntimeError("No OPPORTUNITY data loaded")
    upper  = np.concatenate(all_upper)
    lower  = np.concatenate(all_lower)
    obj    = np.concatenate(all_obj)
    amb    = np.concatenate(all_amb)
    labels = np.concatenate(all_labels)
    split  = int(len(labels) * 0.8)
    silo_names   = ['body_upper', 'body_lower', 'objects', 'ambient']
    silo_train   = [upper[:split], lower[:split], obj[:split], amb[:split]]
    silo_test    = [upper[split:], lower[split:], obj[split:],  amb[split:]]
    def make_loader(data, lbl, shuffle):
        return DataLoader(OppVerticalSiloDataset(data, lbl),
                          batch_size=batch_size, shuffle=shuffle,
                          num_workers=NUM_WORKERS, drop_last=shuffle)
    # VFL batches must remain index-aligned across silos and labels.
    silo_loaders      = {n: make_loader(d, labels[:split], False) for n, d in zip(silo_names, silo_train)}
    test_silo_loaders = {n: make_loader(d, labels[split:], False) for n, d in zip(silo_names, silo_test)}
    label_loader      = DataLoader(TensorDataset(torch.from_numpy(labels[:split])),
                                   batch_size=batch_size, shuffle=False,
                                   num_workers=NUM_WORKERS, drop_last=False)
    test_label_loader = DataLoader(TensorDataset(torch.from_numpy(labels[split:])),
                                   batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS)
    return silo_loaders, label_loader, test_silo_loaders, test_label_loader


# --- MODELS (copied from exp01 v24) ---
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
            nn.Dropout(dropout_p),
        )
    def forward(self, x): return self.conv(x)


class IntermediateFusionModel(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        enc_d = dropout_p * 0.3
        self.body_net = DeepConvEncoder(30, enc_d)
        self.obj_net  = DeepConvEncoder(15, enc_d)
        self.amb_net  = DeepConvEncoder(10, enc_d)
        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 13 * 3, 256), nn.ReLU(),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(256, 18),
        )
    def forward(self, x):
        return self.cls(torch.cat([
            self.body_net(x['body_sensors']),
            self.obj_net(x['object_sensors']),
            self.amb_net(x['ambient_sensors']),
        ], dim=1))


class LateFusionModel(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.body_head = nn.Sequential(DeepConvEncoder(30, dropout_p), nn.Flatten(), nn.Linear(64*13, 18))
        self.obj_head  = nn.Sequential(DeepConvEncoder(15, dropout_p), nn.Flatten(), nn.Linear(64*13, 18))
        self.amb_head  = nn.Sequential(DeepConvEncoder(10, dropout_p), nn.Flatten(), nn.Linear(64*13, 18))
    def forward(self, x):
        return (self.body_head(x['body_sensors']) +
                self.obj_head(x['object_sensors']) +
                self.amb_head(x['ambient_sensors'])) / 3.0


class SiloEncoder(nn.Module):
    def __init__(self, in_channels, dropout_p=0.5):
        super().__init__()
        self.encoder = DeepConvEncoder(in_channels, dropout_p)
    def forward(self, x): return self.encoder(x)


class IntermediateFusionVFL(nn.Module):
    def __init__(self, num_silos=4, emb=832, num_classes=18, dropout_p=0.5):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_silos * emb, 256), nn.ReLU(),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(256, num_classes),
        )
    def forward(self, embs): return self.classifier(torch.cat([e.flatten(1) for e in embs], dim=1))


class LateFusionVFL(nn.Module):
    def __init__(self): super().__init__()
    def forward(self, logits): return torch.mean(torch.stack(logits), dim=0)


# --- FL HELPERS ---
class FederatedClient:
    def __init__(self, client_id, loader, device, params):
        self.client_id = client_id
        self.loader    = loader
        self.device    = device
        self.params    = params

    def train(self, global_model, epochs, use_dp, noise_multiplier):
        model = copy.deepcopy(global_model).to(self.device)
        model.train()
        optimizer = optim.SGD(model.parameters(), lr=self.params['lr'], momentum=self.params['momentum'])
        privacy_engine = None
        if use_dp:
            privacy_engine = PrivacyEngine()
            model, optimizer, loader = privacy_engine.make_private(
                module=model, optimizer=optimizer, data_loader=self.loader,
                noise_multiplier=noise_multiplier, max_grad_norm=5.0,
            )
        else:
            loader = self.loader
        for _ in range(epochs):
            for data, target in loader:
                target = target.to(self.device)
                data   = {k: v.to(self.device) for k, v in data.items()} if isinstance(data, dict) else data.to(self.device)
                optimizer.zero_grad()
                loss = F.cross_entropy(model(data), target)
                loss.backward()
                if not use_dp:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
        state = model.state_dict()
        if use_dp:
            state = {k.replace('_module.', ''): v for k, v in state.items()}
            privacy_engine._module = None
        return state, len(self.loader.dataset)


def fedavg(weights, sizes):
    total = sum(sizes)
    agg   = {k: torch.zeros_like(v) for k, v in weights[0].items()}
    for w, s in zip(weights, sizes):
        for k in agg: agg[k] += w[k] * (s / total)
    return agg


def evaluate_hfl(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data, target in loader:
            target = target.to(device)
            data   = {k: v.to(device) for k, v in data.items()} if isinstance(data, dict) else data.to(device)
            preds.extend(model(data).argmax(1).cpu().numpy())
            targets.extend(target.cpu().numpy())
    return f1_score(targets, preds, average='macro', zero_division=0)


class VerticalSilo:
    def __init__(self, sid, name, in_ch, device, params, fusion):
        self.name          = name
        self.device        = device
        self.fusion        = fusion
        self.encoder       = SiloEncoder(in_ch, params['dropout']).to(device)
        if fusion == 'Late':
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64*13, 18)).to(device)
        self.optimizer     = optim.SGD(self.encoder.parameters(), lr=params['lr'], momentum=params['momentum'])
        self.privacy_engine = None

    def make_private(self, loader, noise_multiplier):
        self.encoder.train()
        pe = PrivacyEngine()
        self.encoder, self.optimizer, _ = pe.make_private(
            module=self.encoder, optimizer=self.optimizer, data_loader=loader,
            noise_multiplier=noise_multiplier, max_grad_norm=5.0,
        )
        self.privacy_engine = pe

    def forward(self, x):
        emb = self.encoder(x)
        if self.fusion == 'Late':
            return self.classifier(emb)
        return emb

    def cleanup(self):
        if self.privacy_engine is not None:
            self.privacy_engine = None
        self.encoder.cpu()
        del self.optimizer


class VerticalCoordinator:
    def __init__(self, silos, fusion_head, device, params):
        self.silos      = silos
        self.head       = fusion_head.to(device)
        self.device     = device
        self.optimizer  = optim.SGD(fusion_head.parameters(), lr=params['lr'], momentum=params['momentum']) if list(fusion_head.parameters()) else None
        self.fusion_head_dp_enabled = False
        self.fusion_head_dp_noise_multiplier = 0.0

    def make_fusion_head_private(self, label_loader, noise_multiplier):
        # Manual DP-SGD on the fusion head (here named self.head): wrap with
        # GradSampleModule and apply clip+Gaussian noise in train_round via
        # manual_dp_fusion_step.
        if self.optimizer is None:
            return
        self.head = attach_dp_to_fusion_head(self.head)
        self.fusion_head_dp_enabled = True
        self.fusion_head_dp_noise_multiplier = noise_multiplier

    def train_round(self, silo_loaders, label_loader, epochs):
        self.head.train()
        for s in self.silos.values(): s.encoder.train()
        for _ in range(epochs):
            silo_iters = {n: iter(l) for n, l in silo_loaders.items()}
            label_iter = iter(label_loader)
            try:
                while True:
                    batches = {n: next(silo_iters[n])[0].to(self.device) for n in self.silos}
                    labels  = next(label_iter)[0].to(self.device)
                    if self.optimizer: self.optimizer.zero_grad()
                    for s in self.silos.values(): s.optimizer.zero_grad()
                    embs   = [self.silos[n].forward(batches[n]) for n in self.silos]
                    output = self.head(embs)
                    loss   = F.cross_entropy(output, labels)
                    loss.backward()
                    if self.optimizer:
                        if self.fusion_head_dp_enabled:
                            manual_dp_fusion_step(
                                self.head,
                                max_grad_norm=5.0,
                                noise_multiplier=self.fusion_head_dp_noise_multiplier,
                            )
                        else:
                            torch.nn.utils.clip_grad_norm_(self.head.parameters(), max_norm=5.0)
                    for s in self.silos.values():
                        if not s.privacy_engine:
                            torch.nn.utils.clip_grad_norm_(s.encoder.parameters(), max_norm=5.0)
                    if self.optimizer: self.optimizer.step()
                    for s in self.silos.values(): s.optimizer.step()
            except StopIteration:
                pass

    def evaluate(self, test_silo_loaders, test_label_loader):
        self.head.eval()
        for s in self.silos.values(): s.encoder.eval()
        preds, targets = [], []
        silo_iters = {n: iter(l) for n, l in test_silo_loaders.items()}
        label_iter = iter(test_label_loader)
        with torch.no_grad():
            try:
                while True:
                    batches = {n: next(silo_iters[n])[0].to(self.device) for n in self.silos}
                    labels  = next(label_iter)[0].to(self.device)
                    embs    = [self.silos[n].forward(batches[n]) for n in self.silos]
                    output  = self.head(embs)
                    preds.extend(output.argmax(1).cpu().numpy())
                    targets.extend(labels.cpu().numpy())
            except StopIteration:
                pass
        return f1_score(targets, preds, average='macro', zero_division=0)


# --- MEMORY UTILS ---
def gpu_mem_mb():
    if not torch.cuda.is_available():
        return 0.0, 0.0
    allocated = torch.cuda.memory_allocated(DEVICE) / 1024**2
    peak      = torch.cuda.max_memory_allocated(DEVICE) / 1024**2
    return round(allocated, 2), round(peak, 2)


def count_params(*models):
    total = 0
    for m in models:
        if isinstance(m, nn.Module):
            total += sum(p.numel() for p in m.parameters())
        elif hasattr(m, 'encoder'):
            total += sum(p.numel() for p in m.encoder.parameters())
            if hasattr(m, 'classifier'):
                total += sum(p.numel() for p in m.classifier.parameters())
    return total


# --- BENCHMARK RUNNERS ---
def benchmark_hfl(fusion, epsilon, client_loaders, test_loader):
    set_seed(SEED)
    params = DATASET_PARAMS.copy()
    if fusion == 'Intermediate':
        params['lr'] *= 0.3
    use_dp = epsilon > 0
    noise_multiplier, _, _ = dp_plan_from_loaders(
        epsilon, client_loaders, params['batch_size'], NUM_ROUNDS, LOCAL_EPOCHS, DP_DELTA
    )

    global_model = (IntermediateFusionModel(params['dropout']) if fusion == 'Intermediate'
                    else LateFusionModel(params['dropout'])).to(DEVICE)
    n_params = count_params(global_model)
    clients  = [FederatedClient(i, l, DEVICE, params) for i, l in enumerate(client_loaders)]

    rows = []
    for rnd in range(NUM_ROUNDS):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(DEVICE)
        t0 = time.perf_counter()
        ws, ss = [], []
        for c in clients:
            w, s = c.train(global_model, LOCAL_EPOCHS, use_dp, noise_multiplier)
            ws.append(w); ss.append(s)
        global_model.load_state_dict(fedavg(ws, ss))
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        wall = round(time.perf_counter() - t0, 3)
        f1   = evaluate_hfl(global_model, test_loader, DEVICE)
        alloc, peak = gpu_mem_mb()
        rows.append({
            'topology': 'HFL', 'fusion': fusion, 'epsilon': epsilon,
            'round': rnd + 1, 'wall_time_s': wall,
            'gpu_mem_allocated_mb': alloc, 'gpu_mem_peak_mb': peak,
            'num_params': n_params, 'f1': round(f1, 4),
        })
        print(f"  R{rnd+1:02d}: {wall:.1f}s  mem={alloc:.0f}/{peak:.0f}MB  F1={f1:.4f}")
    return rows


def benchmark_vfl(fusion, epsilon, silo_loaders, label_loader, test_silo_loaders, test_label_loader):
    set_seed(SEED)
    params = DATASET_PARAMS.copy()
    if fusion == 'Intermediate':
        params['lr'] *= 0.3
    use_dp = epsilon > 0
    noise_multiplier, _, _ = dp_plan_from_loaders(
        epsilon, list(silo_loaders.values()), params['batch_size'], NUM_ROUNDS, LOCAL_EPOCHS, DP_DELTA,
        mechanisms_per_step=len(silo_loaders) + 1,
    )

    silo_defs = {'body_upper': 15, 'body_lower': 15, 'objects': 15, 'ambient': 10}
    silos = {n: VerticalSilo(i, n, ch, DEVICE, params, fusion)
             for i, (n, ch) in enumerate(silo_defs.items())}
    if use_dp:
        for s in silos.values():
            s.make_private(silo_loaders[s.name], noise_multiplier)
    fusion_head = (IntermediateFusionVFL(dropout_p=params['dropout']) if fusion == 'Intermediate'
                   else LateFusionVFL())
    coordinator = VerticalCoordinator(silos, fusion_head, DEVICE, params)
    if use_dp:
        coordinator.make_fusion_head_private(label_loader, noise_multiplier)
    n_params = count_params(fusion_head, *silos.values())

    rows = []
    for rnd in range(NUM_ROUNDS):
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(DEVICE)
        t0 = time.perf_counter()
        coordinator.train_round(silo_loaders, label_loader, LOCAL_EPOCHS)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        wall = round(time.perf_counter() - t0, 3)
        f1   = coordinator.evaluate(test_silo_loaders, test_label_loader)
        alloc, peak = gpu_mem_mb()
        rows.append({
            'topology': 'VFL', 'fusion': fusion, 'epsilon': epsilon,
            'round': rnd + 1, 'wall_time_s': wall,
            'gpu_mem_allocated_mb': alloc, 'gpu_mem_peak_mb': peak,
            'num_params': n_params, 'f1': round(f1, 4),
        })
        print(f"  R{rnd+1:02d}: {wall:.1f}s  mem={alloc:.0f}/{peak:.0f}MB  F1={f1:.4f}")

    for s in silos.values():
        s.cleanup()
    del coordinator, silos, fusion_head
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows


# --- MAIN ---
if __name__ == "__main__":
    print(f"Exp04 Overhead Benchmark — {os.uname().nodename}")
    print(f"Device: {DEVICE}  |  Rounds: {NUM_ROUNDS}  |  Seed: {SEED}")
    print(f"Configs: {len(CONFIGS)}\n")
    os.makedirs("results", exist_ok=True)

    # Load data once
    batch_size = DATASET_PARAMS['batch_size']
    print("Loading OPPORTUNITY data...")
    client_loaders, test_loader = partition_opportunity_horizontal(NUM_CLIENTS, batch_size)
    silo_loaders, label_loader, test_silo_loaders, test_label_loader = partition_opportunity_vertical(batch_size)
    print("Data loaded.\n")

    all_rows = []
    csv_path = "results/exp04_overhead_benchmark.csv"
    fieldnames = ['topology', 'fusion', 'epsilon', 'round',
                  'wall_time_s', 'gpu_mem_allocated_mb', 'gpu_mem_peak_mb',
                  'num_params', 'f1']

    for i, cfg in enumerate(CONFIGS):
        topo, fusion, eps = cfg['topology'], cfg['fusion'], cfg['epsilon']
        print(f"[{i+1}/{len(CONFIGS)}] {topo} {fusion} ε={eps}")
        if topo == 'HFL':
            rows = benchmark_hfl(fusion, eps, client_loaders, test_loader)
        else:
            rows = benchmark_vfl(fusion, eps, silo_loaders, label_loader,
                                 test_silo_loaders, test_label_loader)
        all_rows.extend(rows)
        # Incremental save
        with open(csv_path, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader(); w.writerows(all_rows)
        avg_wall = np.mean([r['wall_time_s'] for r in rows])
        print(f"  → avg {avg_wall:.1f}s/round  saved to {csv_path}\n")

    print(f"DONE. {len(all_rows)} rows saved to {csv_path}")
