#!/usr/bin/env python3
# 01_FL_Baseline_Grid_v24_part2.py
# PART 2 of 2: Split for 24h grid limit
# Includes: OPPORTUNITY Vertical Cross-Silo (Early/Inter/Late, Eps=[0,1,5,8])
# Features:
# - v24: Checkpoint restoration fix, No LSTM (DP compatible), Multi-epsilon testing, Early fusion VFL
# - GroupNorm (DP-compatible), reduced dropout, lower LR, wider bottleneck
# - Opacus Differential Privacy

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
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset, TensorDataset
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
    vfl_mechanisms_per_step,
    select_malicious_indices,
    label_flip_ratio_for_vfl,
    attach_dp_to_fusion_head,
    manual_dp_fusion_step,
)

# --- CONFIGURATION ---
# v16 settings
NUM_ROUNDS = 25
LOCAL_EPOCHS = 3
DATA_ROOT = "./data"
NUM_SEEDS = 3
SEEDS = [42, 123, 456]

# Client Configuration
NUM_CLIENTS_CIFAR = 10
NUM_CLIENTS_OPP = 12  # Cross-Device
NUM_SILOS_OPP = 4

# Hyperparameters
DATASET_PARAMS = {
    "CIFAR10": {
        "lr": 0.01,
        "momentum": 0.9,
        "batch_size": 16,  # Reduced for OOM safety
        "dropout": 0.25
    },
    "OPPORTUNITY": {
        "lr": 0.01,
        "momentum": 0.9,
        "batch_size": 16, 
        "dropout": 0.5
    }
}

DATASET_CIFAR10 = "CIFAR10"
DATASET_OPPORTUNITY = "OPPORTUNITY"
NUM_WORKERS = 4
DP_DELTA = 1e-5

# Early Stopping Configuration (v24 improvements)
EARLY_STOP_LOSS_THRESHOLD = 1e6  # Stop if loss explodes
EARLY_STOP_PATIENCE = 3  # No F1 improvement for N rounds
LOSS_STAGNATION_THRESHOLD = 0.01  # < 1% loss change
LOSS_STAGNATION_PATIENCE = 3  # 3 consecutive stagnant rounds
WARMUP_ROUNDS = 5  # Relaxed from 10
DP_EPSILONS = [0.0, 1.0, 5.0, 8.0]  # v24: Test multiple epsilon values

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
    if os.path.exists(extract_path): return
    zip_path = os.path.join(root, "OpportunityUCIDataset.zip")
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(root)
    else:
        # On grid, we expect data to be present.
        print(f"WARNING: Opportunity zip not found at {zip_path}")

def load_opportunity_subject(root, subject_id, runs=['Drill', 'ADL1', 'ADL2']):
    base = os.path.join(root, "OpportunityUCIDataset", "dataset")
    all_body, all_obj, all_amb, all_labels = [], [], [], []
    
    for run in runs:
        filename = f"S{subject_id}-{run}.dat"
        filepath = os.path.join(base, filename)
        if not os.path.exists(filepath): continue
        
        try:
            df = pd.read_csv(filepath, sep='\\s+', header=None, engine='python')
            df = df.interpolate(method='linear', limit_direction='forward').fillna(0)
            data = df.values
        except: continue
        
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

class OpportunityDataset(Dataset):
    def __init__(self, body, obj, amb, labels):
        self.b, self.o, self.a, self.y = body, obj, amb, labels
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return {'body_sensors': self.b[idx], 'object_sensors': self.o[idx], 'ambient_sensors': self.a[idx]}, self.y[idx]

# --- PARTITIONING ---
def partition_cifar10_iid(num_clients, batch_size):
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,),(0.5,))])
    try:
        trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=False, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=False, transform=transform)
    except:
        trainset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=DATA_ROOT, train=False, download=True, transform=transform)
    
    total_samples = len(trainset)
    indices = np.random.permutation(total_samples)
    client_indices = np.array_split(indices, num_clients)
    
    client_loaders = [DataLoader(Subset(trainset, idx), batch_size=batch_size, shuffle=True, 
                                  num_workers=NUM_WORKERS, pin_memory=False) for idx in client_indices]
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    return client_loaders, test_loader

def partition_opportunity_horizontal(num_clients, batch_size):
    download_opportunity(DATA_ROOT)
    client_loaders = []
    num_subjects = 4
    clients_per_subject = num_clients // num_subjects
    
    for subject_id in range(1, num_subjects + 1):
        try:
            res = load_opportunity_subject(DATA_ROOT, subject_id)
            if res is None: continue
            b, o, a, y = res
            dataset = OpportunityDataset(b, o, a, y)
            total_samples = len(dataset)
            indices = np.arange(total_samples)
            client_indices = np.array_split(indices, clients_per_subject)
            for idx in client_indices:
                loader = DataLoader(Subset(dataset, idx), batch_size=batch_size, 
                                   shuffle=True, num_workers=NUM_WORKERS, pin_memory=False)
                client_loaders.append(loader)
        except RuntimeError: pass
    
    try:
        res_test = load_opportunity_subject(DATA_ROOT, 2, runs=["ADL4", "ADL5"])
        if res_test:
            b_test, o_test, a_test, y_test = res_test
            test_loader = DataLoader(OpportunityDataset(b_test, o_test, a_test, y_test), 
                                    batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
        else:
            return [], None
    except: return [], None
    return client_loaders, test_loader

def partition_opportunity_vertical(batch_size):
    download_opportunity(DATA_ROOT)
    all_body, all_obj, all_amb, all_labels = [], [], [], []
    for subject_id in range(1, NUM_SILOS_OPP + 1):
        try:
            res = load_opportunity_subject(DATA_ROOT, subject_id)
            if res:
                b, o, a, y = res
                all_body.append(b); all_obj.append(o); all_amb.append(a); all_labels.append(y)
        except RuntimeError: continue
    
    if not all_body: return None, None, None, None
    
    body, obj, amb, labels = torch.cat(all_body), torch.cat(all_obj), torch.cat(all_amb), torch.cat(all_labels)
    
    # FIX v22: Pre-shuffle training data with consistent indices to ensure alignment
    num_samples = len(labels)
    shuffle_indices = torch.randperm(num_samples)
    body = body[shuffle_indices]
    obj = obj[shuffle_indices]
    amb = amb[shuffle_indices]
    labels = labels[shuffle_indices]
    
    silo_data = {"body_upper": body[:, :15, :], "body_lower": body[:, 15:, :], "objects": obj, "ambient": amb}
    # Now use shuffle=False since data is already pre-shuffled
    silo_loaders = {name: DataLoader(TensorDataset(feats), batch_size=batch_size, shuffle=False, 
                                      num_workers=NUM_WORKERS, pin_memory=False) for name, feats in silo_data.items()}
    label_loader = DataLoader(TensorDataset(labels), batch_size=batch_size, shuffle=False)
    
    res_test = load_opportunity_subject(DATA_ROOT, 2, runs=["ADL4", "ADL5"])
    if res_test:
        b_t, o_t, a_t, y_t = res_test
        # Test data: no shuffle needed
        test_silo_data = {"body_upper": b_t[:,:15,:], "body_lower": b_t[:,15:,:], "objects": o_t, "ambient": a_t}
        test_silo_loaders = {n: DataLoader(TensorDataset(f), batch_size=batch_size, shuffle=False) for n, f in test_silo_data.items()}
        test_label_loader = DataLoader(TensorDataset(y_t), batch_size=batch_size, shuffle=False)
        return silo_loaders, label_loader, test_silo_loaders, test_label_loader
    return None, None, None, None

# --- MODELS ---
class SimpleCNN_GN(nn.Module):
    def __init__(self, dropout_p=0.25):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1); self.gn1 = nn.GroupNorm(8, 32)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1); self.gn2 = nn.GroupNorm(8, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1); self.gn3 = nn.GroupNorm(8, 64)
        self.conv4 = nn.Conv2d(64, 64, 3, padding=1); self.gn4 = nn.GroupNorm(8, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64*8*8, 256); self.fc2 = nn.Linear(256, 10)
        self.dropout = nn.Dropout(dropout_p)
    def forward(self, x):
        if isinstance(x, dict): x = x['image']
        x = self.pool1(F.relu(self.gn2(self.conv2(F.relu(self.gn1(self.conv1(x)))))))
        x = self.pool2(F.relu(self.gn4(self.conv4(F.relu(self.gn3(self.conv3(x)))))))
        return self.fc2(self.dropout(F.relu(self.fc1(torch.flatten(x, 1)))))

# v24: Simple Conv encoder without LSTM (DP-compatible)
class DeepConvEncoder(nn.Module):
    def __init__(self, in_channels, dropout_p=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 5), 
            nn.GroupNorm(8, 64),  # DP-compatible normalization
            nn.ReLU(), 
            nn.MaxPool1d(2), 
            nn.Conv1d(64, 64, 3, padding=1),  # Additional conv layer to compensate for no LSTM
            nn.GroupNorm(8, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p)
        )
    def forward(self, x): return self.conv(x)

class EarlyFusionModel(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.encoder = DeepConvEncoder(55, dropout_p)
        self.cls = nn.Sequential(nn.Flatten(), nn.Linear(64*13, 128), nn.ReLU(), nn.Linear(128, 18))
    def forward(self, x):
        return self.cls(self.encoder(torch.cat([x['body_sensors'], x['object_sensors'], x['ambient_sensors']], dim=1)))

class IntermediateFusionModel(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        # v23: Reduce encoder dropout to 0.15 (was 0.5)
        encoder_dropout = dropout_p * 0.3
        self.body_net = DeepConvEncoder(30, encoder_dropout)
        self.obj_net = DeepConvEncoder(15, encoder_dropout)
        self.amb_net = DeepConvEncoder(10, encoder_dropout)
        # v23: Wider bottleneck (256 vs 128), reduced classifier dropout (0.25 vs 0.5)
        self.cls = nn.Sequential(
            nn.Flatten(), 
            nn.Linear(64*13*3, 256), 
            nn.ReLU(), 
            nn.Dropout(dropout_p * 0.5), 
            nn.Linear(256, 18)
        )
    def forward(self, x):
        return self.cls(torch.cat([self.body_net(x['body_sensors']), self.obj_net(x['object_sensors']), self.amb_net(x['ambient_sensors'])], dim=1))

class LateFusionModel(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.body_head = nn.Sequential(DeepConvEncoder(30, dropout_p), nn.Flatten(), nn.Linear(64*13, 18))
        self.obj_head = nn.Sequential(DeepConvEncoder(15, dropout_p), nn.Flatten(), nn.Linear(64*13, 18))
        self.amb_head = nn.Sequential(DeepConvEncoder(10, dropout_p), nn.Flatten(), nn.Linear(64*13, 18))
    def forward(self, x):
        return (self.body_head(x['body_sensors']) + self.obj_head(x['object_sensors']) + self.amb_head(x['ambient_sensors'])) / 3.0

class SiloEncoder(nn.Module):
    def __init__(self, in_channels, dropout_p=0.5):
        super().__init__()
        self.encoder = DeepConvEncoder(in_channels, dropout_p)
    def forward(self, x): return self.encoder(x)

class IntermediateFusionVFL(nn.Module):
    def __init__(self, num_silos=4, embedding_size_per_silo=832, num_classes=18, dropout_p=0.5):
        super().__init__()
        # v23: Wider bottleneck (256 vs 128), reduced dropout (0.25 vs 0.5)
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
    def __init__(self): super().__init__()
    def forward(self, logits_list): return torch.mean(torch.stack(logits_list), dim=0)

# v24: Early Fusion for VFL - concatenate all silo inputs before encoding
class EarlyFusionVFL(nn.Module):
    def __init__(self, dropout_p=0.5, num_classes=18):
        super().__init__()
        # Total input channels: body_upper(15) + body_lower(15) + objects(15) + ambient(10) = 55
        self.encoder = DeepConvEncoder(55, dropout_p)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*13, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p * 0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, embeddings_list):
        # embeddings_list contains raw features from each silo, not encoded yet
        # Concatenate all features: [batch, channels, time]
        concatenated = torch.cat(embeddings_list, dim=1)  # Concatenate along channel dimension
        encoded = self.encoder(concatenated)
        return self.classifier(encoded)

# --- CLASSES FOR FL ---
class VerticalSilo:
    def __init__(self, silo_id, silo_name, in_channels, device, params, fusion_mode='Intermediate'):
        self.silo_id = silo_id
        self.silo_name = silo_name
        self.device = device
        self.params = params
        self.fusion_mode = fusion_mode
        self.encoder = SiloEncoder(in_channels, params['dropout']).to(device)
        if fusion_mode == 'Late':
            self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(64*13, 18)).to(device)
        self.optimizer = optim.SGD(self.encoder.parameters(), lr=params['lr'], momentum=params['momentum'])
        self.privacy_engine = None
    
    def make_private(self, data_loader, noise_multiplier):
        self.encoder.train()
        self.privacy_engine = PrivacyEngine()
        self.encoder, self.optimizer, _ = self.privacy_engine.make_private(
            module=self.encoder, optimizer=self.optimizer, data_loader=data_loader,
            noise_multiplier=noise_multiplier, max_grad_norm=5.0
        )
    def get_epsilon(self):
        return self.privacy_engine.get_epsilon(delta=DP_DELTA) if self.privacy_engine else 0.0
    def forward(self, data):
        embedding = self.encoder(data)
        if self.fusion_mode == 'Late': return self.classifier(embedding)
        return embedding

class VerticalCoordinator:
    def __init__(self, silos, fusion_head, device, params):
        self.silos = silos
        self.fusion_head = fusion_head.to(device)
        self.device = device
        self.params = params
        # Late fusion has no trainable parameters (just averaging), so no optimizer needed
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
        for silo in self.silos.values(): silo.encoder.train()
        total_loss = 0; num_batches = 0
        for epoch in range(epochs):
            silo_iters = {name: iter(loader) for name, loader in silo_loaders.items()}
            label_iter = iter(label_loader)
            try:
                while True:
                    silo_batches = {name: next(silo_iters[name])[0].to(self.device) for name in self.silos.keys()}
                    labels = next(label_iter)[0].to(self.device)
                    if self.optimizer:
                        self.optimizer.zero_grad()
                    for silo in self.silos.values(): silo.optimizer.zero_grad()
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
                    for silo in self.silos.values():
                        if not silo.privacy_engine:  # Only clip if not using DP (DP does it automatically)
                            torch.nn.utils.clip_grad_norm_(silo.encoder.parameters(), max_norm=5.0)
                    if self.optimizer:
                        self.optimizer.step()
                    for silo in self.silos.values(): silo.optimizer.step()
                    total_loss += loss.item(); num_batches += 1
            except StopIteration: pass
        return total_loss / max(num_batches, 1)

    def evaluate(self, test_silo_loaders, test_label_loader, return_per_class=False):
        self.fusion_head.eval()
        for silo in self.silos.values(): silo.encoder.eval()
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
            except StopIteration: pass
        acc = accuracy_score(all_targets, all_preds)
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        return acc, f1, all_preds  # Return predictions for diagnostic

class FederatedClient:
    def __init__(self, client_id, train_loader, device, params):
        self.client_id = client_id; self.train_loader = train_loader; self.device = device; self.params = params
        self.model = None; self.privacy_engine = None
    
    def train(self, global_model, local_epochs, use_dp=False, noise_multiplier=1.0):
        self.model = copy.deepcopy(global_model).to(self.device); self.model.train()
        optimizer = optim.SGD(self.model.parameters(), lr=self.params['lr'], momentum=self.params['momentum'])
        if use_dp:
            self.privacy_engine = PrivacyEngine()
            self.model, optimizer, train_loader_private = self.privacy_engine.make_private(module=self.model, optimizer=optimizer, data_loader=self.train_loader, noise_multiplier=noise_multiplier, max_grad_norm=5.0)
            loader = train_loader_private
        else: loader = self.train_loader
        
        epoch_loss = 0
        for epoch in range(local_epochs):
            for data, target in loader:
                target = target.to(self.device)
                data = {k: v.to(self.device) for k, v in data.items()} if isinstance(data, dict) else data.to(self.device)
                optimizer.zero_grad()
                loss = F.cross_entropy(self.model(data), target)
                loss.backward()
                if not use_dp:  # Apply gradient clipping manually when not using DP
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
    return acc, f1, all_preds  # Return predictions for diagnostic

# --- RUNNERS ---
def run_horizontal_fl(dataset_name, fusion_mode, dp_epsilon, seed, num_rounds=NUM_ROUNDS, epochs=LOCAL_EPOCHS):
    set_seed(seed)
    params = DATASET_PARAMS[dataset_name].copy()
    
    # v23: Lower LR for Intermediate fusion (3x more parameters)
    if fusion_mode == 'Intermediate':
        params['lr'] = params['lr'] * 0.3  # 0.01 → 0.003
    
    if dataset_name == DATASET_CIFAR10:
        client_loaders, test_loader = partition_cifar10_iid(NUM_CLIENTS_CIFAR, params['batch_size'])
        global_model = SimpleCNN_GN(dropout_p=params['dropout']).to(DEVICE)
    else:
        client_loaders, test_loader = partition_opportunity_horizontal(NUM_CLIENTS_OPP, params['batch_size'])
        if not client_loaders: raise RuntimeError("Data Partition Failed")
        if fusion_mode == 'Early': global_model = EarlyFusionModel(params['dropout']).to(DEVICE)
        elif fusion_mode == 'Intermediate': global_model = IntermediateFusionModel(params['dropout']).to(DEVICE)
        elif fusion_mode == 'Late': global_model = LateFusionModel(params['dropout']).to(DEVICE)
    use_dp = dp_epsilon > 0
    noise_multiplier, dp_sample_rate, dp_steps_per_round = dp_plan_from_loaders(
        dp_epsilon, client_loaders, params['batch_size'], num_rounds, epochs, DP_DELTA
    )
    
    clients = [FederatedClient(i, loader, DEVICE, params) for i, loader in enumerate(client_loaders)]
    round_metrics = []
    
    # Early stopping tracking
    best_f1 = 0.0
    rounds_without_improvement = 0
    
    for round_idx in range(num_rounds):
        client_weights, client_sizes, client_losses, client_eps = [], [], [], []
        for client in clients:
            w, s, l, e = client.train(global_model, epochs, use_dp=use_dp, noise_multiplier=noise_multiplier)
            client_weights.append(w); client_sizes.append(s); client_losses.append(l); client_eps.append(e)
        
        avg_loss = np.mean(client_losses)
        
        # Early stop if loss explodes
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
        
        # Track improvement for patience-based early stopping
        if f1 > best_f1:
            best_f1 = f1
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
        
        # Early stop if no improvement for patience rounds
        if rounds_without_improvement >= EARLY_STOP_PATIENCE and round_idx >= 10:  # Allow warmup
            print(f" [EARLY STOP: No improvement for {EARLY_STOP_PATIENCE} rounds] ", end='')
            break
        
        if (round_idx + 1) % 5 == 0: print(f" (Loss: {avg_loss:.3f}) ", end='', flush=True)
        else: print('.', end='', flush=True)
    return round_metrics

def run_vertical_fl(fusion_mode, dp_epsilon, seed, num_rounds=NUM_ROUNDS, epochs=LOCAL_EPOCHS):
    set_seed(seed)
    params = DATASET_PARAMS[DATASET_OPPORTUNITY].copy()
    
    # v23: Lower LR for Intermediate fusion
    if fusion_mode == 'Intermediate':
        params['lr'] = params['lr'] * 0.3  # 0.01 → 0.003
    
    silo_loaders, label_loader, test_silo_loaders, test_label_loader = partition_opportunity_vertical(params['batch_size'])
    use_dp = dp_epsilon > 0
    # v24: Support all fusion modes including Early
    if fusion_mode == 'Early':
        fusion_head = EarlyFusionVFL(dropout_p=params['dropout'])
    elif fusion_mode == 'Intermediate':
        fusion_head = IntermediateFusionVFL(dropout_p=params['dropout'])
    else:  # Late
        fusion_head = LateFusionVFL()

    noise_multiplier, dp_sample_rate, dp_steps_per_round = dp_plan_from_loaders(
        dp_epsilon, list(silo_loaders.values()), params['batch_size'], num_rounds, epochs, DP_DELTA,
        mechanisms_per_step=vfl_mechanisms_per_step(silo_loaders, fusion_head),
    )
    silos = {name: VerticalSilo(i, name, ch, DEVICE, params, fusion_mode) for i, (name, ch) in enumerate({'body_upper':15, 'body_lower':15, 'objects':15, 'ambient':10}.items())}
    if use_dp:
        for silo in silos.values(): silo.make_private(silo_loaders[silo.silo_name], noise_multiplier=noise_multiplier)

    coordinator = VerticalCoordinator(silos, fusion_head, DEVICE, params)
    if use_dp:
        coordinator.make_fusion_head_private(label_loader, noise_multiplier)
    round_metrics = []
    
    # v24: Enhanced early stopping tracking with checkpoint
    best_f1 = 0.0
    best_checkpoint = None  # v24: Save best checkpoint {fusion_head, silos}
    rounds_without_improvement = 0
    prev_loss = 0.0
    loss_stagnation_count = 0
    
    for round_idx in range(num_rounds):
        loss = coordinator.train_round(silo_loaders, label_loader, epochs=epochs)
        
        # Early stop if loss explodes
        if loss > EARLY_STOP_LOSS_THRESHOLD:
            print(f" [EARLY STOP: Loss={loss:.2e} > threshold] ", end='')
            break
        
        # v24: Loss stagnation detection
        if round_idx > 0:
            loss_change_pct = abs(loss - prev_loss) / max(prev_loss, 1e-6)
            if loss_change_pct < LOSS_STAGNATION_THRESHOLD:
                loss_stagnation_count += 1
                if loss_stagnation_count >= LOSS_STAGNATION_PATIENCE:
                    print(f" [EARLY STOP: Loss stagnation Δ{loss_change_pct:.4f}] ", end='')
                    break
            else:
                loss_stagnation_count = 0
        prev_loss = loss
        
        gc.collect(); torch.cuda.empty_cache()
        acc, f1, preds = coordinator.evaluate(test_silo_loaders, test_label_loader)
        epsilon_spent = composed_epsilon(
            noise_multiplier, dp_sample_rate, (round_idx + 1) * dp_steps_per_round, DP_DELTA
        ) if use_dp else 0.0
        round_metrics.append({'round': round_idx+1, 'loss': loss, 'acc': acc, 'f1': f1, 'epsilon': epsilon_spent})
        
        # Track F1 improvement for patience-based early stopping (v24 fix)
        if f1 > best_f1:
            best_f1 = f1
            # v24: Save best checkpoint
            best_checkpoint = {
                'fusion_head': copy.deepcopy(coordinator.fusion_head.state_dict()),
                'silos': {name: copy.deepcopy(silo.encoder.state_dict()) for name, silo in coordinator.silos.items()}
            }
            rounds_without_improvement = 0
        else:
            rounds_without_improvement += 1
        
        # v24: Relaxed warmup (10 → 5)
        if rounds_without_improvement >= EARLY_STOP_PATIENCE and round_idx >= WARMUP_ROUNDS:
            print(f" [EARLY STOP: No F1 improvement for {EARLY_STOP_PATIENCE} rounds] ", end='')
            break
        
        # v24: Enhanced progress logging
        if (round_idx + 1) % 5 == 0:
            loss_delta = loss - prev_loss if round_idx > 0 else 0
            print(f" (R{round_idx+1}: Loss={loss:.3f} Δ{loss_delta:+.3f}, F1={f1:.4f}) ", end='', flush=True)
        else:
            print('.', end='', flush=True)
    
    # v24 fix: Restore best checkpoint before returning
    if best_checkpoint is not None:
        coordinator.fusion_head.load_state_dict(best_checkpoint['fusion_head'])
        for name, silo in coordinator.silos.items():
            silo.encoder.load_state_dict(best_checkpoint['silos'][name])
        # Re-evaluate with best model
        acc, f1, preds = coordinator.evaluate(test_silo_loaders, test_label_loader)
        round_metrics.append({'round': len(round_metrics)+1, 'loss': round_metrics[-1]['loss'] if round_metrics else 0,
                             'acc': acc, 'f1': f1, 'epsilon': round_metrics[-1]['epsilon'] if round_metrics else 0, 'note':  'best_checkpoint'})
    
    return round_metrics

# --- MAIN CONTROL ---
def run_smoke_test():
    print("\n" + "="*60)
    print(">> PRE-FLIGHT SMOKE TEST (1 Round, 1 Epoch) - VFL Only")
    print("="*60)
    
    configs = [
        {'dataset': DATASET_OPPORTUNITY, 'topology': 'vertical', 'fusion': 'Intermediate', 'epsilon': 1.0}
    ]
    
    for cfg in configs:
        print(f"Testing: {cfg['dataset']} {cfg['fusion']} Eps={cfg['epsilon']}", end=' ')
        try:
            run_vertical_fl(cfg['fusion'], cfg['epsilon'], 42, 1, 1)
            print(" -> OK")
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f" -> FAIL: {e}")
            sys.exit(1)
    print(">> SMOKE TEST PASSED\n")

if __name__ == "__main__":
    print(f"Starting Grid Job on {os.uname().nodename}")
    print(f"Device: {DEVICE}")
    
    # 1. Smoke Test
    run_smoke_test()
    
    # 2. Main Experiments - PART 2: OPPORTUNITY Vertical Cross-Silo
    experiments = []
    
    # NOTE: No Early fusion for Vertical - doesn't make sense in VFL (breaks privacy)
    # OPPORTUNITY Vertical: Intermediate fusion, all epsilons (4 configs)
    experiments.extend([{'dataset': DATASET_OPPORTUNITY, 'topo': 'vertical', 'fusion': 'Intermediate', 'eps': eps} for eps in DP_EPSILONS])
    # OPPORTUNITY Vertical: Late fusion, all epsilons (4 configs)
    experiments.extend([{'dataset': DATASET_OPPORTUNITY, 'topo': 'vertical', 'fusion': 'Late', 'eps': eps} for eps in DP_EPSILONS])
    
    print(f"\n=== PART 2: Vertical Cross-Silo (Eps={DP_EPSILONS}) v24 - Running {len(experiments)} experiment configurations ===")
    print(f"⏱️  Timeout protection: {format_timeout_duration(7200)} per config\n")
    
    all_results = []
    for i, exp in enumerate(experiments):
        # Format topology label
        topo_label = exp['topo'].capitalize() if exp['dataset'] == DATASET_OPPORTUNITY else ''
        fusion_label = f"{topo_label} {exp['fusion']}" if topo_label else exp['fusion']
        print(f"\n[{i+1}/{len(experiments)}] {exp['dataset']} {fusion_label} Eps={exp['eps']}")
        
        seed_results = []
        for seed in SEEDS:
            print(f"  Seed {seed}", end=' ', flush=True)
            
            # Define the experiment function to run with timeout
            def run_single_experiment():
                if exp['topo'] == 'vertical':
                    return run_vertical_fl(exp['fusion'], exp['eps'], seed)
                else:
                    return run_horizontal_fl(exp['dataset'], exp['fusion'], exp['eps'], seed)
            
            # Run with 2-hour timeout protection
            success, res = run_with_timeout(
                run_single_experiment,
                timeout_seconds=7200,  # 2 hours
                on_timeout=lambda: print(f"\n  ⚠️  TIMEOUT after 2h - skipping to next config")
            )
            
            if not success:
                # Timeout occurred
                print(f" [TIMEOUT]")
                seed_results.append([])  # Empty result for this seed
                continue
            
            seed_results.append(res)
            
            # Handle early stop case where res might be empty
            if res:
                final_f1 = res[-1]['f1']
                # Diagnostic: Check if model collapsed (only for Intermediate fusion)
                if 'Intermediate' in exp['fusion'] and final_f1 < 0.10:
                    # Get last prediction from round metrics if available
                    print(f" F1={final_f1:.4f} [COLLAPSED]")
                else:
                    print(f" F1={final_f1:.4f}")
            else:
                print(f" [FAILED: No rounds completed]")
        
        # Save per-experiment result
        exp_data = {'config': exp, 'runs': seed_results}
        all_results.append(exp_data)
        
        # Incremental Save
        with open('results/partial_results.json', 'w') as f: json.dump(all_results, f)

    # Final Save
    os.makedirs("results", exist_ok=True)
    with open('results/exp01_v24_results_part2.json', 'w') as f: json.dump(all_results, f)
    print("\nPART 2 DONE. Results saved to results/exp01_v24_results_part2.json")
