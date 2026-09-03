#!/usr/bin/env python3
# 08_FL_RobustAgg_DP_Frontier_part2.py — Median + Trimmed Mean under DP frontier
# Re-roda exp03a parte 2 com DP corrigido e ε∈{20,100} (do Exp02 frontier). Bug
# σ=1/ε ausente; calibração via dp_plan_from_loaders (Opacus RDPAccountant).
# Experiment 03a — Part 2: HFL Trimmed Mean + Median with Differential Privacy
#
# Extends Exp03 Part2 by adding DP (Opacus) with epsilon in {3.0, 8.0}.
#
# Aggregations : Trimmed Mean (Yin et al., 2018), Coordinate-wise Median (Yin et al., 2018)
# Attack Types : Label Flip, Sign Flip, Scaling x10, Free-Rider
# Attack Ratios: 10%, 25%, 50% of clients
# DP Epsilons  : 3.0, 8.0
# Fusion Modes : Intermediate, Late
# Seeds        : 42, 123, 456

import os, sys, gc, copy, json, warnings
import numpy as np
import pandas as pd
import zipfile
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from opacus import PrivacyEngine
from sklearn.metrics import f1_score, accuracy_score

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
    SIGN_FLIP  = "sign_flip"
    SCALING    = "scaling"
    FREE_RIDER = "free_rider"
    NONE       = "none"
    ATTACK_TYPES      = [LABEL_FLIP, SIGN_FLIP, SCALING, FREE_RIDER]
    HFL_ATTACK_RATIOS = [0.10, 0.25, 0.50, 0.75]
    # exp08: ratios estendidos (incluindo 0.75) — antes era [0.10, 0.25, 0.50]

    @staticmethod
    def get_num_malicious_clients(total_clients, attack_ratio):
        return max(1, int(total_clients * attack_ratio))


def apply_label_flipping(labels, num_classes, flip_ratio=1.0):
    flipped = labels.clone()
    n_to_flip = int(len(labels) * flip_ratio)
    flip_indices = np.random.choice(len(labels), n_to_flip, replace=False)
    for idx in flip_indices:
        original = flipped[idx].item()
        flipped[idx] = np.random.choice([c for c in range(num_classes) if c != original])
    return flipped


def apply_attack_to_weights(state_dict, attack_type):
    if attack_type == AttackConfig.SIGN_FLIP:
        return {k: -v.clone() for k, v in state_dict.items()}
    elif attack_type == AttackConfig.SCALING:
        return {k: v * 10.0 for k, v in state_dict.items()}
    elif attack_type == AttackConfig.FREE_RIDER:
        return {k: torch.zeros_like(v) for k, v in state_dict.items()}
    return state_dict

# =============================================================================
# AGGREGATION SCHEMES
# =============================================================================

def trimmed_mean_aggregate(client_weights, client_sizes, num_malicious):
    n = len(client_weights)
    beta = max(1, min(num_malicious, (n - 1) // 2))
    aggregated = {}
    for key in client_weights[0]:
        stacked = torch.stack([cw[key].float() for cw in client_weights])
        shape = stacked.shape[1:]
        flat = stacked.view(n, -1)
        sorted_flat, _ = torch.sort(flat, dim=0)
        trimmed = sorted_flat[beta:n - beta, :]
        aggregated[key] = trimmed.mean(dim=0).view(shape)
    return aggregated


def median_aggregate(client_weights, client_sizes, num_malicious):
    aggregated = {}
    for key in client_weights[0]:
        stacked = torch.stack([cw[key].float() for cw in client_weights])
        shape = stacked.shape[1:]
        flat = stacked.view(len(client_weights), -1)
        aggregated[key] = torch.median(flat, dim=0).values.view(shape)
    return aggregated


AGGREGATION_FNS = {
    "trimmed_mean": trimmed_mean_aggregate,
    "median":       median_aggregate,
}

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

# =============================================================================
# CONFIGURATION
# =============================================================================

NUM_ROUNDS   = 25
LOCAL_EPOCHS = 3
DATA_ROOT    = "./data"
SEEDS        = [42, 123, 456]
NUM_CLIENTS_OPP = 12
NUM_WORKERS  = 4
NUM_CLASSES  = opportunity_num_classes()
DP_DELTA     = 1e-5
DP_EPSILONS  = [20.0, 100.0]

DATASET_PARAMS = {
    "OPPORTUNITY": {
        "lr": 0.01, "momentum": 0.9, "batch_size": 16, "dropout": 0.5,
    }
}

ATTACK_TYPES      = AttackConfig.ATTACK_TYPES
ATTACK_RATIOS     = AttackConfig.HFL_ATTACK_RATIOS
FUSION_MODES      = ['Intermediate', 'Late']
AGGREGATION_NAMES = ["trimmed_mean", "median"]

EARLY_STOP_LOSS_THRESHOLD = 1e6
EARLY_STOP_PATIENCE = 3
WARMUP_ROUNDS = 5

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); torch.backends.cudnn.deterministic = True

# =============================================================================
# DATASET  (identical to part1)
# =============================================================================

def download_opportunity(root):
    ep = os.path.join(root, "OpportunityUCIDataset")
    if os.path.exists(ep): return
    zp = os.path.join(root, "OpportunityUCIDataset.zip")
    if os.path.exists(zp):
        with zipfile.ZipFile(zp, 'r') as zf: zf.extractall(root)
    else:
        print(f"WARNING: Opportunity zip not found at {zp}")


def load_opportunity_subject(root, subject_id, runs=['Drill', 'ADL1', 'ADL2']):
    base = os.path.join(root, "OpportunityUCIDataset", "dataset")
    all_body, all_obj, all_amb, all_labels = [], [], [], []
    for run in runs:
        fp = os.path.join(base, f"S{subject_id}-{run}.dat")
        if not os.path.exists(fp): continue
        try:
            df = pd.read_csv(fp, sep=r'\s+', header=None, engine='python')
            df = df.interpolate(method='linear', limit_direction='forward').fillna(0)
            data = df.values
        except Exception:
            continue
        body = data[:, 1:31]; obj = data[:, 50:65]; amb = data[:, 100:110]
        labels = encode_opportunity_labels(data[:, OPPORTUNITY_LABEL_COLUMN] if data.shape[1] > OPPORTUNITY_LABEL_COLUMN else np.zeros(len(data)))
        ws, step = 30, 15
        for i in range(0, len(data) - ws, step):
            all_body.append(body[i:i+ws].T); all_obj.append(obj[i:i+ws].T)
            all_amb.append(amb[i:i+ws].T);   all_labels.append(labels[i+ws-1])
    if not all_labels: return None
    return (torch.tensor(np.array(all_body), dtype=torch.float32),
            torch.tensor(np.array(all_obj),  dtype=torch.float32),
            torch.tensor(np.array(all_amb),  dtype=torch.float32),
            torch.tensor(np.array(all_labels), dtype=torch.long))


class OpportunityDataset(Dataset):
    def __init__(self, body, obj, amb, labels):
        self.b, self.o, self.a, self.y = body, obj, amb, labels
    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return {'body_sensors': self.b[idx], 'object_sensors': self.o[idx],
                'ambient_sensors': self.a[idx]}, self.y[idx]


def partition_opportunity_horizontal(num_clients, batch_size):
    download_opportunity(DATA_ROOT)
    client_loaders = []
    cps = num_clients // 4
    for sid in range(1, 5):
        try:
            res = load_opportunity_subject(DATA_ROOT, sid)
            if res is None: continue
            b, o, a, y = res
            ds = OpportunityDataset(b, o, a, y)
            for idx in np.array_split(np.arange(len(ds)), cps):
                client_loaders.append(DataLoader(Subset(ds, idx), batch_size=batch_size,
                    shuffle=True, num_workers=NUM_WORKERS, pin_memory=False))
        except RuntimeError:
            pass
    try:
        res_t = load_opportunity_subject(DATA_ROOT, 2, runs=["ADL4", "ADL5"])
        if res_t is None: return [], None
        b_t, o_t, a_t, y_t = res_t
        test_loader = DataLoader(OpportunityDataset(b_t, o_t, a_t, y_t),
            batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS, pin_memory=False)
    except Exception:
        return [], None
    return client_loaders, test_loader

# =============================================================================
# MODELS
# =============================================================================

class DeepConvEncoder(nn.Module):
    def __init__(self, in_channels, dropout_p=0.5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 64, 5), nn.GroupNorm(8, 64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.GroupNorm(8, 64), nn.ReLU(), nn.Dropout(dropout_p),
        )
    def forward(self, x): return self.conv(x)


class IntermediateFusionModel(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        dp = dropout_p * 0.3
        self.body_net = DeepConvEncoder(30, dp)
        self.obj_net  = DeepConvEncoder(15, dp)
        self.amb_net  = DeepConvEncoder(10, dp)
        self.cls = nn.Sequential(nn.Flatten(), nn.Linear(64*13*3, 256),
                                 nn.ReLU(), nn.Dropout(dropout_p * 0.5), nn.Linear(256, NUM_CLASSES))
    def forward(self, x):
        return self.cls(torch.cat([self.body_net(x['body_sensors']),
                                   self.obj_net(x['object_sensors']),
                                   self.amb_net(x['ambient_sensors'])], dim=1))


class LateFusionModel(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.body_head = nn.Sequential(DeepConvEncoder(30, dropout_p), nn.Flatten(), nn.Linear(64*13, NUM_CLASSES))
        self.obj_head  = nn.Sequential(DeepConvEncoder(15, dropout_p), nn.Flatten(), nn.Linear(64*13, NUM_CLASSES))
        self.amb_head  = nn.Sequential(DeepConvEncoder(10, dropout_p), nn.Flatten(), nn.Linear(64*13, NUM_CLASSES))
    def forward(self, x):
        return (self.body_head(x['body_sensors']) +
                self.obj_head(x['object_sensors']) +
                self.amb_head(x['ambient_sensors'])) / 3.0

# =============================================================================
# FEDERATED CLIENT  (DP-aware)
# =============================================================================

class MaliciousFederatedClient:
    def __init__(self, client_id, train_loader, device, params,
                 is_malicious=False, attack_type=AttackConfig.NONE):
        self.client_id    = client_id
        self.train_loader = train_loader
        self.device       = device
        self.params       = params
        self.is_malicious = is_malicious
        self.attack_type  = attack_type
        self.model = None
        self.privacy_engine = None

    def train(self, global_model, local_epochs, use_dp=False, noise_multiplier=1.0):
        self.model = copy.deepcopy(global_model).to(self.device)
        if self.is_malicious and self.attack_type == AttackConfig.FREE_RIDER:
            return self.model.state_dict(), len(self.train_loader.dataset), 0.0, 0.0

        self.model.train()
        optimizer = optim.SGD(self.model.parameters(),
                              lr=self.params['lr'], momentum=self.params['momentum'])
        if use_dp:
            self.privacy_engine = PrivacyEngine()
            self.model, optimizer, loader = self.privacy_engine.make_private(
                module=self.model, optimizer=optimizer, data_loader=self.train_loader,
                noise_multiplier=noise_multiplier, max_grad_norm=5.0,
            )
        else:
            loader = self.train_loader

        epoch_loss = 0.0
        for _ in range(local_epochs):
            for data, target in loader:
                target = target.to(self.device)
                data   = {k: v.to(self.device) for k, v in data.items()}
                if self.is_malicious and self.attack_type == AttackConfig.LABEL_FLIP:
                    target = apply_label_flipping(target, NUM_CLASSES).to(self.device)
                optimizer.zero_grad()
                loss = F.cross_entropy(self.model(data), target)
                loss.backward()
                if not use_dp:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                epoch_loss += loss.item()

        epsilon = self.privacy_engine.get_epsilon(delta=DP_DELTA) if use_dp else 0.0
        state_dict = self.model.state_dict()
        if use_dp:
            state_dict = {k.replace('_module.', ''): v for k, v in state_dict.items()}
        if self.is_malicious and self.attack_type in [AttackConfig.SIGN_FLIP, AttackConfig.SCALING]:
            state_dict = apply_attack_to_weights(state_dict, self.attack_type)
        return state_dict, len(self.train_loader.dataset), epoch_loss / local_epochs, epsilon


def evaluate(model, loader, device):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for data, target in loader:
            data = {k: v.to(device) for k, v in data.items()}
            preds.extend(model(data).argmax(dim=1).cpu().numpy())
            targets.extend(target.numpy())
    return (accuracy_score(targets, preds),
            f1_score(targets, preds, average='macro', zero_division=0))

# =============================================================================
# MAIN RUNNER
# =============================================================================

def run_horizontal_fl_robust(fusion_mode, aggregation_name, attack_type,
                              attack_ratio, dp_epsilon, seed,
                              num_rounds=NUM_ROUNDS, epochs=LOCAL_EPOCHS):
    set_seed(seed)
    params = DATASET_PARAMS["OPPORTUNITY"].copy()
    if fusion_mode == 'Intermediate':
        params['lr'] *= 0.3

    use_dp = dp_epsilon > 0

    client_loaders, test_loader = partition_opportunity_horizontal(NUM_CLIENTS_OPP, params['batch_size'])
    if not client_loaders:
        raise RuntimeError("Data partition failed")
    noise_multiplier, dp_sample_rate, dp_steps_per_round = dp_plan_from_loaders(
        dp_epsilon, client_loaders, params['batch_size'], num_rounds, epochs, DP_DELTA
    )

    global_model = (IntermediateFusionModel(params['dropout']) if fusion_mode == 'Intermediate'
                    else LateFusionModel(params['dropout'])).to(DEVICE)

    num_malicious = AttackConfig.get_num_malicious_clients(len(client_loaders), attack_ratio)
    malicious_ids = select_malicious_indices(len(client_loaders), attack_ratio, seed)

    clients = [
        MaliciousFederatedClient(i, loader, DEVICE, params,
                                 is_malicious=(i in malicious_ids),
                                 attack_type=attack_type if i in malicious_ids else AttackConfig.NONE)
        for i, loader in enumerate(client_loaders)
    ]

    aggregate_fn = AGGREGATION_FNS[aggregation_name]
    round_metrics, best_f1, best_state, rounds_no_improve = [], 0.0, None, 0

    for round_idx in range(num_rounds):
        weights, sizes, losses, epsilons = [], [], [], []
        for client in clients:
            w, s, l, e = client.train(global_model, epochs, use_dp=use_dp,
                                       noise_multiplier=noise_multiplier)
            weights.append(w); sizes.append(s); losses.append(l); epsilons.append(e)

        avg_loss = float(np.mean(losses))
        if avg_loss > EARLY_STOP_LOSS_THRESHOLD:
            print(f" [EARLY STOP: loss={avg_loss:.2e}]", end=''); break

        global_model.load_state_dict(aggregate_fn(weights, sizes, num_malicious))
        gc.collect(); torch.cuda.empty_cache()

        acc, f1 = evaluate(global_model, test_loader, DEVICE)
        epsilon_spent = composed_epsilon(
            noise_multiplier, dp_sample_rate, (round_idx + 1) * dp_steps_per_round, DP_DELTA
        ) if use_dp else 0.0
        round_metrics.append({
            'round': round_idx + 1, 'loss': avg_loss, 'acc': acc, 'f1': f1,
            'epsilon': epsilon_spent,
            'attack_type': attack_type, 'attack_ratio': attack_ratio,
            'num_malicious': num_malicious, 'aggregation': aggregation_name,
            'dp_epsilon': dp_epsilon, 'topology': 'HFL', 'fusion_mode': fusion_mode,
        })

        if f1 > best_f1:
            best_f1 = f1; best_state = copy.deepcopy(global_model.state_dict())
            rounds_no_improve = 0
        else:
            rounds_no_improve += 1

        if rounds_no_improve >= EARLY_STOP_PATIENCE and round_idx >= WARMUP_ROUNDS:
            print(f" [EARLY STOP]", end=''); break

        if (round_idx + 1) % 5 == 0:
            print(f" (R{round_idx+1}: F1={f1:.4f})", end='', flush=True)
        else:
            print('.', end='', flush=True)

    if best_state:
        global_model.load_state_dict(best_state)
        acc, f1 = evaluate(global_model, test_loader, DEVICE)
        round_metrics.append({**round_metrics[-1], 'round': len(round_metrics)+1,
                               'acc': acc, 'f1': f1, 'note': 'best_checkpoint'})
    return round_metrics

# =============================================================================
# SMOKE TEST + MAIN
# =============================================================================

def run_smoke_test():
    print("\n" + "="*60)
    print("PRE-FLIGHT SMOKE TEST — 03a Part2 (TrimMean+Median with DP)")
    print("="*60)
    for cfg in [
        {'fusion': 'Intermediate', 'agg': 'trimmed_mean', 'eps': 3.0,
         'attack_type': AttackConfig.SIGN_FLIP, 'ratio': 0.25},
        {'fusion': 'Late', 'agg': 'median', 'eps': 8.0,
         'attack_type': AttackConfig.SCALING, 'ratio': 0.10},
    ]:
        print(f"  {cfg['agg']} {cfg['fusion']} eps={cfg['eps']} "
              f"attack={cfg['attack_type']} ratio={cfg['ratio']}", end=' ')
        try:
            run_horizontal_fl_robust(cfg['fusion'], cfg['agg'], cfg['attack_type'],
                                     cfg['ratio'], cfg['eps'], 42, 1, 1)
            print("-> OK")
        except Exception:
            import traceback; traceback.print_exc(); sys.exit(1)
    print("SMOKE TEST PASSED\n")


if __name__ == "__main__":
    print(f"Starting 03a Part2 (TrimMean+Median+DP) on {os.uname().nodename}")
    print(f"Device: {DEVICE}")

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=str, default='all',
                        choices=['a', 'b', 'c', 'd', 'all'])
    args = parser.parse_args()

    if args.part in ('a', 'all'):
        run_smoke_test()

    experiments = []
    for agg in AGGREGATION_NAMES:
        for fusion in FUSION_MODES:
            for attack in ATTACK_TYPES:
                for ratio in ATTACK_RATIOS:
                    for eps in DP_EPSILONS:
                        experiments.append({'aggregation': agg, 'fusion': fusion,
                                            'attack_type': attack, 'attack_ratio': ratio,
                                            'dp_epsilon': eps})

    # exp08: 2 aggs × 2 fusions × 4 attacks × 4 ratios × 2 eps = 128 configs
    part_slices = {'a': (0, 32), 'b': (32, 64), 'c': (64, 96), 'd': (96, 128), 'all': (0, 128)}
    start, end = part_slices[args.part]
    experiments = experiments[start:end]

    os.makedirs("results", exist_ok=True)
    print(f"\n=== 08 PART 2{args.part.upper()}: TrimMean+Median+DP_Frontier — {len(experiments)} configs ===")

    all_results = []
    for i, exp in enumerate(experiments):
        global_idx = start + i + 1
        print(f"\n[{global_idx}/128] {exp['aggregation']} {exp['fusion']} "
              f"eps={exp['dp_epsilon']} attack={exp['attack_type']} ratio={exp['attack_ratio']}")
        seed_results = []
        for seed in SEEDS:
            print(f"  Seed {seed}", end=' ', flush=True)
            success, res = run_with_timeout(
                lambda: run_horizontal_fl_robust(
                    exp['fusion'], exp['aggregation'], exp['attack_type'],
                    exp['attack_ratio'], exp['dp_epsilon'], seed),
                timeout_seconds=7200,
                on_timeout=lambda: print("\n  TIMEOUT"))
            if not success:
                print(" [TIMEOUT]"); seed_results.append([]); continue
            seed_results.append(res)
            print(f" F1={res[-1]['f1']:.4f}" if res else " [FAILED]")

        exp_data = {'config': {**exp, 'topology': 'HFL'}, 'runs': seed_results}
        all_results.append(exp_data)
        with open(f'results/partial_08_p2{args.part}.json', 'w') as f:
            json.dump(all_results, f)

    outfile = f'results/exp08_robustagg_dp_frontier_part2{args.part}.json'
    with open(outfile, 'w') as f:
        json.dump(all_results, f)
    print(f"\nPART 2{args.part.upper()} DONE -> {outfile}")
