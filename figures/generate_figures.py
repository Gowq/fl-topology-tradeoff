#!/usr/bin/env python3
"""
generate_figures.py — regenerate every figure and headline number in the paper
directly from the result JSON/CSV files shipped in experiments/*/results/.

Usage:
    python figures/generate_figures.py            # writes PDFs into figures/
    OUT_FIGS=/somewhere python figures/generate_figures.py

The script is self-contained: it reads only from this repository's results
directories, so the figures are reproducible from the committed artifacts
without rerunning any experiment.
"""
import json, glob, os, statistics
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXP  = os.path.join(ROOT, "experiments")
FIGS = os.environ.get("OUT_FIGS", HERE)
os.makedirs(FIGS, exist_ok=True)

FLOOR = 1.0 / 19.0   # OPPORTUNITY random-prediction floor (18 classes + null)
COL_W, PAGE_W = 3.50, 7.16

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 10, "axes.titlesize": 10, "axes.labelsize": 10,
    "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9,
    "figure.dpi": 150,
})

ATTACK_COLORS = {"label_flip": "#2E86AB", "sign_flip": "#C73E1D",
                 "scaling": "#F18F01", "free_rider": "#3BB273"}
ATTACK_LABELS = {"label_flip": "Label Flip", "sign_flip": "Sign Flip",
                 "scaling": "Scaling ×10", "free_rider": "Free-Rider"}
AGG_COLORS = {"fedavg": "#2E86AB", "krum": "#C73E1D", "trimmed_mean": "#F18F01",
              "median": "#8E44AD", "coordinator": "#1A1A2E"}
AGG_LABELS = {"fedavg": "FedAvg", "krum": "Krum", "trimmed_mean": "Trimmed Mean",
              "median": "Median", "coordinator": "VFL Coordinator"}
AGG_MARKERS = {"fedavg": "o", "krum": "^", "trimmed_mean": "P",
               "median": "D", "coordinator": "s"}
ATTACKS = ["label_flip", "sign_flip", "scaling", "free_rider"]
AGGS    = ["fedavg", "krum", "trimmed_mean", "median"]
RATIOS  = [0.10, 0.25, 0.50, 0.75]


def save(fig, name):
    fig.savefig(os.path.join(FIGS, name), bbox_inches="tight")
    plt.close(fig)
    print("  saved", name)


def best_f1_stats(entry):
    f1s = [max(r["f1"] for r in seed) for seed in entry["runs"] if seed]
    if not f1s:
        return float("nan"), float("nan")
    return statistics.mean(f1s), (statistics.stdev(f1s) if len(f1s) > 1 else 0.0)


def load(*relpaths):
    out = []
    for rp in relpaths:
        for fp in sorted(glob.glob(os.path.join(EXP, rp))):
            out.extend(json.load(open(fp)))
    return out


def floor_line(ax):
    ax.axhline(FLOOR, color="gray", linewidth=0.9, linestyle=":", zorder=1)


# ── load calibrated data ───────────────────────────────────
exp01 = load("exp01_baseline/results/baseline_opportunity_cifar10_part1.json",
             "exp01_baseline/results/baseline_opportunity_vfl_part2.json",
             "exp01_baseline/results/baseline_eps3_part3.json")
cifar100 = load("exp01_baseline/results/baseline_cifar100.json")
frontier = load("exp02_dp_frontier/results/dp_frontier_eps10-200.json")
exp03_hfl = load("exp03_attacks_aggregation/results/hfl_*.json")
exp03_vfl = load("exp03_attacks_aggregation/results/vfl_coordinator.json")
print(f"loaded: baseline={len(exp01)} cifar100={len(cifar100)} frontier={len(frontier)} "
      f"attacks_hfl={len(exp03_hfl)} attacks_vfl={len(exp03_vfl)}")

e08 = {}
for e in exp03_hfl + exp03_vfl:
    c = e["config"]
    e08[(c["aggregation"], c["fusion"], c["attack_type"],
         round(float(c["attack_ratio"]), 2), float(c["dp_epsilon"]))] = best_f1_stats(e)
e01 = {}
for e in exp01 + cifar100:
    c = e["config"]
    e01[(c["dataset"], c["topo"], c["fusion"], float(c["eps"]))] = best_f1_stats(e)
e06 = {}
for e in frontier:
    c = e["config"]
    e06[(c["topo"], c["fusion"], float(c["eps"]))] = best_f1_stats(e)

EPS01 = [0.0, 1.0, 3.0, 5.0, 8.0]
EPS06 = [10.0, 20.0, 50.0, 100.0, 200.0]
EPS_ALL = EPS01 + EPS06
XPOS = list(range(len(EPS_ALL)))
xlab = lambda e: "0\n(no DP)" if e == 0 else str(int(e))

# Fig: opportunity_topology_f1 (baseline + frontier, HFL vs VFL Intermediate)
fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.80))
for topo, color, marker, label in [("horizontal", "#2E86AB", "o", "HFL Intermediate"),
                                    ("vertical", "#C73E1D", "s", "VFL Intermediate")]:
    means, stds = [], []
    for e in EPS_ALL:
        src = e01.get(("OPPORTUNITY", topo, "Intermediate", e)) if e in EPS01 \
              else e06.get((topo, "Intermediate", e))
        means.append(src[0] if src else np.nan)
        stds.append(src[1] if src else np.nan)
    ax.errorbar(XPOS, means, yerr=stds, marker=marker, markersize=5, capsize=3,
                linewidth=1.4, color=color, label=label)
floor_line(ax)
ax.set_xticks(XPOS); ax.set_xticklabels([xlab(e) for e in EPS_ALL])
ax.axvline(4.5, color="gray", linewidth=0.8, linestyle="--")
ax.set_xlabel("Privacy Budget (ε, RDP accountant)")
ax.set_ylabel("F1 Score (Macro)"); ax.set_ylim(0, 1.05)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92))
fig.tight_layout(); save(fig, "opportunity_topology_f1.pdf")

# Fig: cifar_f1_vs_epsilon
fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.78))
for ds, color, marker in [("CIFAR10", "#2E86AB", "o"), ("CIFAR100", "#C73E1D", "s")]:
    eps_ds = sorted({k[3] for k in e01 if k[0] == ds})
    means = [e01[(ds, "horizontal", "Horizontal", e)][0] for e in eps_ds]
    stds  = [e01[(ds, "horizontal", "Horizontal", e)][1] for e in eps_ds]
    ax.errorbar(eps_ds, means, yerr=stds, marker=marker, markersize=5, capsize=3,
                linewidth=1.4, color=color, label=ds.replace("CIFAR", "CIFAR-"))
ax.set_xticks([0, 1, 3, 5, 8]); ax.set_xticklabels(["0\n(no DP)", "1", "3", "5", "8"])
ax.set_xlabel("Privacy Budget (ε, RDP accountant)")
ax.set_ylabel("F1 Score (Macro)"); ax.set_ylim(0, 1.05)
ax.legend(loc="upper right"); fig.tight_layout(); save(fig, "cifar_f1_vs_epsilon.pdf")

# Fig: opportunity_fusion_f1
fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.78))
for topo, fusion, color, marker, label in [
        ("horizontal", "Early", "#8E44AD", "v", "HFL Early"),
        ("horizontal", "Intermediate", "#2E86AB", "o", "HFL Intermediate"),
        ("horizontal", "Late", "#3BB273", "^", "HFL Late"),
        ("vertical", "Intermediate", "#C73E1D", "s", "VFL Intermediate"),
        ("vertical", "Late", "#F18F01", "D", "VFL Late")]:
    means = [e01.get(("OPPORTUNITY", topo, fusion, e), (np.nan,))[0] for e in EPS01]
    ax.plot(EPS01, means, marker=marker, markersize=4.5, linewidth=1.3, color=color, label=label)
floor_line(ax)
ax.set_xticks([0, 1, 3, 5, 8]); ax.set_xticklabels(["0\n(no DP)", "1", "3", "5", "8"])
ax.set_xlabel("Privacy Budget (ε, RDP accountant)")
ax.set_ylabel("F1 Score (Macro)"); ax.set_ylim(0, 1.05)
ax.legend(loc="upper right", ncol=2, fontsize=8); fig.tight_layout()
save(fig, "opportunity_fusion_f1.pdf")

# Figs: attack curves (FedAvg HFL, VFL coordinator)
def attack_panel(agg, fname):
    fig, axes = plt.subplots(1, 4, figsize=(PAGE_W, PAGE_W * 0.26), sharey=True)
    for ax, (fusion, eps) in zip(axes, [("Intermediate", 20.0), ("Intermediate", 100.0),
                                        ("Late", 20.0), ("Late", 100.0)]):
        for atk in ATTACKS:
            ys = [e08.get((agg, fusion, atk, r, eps), (np.nan,))[0] for r in RATIOS]
            ax.plot(RATIOS, ys, marker="o", markersize=3.5, linewidth=1.2,
                    color=ATTACK_COLORS[atk], label=ATTACK_LABELS[atk])
        floor_line(ax)
        ax.set_title(f"{fusion}, ε={eps:g}", fontsize=9)
        ax.set_xticks(RATIOS); ax.set_xticklabels(["10%", "25%", "50%", "75%"])
        ax.set_xlabel("Malicious ratio"); ax.set_ylim(0, 0.30)
    axes[0].set_ylabel("F1 (Macro)"); axes[-1].legend(loc="upper right", fontsize=7.5)
    fig.tight_layout(); save(fig, fname)

attack_panel("fedavg", "f1_by_attack_hfl.pdf")
attack_panel("coordinator", "f1_by_attack_vfl.pdf")

# Figs: aggregator comparison (8)
for atk in ATTACKS:
    for fusion in ["Intermediate", "Late"]:
        fig, axes = plt.subplots(1, 2, figsize=(PAGE_W, PAGE_W * 0.30), sharey=True)
        for ax, eps in zip(axes, [20.0, 100.0]):
            for agg in AGGS + ["coordinator"]:
                ys, es = zip(*[e08.get((agg, fusion, atk, r, eps), (np.nan, np.nan)) for r in RATIOS])
                ax.errorbar(RATIOS, ys, yerr=es, marker=AGG_MARKERS[agg], markersize=4,
                            capsize=2.5, linewidth=1.3, color=AGG_COLORS[agg], label=AGG_LABELS[agg])
            floor_line(ax)
            ax.set_title(f"ε = {eps:g}", fontsize=9)
            ax.set_xticks(RATIOS); ax.set_xticklabels(["10%", "25%", "50%", "75%"])
            ax.set_xlabel("Malicious ratio"); ax.set_ylim(0, 0.30)
        axes[0].set_ylabel("F1 (Macro)"); axes[1].legend(loc="upper right", fontsize=7.5, ncol=2)
        fig.tight_layout(); save(fig, f"agg_{atk}_{fusion.lower()}.pdf")

# Figs: floor-adjusted heatmaps
def heatmap(fusion, fname):
    methods = AGGS + ["coordinator"]
    fig, axes = plt.subplots(2, 1, figsize=(COL_W, COL_W * 1.25))
    for ax, eps in zip(axes, [20.0, 100.0]):
        M = np.full((len(methods), len(ATTACKS) * len(RATIOS)), np.nan)
        for i, agg in enumerate(methods):
            for j, (atk, r) in enumerate([(a, r) for a in ATTACKS for r in RATIOS]):
                M[i, j] = e08.get((agg, fusion, atk, r, eps), (np.nan,))[0] - FLOOR
        im = ax.imshow(M, cmap="RdYlGn", vmin=-0.05, vmax=0.15, aspect="auto")
        ax.set_yticks(range(len(methods))); ax.set_yticklabels([AGG_LABELS[a] for a in methods], fontsize=7)
        ax.set_xticks(np.arange(1.5, 16, 4)); ax.set_xticklabels([ATTACK_LABELS[a] for a in ATTACKS], fontsize=7)
        for x in [3.5, 7.5, 11.5]:
            ax.axvline(x, color="white", linewidth=1.5)
        ax.set_title(f"ε = {eps:g}  (cols per attack: 10/25/50/75%)", fontsize=8)
    fig.colorbar(im, ax=axes, shrink=0.85, label="Floor-adjusted F1")
    save(fig, fname)

heatmap("Intermediate", "heatmap_intermediate.pdf")
heatmap("Late", "heatmap_late.pdf")

# Fig: delta_topo + win rate
fig, ax = plt.subplots(figsize=(COL_W, COL_W * 0.80))
bar_w, summary = 0.35, {}
for k, eps in enumerate([20.0, 100.0]):
    deltas = []
    for atk in ATTACKS:
        cells = []
        for fusion in ["Intermediate", "Late"]:
            for r in RATIOS:
                v = e08.get(("coordinator", fusion, atk, r, eps), (np.nan,))[0]
                h = max(e08.get((a, fusion, atk, r, eps), (np.nan,))[0] for a in AGGS)
                if not (np.isnan(v) or np.isnan(h)):
                    cells.append(v - h)
        deltas.append(cells)
        summary[(atk, eps)] = (statistics.mean(cells), sum(d > 0 for d in cells), len(cells))
    xs = np.arange(len(ATTACKS)) + (k - 0.5) * bar_w
    ax.bar(xs, [statistics.mean(c) for c in deltas], bar_w,
           color=[ATTACK_COLORS[a] for a in ATTACKS],
           alpha=(0.55 if eps == 20.0 else 1.0), label=f"ε = {eps:g}")
    for x, c in zip(xs, deltas):
        ax.text(x, max(statistics.mean(c), 0) + 0.004, f"{sum(d>0 for d in c)}/{len(c)}",
                ha="center", fontsize=7)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_xticks(range(len(ATTACKS))); ax.set_xticklabels([ATTACK_LABELS[a] for a in ATTACKS], fontsize=8)
ax.set_ylabel(r"$\Delta_{topo}$ = F1(VFL) − F1(best HFL)")
ax.legend(loc="upper right", fontsize=8); fig.tight_layout()
save(fig, "delta_topo_winrate.pdf")

# headline numbers
tw = tc = 0
for atk in ATTACKS:
    w = summary[(atk, 20.0)][1] + summary[(atk, 100.0)][1]
    n = summary[(atk, 20.0)][2] + summary[(atk, 100.0)][2]
    tw += w; tc += n
    print(f"  {atk:<12} win {w}/{n}")
print(f"  TOTAL win {tw}/{tc}")
print("Done. Figures in", FIGS)
