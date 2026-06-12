# How Topology Conditions the Privacy–Robustness Trade-off in Multimodal Federated Learning

Reproducibility artifact for the paper of the same title. It contains the code,
the result files, and a figure generator for every experiment reported in the
paper, on **OPPORTUNITY** (18-class multimodal Human Activity Recognition) and
**CIFAR-10/100** image baselines.

The study jointly evaluates Horizontal (HFL) and Vertical (VFL) federated
learning across four axes — accountant-calibrated Differential Privacy, four
untargeted Byzantine attacks at ratios up to 75 %, four aggregation schemes vs.
VFL's passive coordinator, and operational overhead — plus a preliminary
exploration of defenses for the semantic-attack failure mode.

---

## TL;DR — reproduce the figures in one command

The result files are committed, so every figure and headline number in the
paper can be regenerated **without rerunning any experiment or owning a GPU**:

```bash
# with Docker (recommended)
docker compose run --rm repro          # prepares data + writes figures/*.pdf

# or natively (Python 3.10+)
pip install -r requirements.txt
python figures/generate_figures.py     # writes figures/*.pdf from experiments/*/results
```

`figures/generate_figures.py` prints the headline numbers as it runs, e.g. the
paired topology win rate `TOTAL win 55/64`.

---

## Installation

### Option A — Docker (self-contained)

Two levels of reproduction are supported from the same image family:

**1. Rebuild the figures from the shipped results (fast, CPU, no GPU/data):**
```bash
docker compose run --rm repro            # default target: figures
```

**2. Reproduce the experiments themselves.** The entrypoint downloads the
datasets, then runs the requested experiment and writes fresh result files into
`experiments/<exp>/results/` (the same files the figures are built from):
```bash
docker compose run --rm repro smoke      # fast end-to-end stack check (1 seed, CPU)
docker compose run --rm repro exp01      # reproduce one experiment on CPU
```

**GPU reproduction (recommended for the full sweep).** Build the CUDA image and
use the `repro-gpu` service (needs the NVIDIA Container Toolkit); device
selection is automatic:
```bash
docker compose build repro-gpu
docker compose run --rm repro-gpu exp03  # the 320-config attack/aggregation sweep
docker compose run --rm repro-gpu all    # reproduce exp01-04 + figures end to end
```

Targets accepted by both services: `figures` (default), `smoke`, `exp01`/`exp02`/
`exp03`/`exp04`, `defense-semantic`, `defense-losses`, `all`. Experiment runs are
heavy — Exp.03 is 960 runs and is intended for a GPU host; `smoke` and `figures`
run in minutes on a laptop.

### Option B — native

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies (see `requirements.txt`): PyTorch, torchvision, **Opacus** (DP-SGD),
NumPy, pandas, SciPy, scikit-learn, matplotlib, tqdm.

---

## Data

Datasets are public and **not** committed; `scripts/prepare_data.sh` fetches them
into `./data`:

- **OPPORTUNITY** — downloaded from UCI (#226) and unzipped to
  `data/OpportunityUCIDataset/`.
- **CIFAR-10/100** — fetched by torchvision (CIFAR-10 pre-fetched by the script;
  CIFAR-100 downloads on first use).

```bash
bash scripts/prepare_data.sh
```

The primary metric is **F1-macro**: OPPORTUNITY is severely imbalanced (a
majority-class predictor scores ≈ 0.85 accuracy on the null class), so the
random-prediction floor is `1/19 ≈ 0.053` (18 activity classes + null). Results
near or below this floor indicate no real utility regardless of accuracy.

---

## Differential Privacy

DP-SGD is applied via Opacus with per-sample gradient clipping
(`max_grad_norm = 5.0`). The privacy guarantee ε is not set directly: each step
leaks a bounded amount of privacy, and the run-level guarantee is the
**composition** of all steps, determined jointly by the noise multiplier σ, the
subsampling rate, and the number of steps. The noise multiplier is obtained by
**inverting the Opacus RDP privacy accountant** so that the composed loss over
the full run meets the target ε at δ = 10⁻⁵. All budgets reported are calibrated
this way (`experiments/shared/code/experiment_validity.py`:
`calibrated_noise_multiplier`, `dp_plan_from_loaders`, `composed_epsilon`).

---

## Repository layout & mapping to the paper

```
experiments/
  shared/code/                    # DP calibration, timeout utils, helpers (used by all)
  exp01_baseline/                 # Paper Exp. 01 — privacy–utility baseline
  exp02_dp_frontier/              # Paper Exp. 02 — DP utility frontier (ε up to 200)
  exp03_attacks_aggregation/      # Paper Exp. 03 — attacks × aggregation under DP
  exp04_overhead/                 # Paper Exp. 04 — operational overhead
  defense_semantic_filtering/     # Preliminary — semantic defenses (future work)
  defense_robust_losses/          # Preliminary — noise-robust loss defenses (future work)
figures/generate_figures.py       # regenerates all paper figures from results/
scripts/prepare_data.sh           # downloads public datasets into ./data
scripts/run.sh                    # orchestrates experiments / figures
```

Each `experiments/<exp>/` has `code/` (runnable scripts) and `results/` (the
committed result files the figures are built from).

---

## The experiments

### Exp. 01 — Privacy–Utility Baseline  (`exp01_baseline`)
**What:** F1 vs. ε ∈ {0,1,3,5,8} without attackers. OPPORTUNITY in HFL
(Early/Intermediate/Late) and VFL (Intermediate/Late, 4 modal silos); CIFAR-10
and CIFAR-100 in HFL. 3 seeds, 25 rounds.
**Shows:** Without DP, topology matters — VFL Intermediate **0.54** vs. HFL
Intermediate **0.43**. Under calibrated DP, every conventional budget (ε ≤ 8)
collapses both topologies to the floor (best case ≈ 0.09 at ε = 8); CIFAR-10/100
reproduce their no-DP difficulty ordering (0.77 / 0.41) and both fall out of the
viable range under DP.
```bash
bash scripts/run.sh exp01
```

### Exp. 02 — DP Utility Frontier  (`exp02_dp_frontier`)
**What:** extends the baseline to ε ∈ {10,20,50,100,200}, OPPORTUNITY HFL+VFL ×
{Intermediate, Late}, 3 seeds.
**Shows:** Useful utility re-emerges only at high budgets. VFL crosses 2× the
floor at ε = 20 and reaches **0.19** at ε = 100; HFL reaches **0.15** at ε = 100
and **0.20** at ε = 200. VFL dominates the privacy-relevant range (ε ≤ 100);
even at ε = 200 both recover < half their no-DP utility.
```bash
bash scripts/run.sh exp02
```

### Diagnostic Exp. 06 — Fixed-Round Frontier Tail  (`exp02_dp_frontier`)
**What:** reruns the high-budget OPPORTUNITY frontier tail at ε ∈ {100,200}
with patience/loss-stagnation early stopping disabled, while preserving the same
DP calibration, seeds, topology/fusion grid, and best-checkpoint reporting.
**Purpose:** determine whether the small HFL-over-VFL crossover at ε = 200 is a
topology effect or an artifact of VFL saturating and stopping before consuming
the target privacy budget. Run `--smoke-only` before Pegasus/Grid deployment.
```bash
bash scripts/run.sh exp06-fixed --smoke-only
bash scripts/run.sh exp06-fixed --part all
```

### Exp. 03 — Byzantine Attacks × Aggregation under DP  (`exp03_attacks_aggregation`)
**What:** the central factorial sweep — 4 attacks (Label Flip, Sign Flip,
Scaling, Free-Rider) × ratios {10,25,50,75}% × {Intermediate, Late} ×
ε ∈ {20,100} × 5 methods (HFL FedAvg/Krum/Trimmed-Mean/Median + VFL coordinator).
**320 configurations, 960 runs, 3 seeds.** Identical ratio grids enable a paired
cell-by-cell comparison.
**Shows:** Topology is the dominant **relative** defense lever — VFL beats the
best per-cell HFL aggregator in **55/64** matched cells (paired sign test,
p < 10⁻⁹), and **48/48** under the parameter-space attacks (Sign Flip, Scaling,
Free-Rider), but only **7/16** under Label Flip, where both sit at the floor. The
shield is a *parameter-space* shield, not a universal one. No classical
aggregator dominates: Krum averages below the floor yet is the best HFL option in
21/64 cells (notably Sign Flip); Median has the worst single cell (0.002 at 50 %
Sign Flip).
```bash
bash scripts/run.sh exp03
```

### Exp. 04 — Operational Overhead  (`exp04_overhead`)
**What:** wall-time/round, peak GPU memory, parameter count across topology ×
fusion × ε, on an RTX 5080.
**Shows:** VFL without DP costs 1.6× HFL; DP-SGD multiplies wall-time 2.8–3.6×.
Worst case (VFL + DP) is 4.7× HFL (≈ 62 s/round) and ≤ 511 MB peak memory —
deployable on commodity hardware.
```bash
bash scripts/run.sh exp04
```

### Preliminary — Defenses for the semantic channel
Exp. 03 shows VFL's shield does not cover Label Flip. Two preliminary,
breadth-first sweeps explore whether that gap can be closed:

- **`defense_semantic_filtering`** — sample-selection / consistency defenses
  (small-loss filtering, leave-one-silo-out weighting, fusion-consistency
  dropout). 3 seeds, Intermediate axis. *Result:* none lifts Label Flip above the
  floor; selection-based defenses destroy the parameter-space shield (retention
  0.36–0.69×) while fusion-consistency preserves it (0.84–1.09×).
  ```bash
  bash scripts/run.sh defense-semantic --smoke-seeds 3
  ```
- **`defense_robust_losses`** — noise-robust training objectives (GCE, MAE, SCE,
  NCE+RCE, label smoothing, soft bootstrapping). 1-seed exploratory sweep.
  *Result (preliminary):* no objective rescues untargeted Label Flip; SCE and
  bootstrapping show promise against *targeted* label flipping. These are
  exploratory results guiding a deeper sweep, not claims.
  ```bash
  bash scripts/run.sh defense-losses --smoke-seeds 1
  ```

---

## Regenerating figures

```bash
bash scripts/run.sh figures        # or: python figures/generate_figures.py
```

Reads only `experiments/*/results/` and writes the paper's PDFs into `figures/`.
The figures shipped under `figures/` were produced this way.

---

## Notes on reproducibility

- Results were produced with 3 seeds (`{42, 123, 456}`) per configuration; each
  reported F1 is the mean of the per-seed best checkpoint.
- Runs that early-stop consume **less** than the target ε, so reported guarantees
  are conservative.
- Exact wall-times depend on hardware; the overhead numbers are from an
  NVIDIA RTX 5080.
