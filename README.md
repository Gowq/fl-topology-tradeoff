# How Topology Conditions the Privacy–Robustness Trade-off in Multimodal Federated Learning

Reproducibility artifact for the paper of the same title. It contains the code,
the result files, and a figure generator for every experiment reported in the
paper, on **OPPORTUNITY** (18-class multimodal Human Activity Recognition) and
**CIFAR-10/100** image baselines. Corrected Exp. 07 adds a preregistered,
cross-dataset robustness comparison on **OPPORTUNITY** and the independent
**MHEALTH** multimodal HAR benchmark.

The study jointly evaluates Horizontal (HFL) and Vertical (VFL) federated
learning across four axes — accountant-calibrated Differential Privacy, four
untargeted Byzantine attacks at ratios up to 75 %, four aggregation schemes vs.
VFL's passive coordinator, and operational overhead — plus a preliminary
exploration of defenses for the semantic-attack failure mode.

---

## TL;DR — reproduce the figures in one command

The result files are committed, so every figure and headline number in the
paper can be regenerated **without rerunning any experiment or owning a GPU**:

The paper-to-artifact map, complete protocol matrix, and hyperparameters are
documented in [`docs/paper_reproducibility.md`](docs/paper_reproducibility.md).

```bash
# with Docker (recommended for cluster runs)
docker compose run --rm repro-gpu list

# or natively (Python 3.10+)
pip install -r requirements.txt
python figures/generate_figures.py     # writes figures/*.pdf from experiments/*/results
```

`figures/generate_figures.py` prints the headline numbers as it runs, e.g. the
paired topology win rate `TOTAL win 55/64`. The paper/manuscript must use these
shipped artifact results as the canonical source for all experimental figures;
older workspace copies such as `mestrado/experiments/results` are not
authoritative.

---

## Installation

### Option A — Docker / cluster image

The Docker image is GPU-first and intended for Pegasus/Grid/cluster execution.
It uses a CUDA/PyTorch base image and expects datasets to be mounted under
`./data`.

**Build the cluster image:**

```bash
docker compose build repro-gpu
```

**Run a smoke or a block of Exp. 07:**

```bash
docker compose run --rm repro-gpu exp07 --smoke-only --config-index 0 --device cuda
docker compose run --rm repro-gpu exp07 --config-start 0 --config-count 6 --device cuda
```

**Export for another cluster:**

```bash
bash scripts/docker_build_cluster_image.sh
SAVE_TAR=1 bash scripts/docker_build_cluster_image.sh
```

More details are in `docs/docker_cluster.md`. Set `PREPARE_DATA=1` only if the
cluster node is allowed to download public datasets itself.

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
- **MHEALTH** — downloaded from UCI (#319), checksum-verified, and unzipped to
  `data/MHEALTHDATASET/` for Exp. 07.
- **CIFAR-10/100** — fetched by torchvision (CIFAR-10 pre-fetched by the script;
  CIFAR-100 downloads on first use).

```bash
bash scripts/prepare_data.sh
```

The primary metric is **F1-macro**: OPPORTUNITY is severely imbalanced (a
majority-class predictor scores ≈ 0.85 accuracy on the null class), so the
random-prediction floor is `1/18 ≈ 0.056` (18 classes, including null). Results
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
  exp01_baseline/                 # Paper Exp. 01 — Privacy–Utility Baseline
  exp02_dp_frontier/              # Paper Exp. 02 — DP Budget-Behavior Analysis
  exp03_attacks_aggregation/      # Paper Exp. 03 — Byzantine Attacks × Aggregation under Calibrated DP
  exp04_overhead/                 # Paper Exp. 04 — Operational Overhead
  exp02_dp_frontier/              # Exp. 05/06 diagnostics live alongside Exp. 02
  exp07_corrected_attacks/        # Exp. 07 — corrected cross-dataset attack comparison
  exp08_timetrojan_topology/      # Exp. 08 — TimeTrojan upstream backdoor topology comparison
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

### Exp. 02 — DP Budget-Behavior Analysis  (`exp02_dp_frontier`)
**What:** extends the baseline to ε ∈ {10,20,50,100,200}, OPPORTUNITY HFL+VFL ×
{Intermediate, Late}, 3 seeds.
**Shows:** Useful utility re-emerges only at high budgets. VFL crosses 2× the
floor at ε = 20 and reaches **0.19** at ε = 100; HFL reaches **0.15** at ε = 100
and **0.20** at ε = 200. VFL dominates the privacy-relevant range (ε ≤ 100);
even at ε = 200 both recover < half their no-DP utility.
```bash
bash scripts/run.sh exp02
```

### Exp. 03 — Byzantine Attacks × Aggregation under Calibrated DP  (`exp03_attacks_aggregation`)
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

### Exp. 05 — Attack Early-Stopping Audit  (`exp02_dp_frontier`)

**What:** a compact audit (HFL/VFL × {Intermediate, Late} × {Label Flip, Sign Flip} ×
ratios {25,75}% × ε ∈ {100,200} × {early, fixed}, 3 seeds) reruns each cell with
patience-based early stopping **on** and **off** to test whether the stopping
protocol influences the observed DP-and-attack resistance. Data ship in
`experiments/exp02_dp_frontier/results/attack_earlystop_audit/`
(`exp05_attack_earlystop_audit_part{a..h}.json`); the figure is rendered as
`audit_signflip_earlystop.pdf`.
**Shows:** the topology ranking is **protocol-invariant** — removing early
stopping does not change who wins. Under Sign Flip (parameter-space corruption)
VFL beats HFL in **16/16** matched cells under *both* protocols, holding
**0.157 mean F1 ≈ 2.8× the floor** while HFL sits **at the floor (0.99×)** and
drops below it under heavy attack. Early stopping only compresses VFL's margin
(it halts before VFL's slow climb peaks); it never reverses the ranking. Under
Label Flip the semantic boundary holds regardless of protocol (both at floor).
```bash
python experiments/exp02_dp_frontier/code/05_FL_AttackEarlyStop_Audit_v25.py --smoke-only
sbatch scripts/grid/run_exp05_attack_earlystop_audit.sh
```

### Exp. 06 — Fixed-Round High-Budget Diagnostic  (`exp02_dp_frontier`)

**What:** reruns the high-budget OPPORTUNITY frontier tail at ε ∈ {100,200}
with patience/loss-stagnation early stopping disabled, while preserving the same
DP calibration, seeds, topology/fusion grid, and best-checkpoint reporting.
These clean-tail runs are reported separately from the early-stopped Exp. 01/02
fusion-mode curves rather than being spliced into them.
**Purpose:** determine whether the small HFL-over-VFL crossover at ε = 200 is a
topology effect or an artifact of VFL saturating and stopping before consuming
the target privacy budget. The shipped GridUNESP and Pegasus repetitions are in
`experiments/exp02_dp_frontier/results/fixed_round_tail/` and are rendered by
`figures/generate_figures.py` as `opportunity_fixed_round_high_budget.pdf`. Run
`--smoke-only` before Pegasus/Grid deployment.
```bash
bash scripts/run.sh exp06-fixed --smoke-only
bash scripts/run.sh exp06-fixed --part all
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

### Exp. 07 — Corrected Cross-Dataset Attack Comparison

**What:** a controlled eight-participant comparison of HFL and VFL on MHEALTH
and OPPORTUNITY under Label Flip, Sign Flip, Scaling and Free-Rider. It uses
matched intermediate/late fusion, fixed 25-round training, global topology-aware
DP budgets, five paired primary seeds, and three-seed Krum/Trimmed Mean/Median
secondary arms.

**Status:** code and preregistered 2,052-job matrix are included; new results
must not be mixed with the superseded modern-attack Exp. 07 run. See
`experiments/exp07_corrected_attacks/README.md`.

```bash
for i in 0 1 2 3 4; do
  bash scripts/run.sh exp07 --smoke-only --config-index "$i" --device cpu
done
bash scripts/run.sh exp07 --list
bash scripts/run.sh exp07-analyze
```

### Exp. 08 — TimeTrojan Topology Comparison

**What:** a controlled HFL/VFL comparison under the same upstream
TimeTrojan-FGSM dirty-label time-series backdoor. Attack artifacts are generated
once per dataset/seed with a centralized Exp. 06-style surrogate, frozen, and
then reused by HFL and VFL. The primary endpoint is `ASR_uplift`; FLTrust and
FoolsGold are included as secondary HFL-only robust aggregation arms.

**Status:** code and the complete 384-job matrix are included. OPPORTUNITY
passes the preregistered transfer/utility gate; MHEALTH fails the clean-utility
criterion and is therefore exploratory rather than confirmatory. Prepare the 10
attack artifacts before launching poisoned federated jobs. See
`experiments/exp08_timetrojan_topology/README.md`.

```bash
bash scripts/run.sh exp08 --list
bash scripts/run.sh exp08 --list-artifacts
bash scripts/run.sh exp08 --prepare-all-artifacts --device cuda
bash scripts/run.sh exp08 --config-index 0 --device cuda
bash scripts/run.sh exp08-analyze --allow-partial
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
