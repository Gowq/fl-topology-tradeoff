# Paper reproducibility guide

This document is the detailed companion to *Federated Learning Topology as a
Security-Utility Trade-off*. It keeps implementation details and the complete
hyperparameter matrix outside the page-limited manuscript while making every
reported result traceable to code and committed outputs.

## Paper-to-artifact map

| Paper evidence | Code | Committed outputs |
|---|---|---|
| Exp. 01: topology/fusion baseline | `experiments/exp01_baseline/code/` | `experiments/exp01_baseline/results/` |
| Exp. 02: privacy-utility frontier | `experiments/exp02_dp_frontier/code/02_FL_Frontier_v24.py` | `experiments/exp02_dp_frontier/results/` |
| Exp. 03: Byzantine attacks and aggregation | `experiments/exp03_attacks_aggregation/code/` | `experiments/exp03_attacks_aggregation/results/` |
| Exp. 04: operational overhead | `experiments/exp04_overhead/code/` | `experiments/exp04_overhead/results/` |
| Exp. 05/06: stopping-rule diagnostics | `experiments/exp02_dp_frontier/code/05_FL_Frontier_FixedRounds_v25.py` | `experiments/exp02_dp_frontier/results/fixed_round_tail/` |
| Exp. 07: corrected cross-dataset attack comparison | `experiments/exp07_corrected_attacks/code/` | `experiments/exp07_corrected_attacks/results/` |
| Exp. 08: TimeTrojan topology comparison | `experiments/exp08_timetrojan_topology/code/` | `experiments/exp08_timetrojan_topology/results/` |

The scripts named `06_FL_Frontier*.py` are compatibility wrappers for the
renamed Exp. 02 and Exp. 05 entry points. Raw cluster transfers and logs are not
part of the scientific artifact.

## Figure index

`figures/opportunity_privacy_utility_overlay.pdf` is the compact overlay used
by the journal manuscript: early-stopped curves and fixed-round points share
one axis. Its two-panel predecessor remains as
`figures/opportunity_privacy_utility_combined.pdf`. The complete fusion-mode
views remain available as `figures/opportunity_fusion_f1.pdf` and
`figures/opportunity_fixed_round_high_budget.pdf`.

Detailed aggregation matrices removed from the page-limited manuscript remain
available as `figures/heatmap_intermediate.pdf` and
`figures/heatmap_late.pdf`. The stopping-rule audit shown in the manuscript for
Intermediate fusion has its complete Late-fusion counterpart in
`figures/audit_late_signflip.pdf`. All figures are regenerated from committed
results by `figures/generate_figures.py`.

## Related-work search audit

The manuscript reports only the databases, date/language/field restrictions,
query themes, and coverage totals. The exact Boolean string and inclusion
accounting live in `scripts/verify_search_coverage.py`. Run it with
`--print-query` to reproduce the query or with `--bibtex` against the paper's
`references.bib` to verify the 37 query matches, four manual preprints, and
three backward-snowballed works in the 44-paper reviewed corpus.

## Shared training configuration

| Parameter | Exp. 01–03 | Exp. 04 | Exp. 07–08 |
|---|---:|---:|---:|
| Federated rounds | 25 | 10 | 25 |
| Local epochs per round | 3 | 3 | 3 |
| Optimizer | SGD | SGD | SGD |
| Learning rate | 0.01 | 0.01 | 0.01 |
| Momentum | 0.9 | 0.9 | 0.9 |
| Batch size | 16 | 16 | 64 |
| Dropout, OPPORTUNITY | 0.5 | 0.5 | 0.1 |
| Dropout, CIFAR controls | 0.25 | — | — |
| Embedding/hidden dimensions | implementation-specific | implementation-specific | 64/128 |
| Early stopping | patience 3 after warm-up | none | none |
| Primary seeds | 42, 123, 456 | 42 | 42, 123, 456, 789, 2026 |
| Secondary-defense seeds | 42, 123, 456 | — | 42, 123, 456 |

Exp. 01–03 use 12 HFL clients and four VFL modality silos on OPPORTUNITY;
CIFAR controls use ten HFL clients. Exp. 07–08 hold the comparison at eight
participants in both HFL and VFL. MHEALTH trains on subjects 1–8 and tests on
subject 10; OPPORTUNITY Exp. 07–08 trains on S1/S3/S4 and tests on S2.

## Differential privacy

All private cells use topology-aware calibration through the shared RDP
utilities in `experiments/shared/code/experiment_validity.py`, with
`delta=1e-5` and clipping norm `C=5`. Exp. 01 evaluates no DP and epsilon in
{1, 5, 8}; Exp. 02 extends the frontier to {10, 20, 50, 100, 200}; Exp. 03 uses
{20, 100}; Exp. 04 compares no DP with epsilon 3; Exp. 07–08 use no DP and
epsilon in {20, 100}.

For Exp. 07–08, HFL uses Opacus Poisson sampling and the formal RDP accountant.
VFL composes the private branch/fusion mechanisms globally, but its shuffled
fixed-size loaders do not implement Poisson sampling. Consequently, reported
VFL epsilon is an accountant-calibrated approximation, not a formal DP bound.

## Attack and aggregation matrices

Exp. 03 evaluates Label Flip, Sign Flip, Scaling, and Free-Rider attacks under
FedAvg and Krum on OPPORTUNITY. Exp. 07 repeats those four attacks on MHEALTH and
OPPORTUNITY at malicious-participant ratios 0.25 and 0.75. Its primary arm uses
FedAvg/coordinator with five paired seeds; its HFL-only secondary arm uses Krum,
Trimmed Mean, and coordinate-wise Median with three seeds. The committed Exp. 07
matrix contains 2,052 unique configurations.

Exp. 08 uses a 5% dirty-label TimeTrojan-FGSM transfer attack targeting class 1.
Each of ten dataset/seed attack artifacts uses five temporal positions,
`eta=2.0`, and ten refinement iterations, and is frozen before the paired
HFL/VFL runs. Its five-seed primary arm compares FedAvg with the VFL coordinator;
the three-seed HFL-only secondary arm evaluates FLTrust and FoolsGold. The
committed matrix contains 384 unique configurations.

The preregistered Exp. 08 gate passes on OPPORTUNITY. MHEALTH fails the maximum
five-percentage-point clean-utility-loss criterion (10.9-point HFL loss), so its
backdoor evidence is exploratory. Artifact hashes match in all topology pairs.

## Reproduction commands

Install dependencies with `pip install -r requirements.txt`, prepare the public
datasets with `bash scripts/prepare_data.sh`, and use the common dispatcher:

```bash
bash scripts/run.sh figures
bash scripts/run.sh exp07 --list
bash scripts/run.sh exp07-analyze
bash scripts/run.sh exp08 --list
bash scripts/run.sh exp08-analyze
```

The figure command rebuilds the manuscript figures from committed results and
does not require a GPU. Full experiment sweeps are computationally expensive;
cluster and container instructions are in `docs/docker_cluster.md`.

## Integrity checks

The checked-in analysis reports assert exact matrix coverage: Exp. 07 contains
2,052/2,052 configurations (1,080 primary and 972 secondary), and Exp. 08
contains 384/384 configurations (240 primary and 144 secondary), with no missing
or extra identifiers. Protocol and analysis invariants are covered by
`tests/test_exp07_corrected.py` and `tests/test_exp08_timetrojan.py`.
