# Exp. 07 — MHEALTH multimodal generalization and reviewer audit

Exp. 07 tests whether the OPPORTUNITY findings survive on an independent,
device-partitioned multimodal HAR benchmark while addressing the code-related
items from Botacin R3. It is an extension experiment; no result is claimed
until the full preregistered matrix has completed.

## Dataset and leakage-safe split

MHEALTH provides 10 people, 12 activities, and three synchronized devices at
50 Hz. HFL ownership is by person; VFL ownership is by physical device:

- chest: acceleration + two ECG leads (5 channels);
- left ankle: acceleration + gyroscope + magnetometer (9 channels);
- right arm/wrist: acceleration + gyroscope + magnetometer (9 channels).

Subjects 1–8 are training participants, subject 9 is used only for
hyperparameter selection and the FLTrust trusted root, and subject 10 is the
untouched final test subject. Normalization is fitted on subjects 1–8 only.
Windows contain 128 samples (2.56 s) with 50% overlap. Null activity is removed;
transition windows with less than 80% label agreement are discarded.

Primary-source rationale and alternatives are recorded in
`docs/research/mhealth_dataset_selection.md`.

## Focused arms

| Arm | Question | Matrix |
|---|---|---|
| Generalization | Does the topology/DP curve replicate? | HFL/VFL × epsilon={0,.25,.5,.75,1,3,8,20} × 5 seeds |
| Scale | Does the result depend on the number of real clients? | HFL N={2,4,8} × epsilon={0,3,20} × clean/50% model replacement × 5 seeds |
| Robustness | Do modern attacks/aggregators alter the conclusion? | model replacement + clean-label sensor backdoor; FedAvg/FLTrust/FoolsGold; ratios 25/50%; 5 seeds |
| Tail | Are high-budget rebounds stable? | HFL/VFL × epsilon={50,100,150,200} × 5 seeds, fixed rounds |
| Redundancy | Is the effect topology or sensor redundancy? | centralized + physical/random VFL partitions + each leave-one-device-out ablation × epsilon={0,3,20} × 5 seeds |

The complete matrix has 510 independently resumable jobs. HFL `N` always
means distinct people; no synthetic clients are presented as a population
scaling result. VFL has three device parties, so HFL client count and VFL party
count are deliberately not conflated.

## Modern attack semantics

- `model_replacement`: a malicious HFL client scales its post-training update
  by `N / malicious_count`; the attack occurs after local DP, matching a
  Byzantine server-side threat model.
- `sensor_backdoor`: malicious subjects receive a localized trigger in the
  ankle gyroscope only on target-class training windows, with labels unchanged.
  The reported attack-success rate is the proportion of triggered non-target
  test windows predicted as the target class.

FLTrust and FoolsGold are HFL aggregators. They are not applied to VFL because
VFL device encoders do not produce homologous model updates. VFL is compared
against the sensor backdoor through its coordinator.

## Run order

```bash
bash scripts/prepare_data.sh

# Validate parsing/training paths first.
bash scripts/run.sh exp07 --smoke-only --config-index 0 --device cpu

# Select HFL/VFL parameters independently on subject 9.
bash scripts/run.sh exp07-tune --device cuda

# Inspect or run one preregistered job.
bash scripts/run.sh exp07 --list
bash scripts/run.sh exp07 --config-index 0 --device cuda \
  --hyperparameters-file experiments/exp07_mhealth_generalization/results/tuned_hyperparameters.json

# GridUNESP/Pegasus.
sbatch scripts/grid/run_exp07_mhealth_tuning.sh
sbatch scripts/grid/run_exp07_mhealth.sh

# Validate all 510 results, aggregate five-seed confidence intervals, and flag
# unexpected high-budget curve reversals for inspection.
bash scripts/run.sh exp07-analyze
```

Run tuning to completion before submitting the 510-job array. Each job writes
one JSON result atomically and exits successfully when that result already
exists. There is no patience-based stopping: all curve and tail cells consume
the same fixed number of rounds.

## Metrics and interpretation guardrails

The runner records final and per-round accuracy, macro-F1, composed epsilon,
runtime, malicious subject IDs, chosen hyperparameters, and backdoor ASR. The
comparison must report uncertainty across all five seeds. The redundancy arm
tests whether an advantage survives a matched random channel partition and
quantifies how much each physical device contributes through leave-one-out
ablations; conclusions must remain conditional on those controls.
