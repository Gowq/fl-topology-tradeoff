# Exp. 07 — corrected cross-dataset topology/attack comparison

This experiment replaces the original Exp. 07 protocol as the confirmatory
robustness study. The historical MHEALTH modern-attack run remains available in
Git history and in the project SILO, but it must not be used as a paired HFL/VFL
attack comparison: model replacement was HFL-only and its sensor backdoor was
not effective on MHEALTH.

## Confirmatory question

Does the VFL robustness advantage previously observed under parameter-space
attacks generalize from OPPORTUNITY to MHEALTH when HFL and VFL receive matched
participants, attack pressure, training, and a single global DP budget?

Label Flip is analyzed separately as a semantic attack. Generalization is
supported only if the seed-clustered 95% confidence interval for
`D_VFL - D_HFL` is above zero independently on both datasets, where
`D = final attacked macro-F1 - final clean macro-F1`.

## Controlled protocol

- datasets: OPPORTUNITY and MHEALTH;
- topologies: HFL and VFL with exactly eight participants/silos;
- fusion: intermediate and late;
- attacks: Label Flip, Sign Flip, Scaling and Free-Rider;
- attack ratios: 25% and 75%;
- privacy: no-DP control plus global `epsilon={20,100}`, `delta=1e-5`, `C=5`;
- training: exactly 25 rounds and three local epochs, no early stopping;
- primary comparison: HFL/FedAvg versus VFL/coordinator, five paired seeds;
- secondary HFL aggregators: Krum, Trimmed Mean and coordinate-wise Median,
  three paired seeds.

The resulting matrix contains 2,052 independently resumable configurations:
1,080 primary and 972 secondary jobs. Every attacked cell has a clean control
with identical dataset, topology, fusion, DP target, aggregator and seed.

## Eight-party mappings

MHEALTH HFL uses subjects 1--8 for training and holds subject 10 out for the
final evaluation; subject 9 is unused. Its VFL silos are chest acceleration,
chest ECG, arm acceleration/gyroscope/magnetometer and ankle
acceleration/gyroscope/magnetometer.

OPPORTUNITY HFL deterministically divides each of subjects 1--4 (runs Drill,
ADL1 and ADL2) into two contiguous clients, and evaluates on subject 2 runs
ADL4/ADL5. Its VFL mapping splits the 30 body-worn channels into four
contiguous groups and the object and ambient blocks into two groups each; the
`upper`/`lower` names denote contiguous channel blocks, not an anatomical
split. The architecture has the same eight branches in both topologies;
topology changes ownership and message flow, not model capacity.

## Attack equivalence

- Label Flip corrupts the exact same example IDs in HFL and VFL. The victim
  examples are those owned by the preregistered malicious HFL participants.
- HFL Sign Flip and Scaling modify transmitted model updates, never absolute
  model weights. VFL modifies the corresponding transmitted embedding
  (intermediate fusion) or logits (late fusion).
- HFL Free-Rider skips local training and transmits a zero update. VFL freezes
  the malicious local branch but still sends its valid representation/logits.
- Participant assignments are paired across topologies and approximately
  balanced across the five seeds.

Krum and the coordinate-wise aggregators are still run in the 75% stress-test
regime, but every result records whether the method's Byzantine tolerance
condition is satisfied. No theoretical robustness guarantee is claimed where
that flag is false.

## Privacy accounting

Opacus/RDP calibrates the noise multiplier from the sampling rate and planned
steps. HFL clients contain disjoint records and use parallel composition. VFL
silos touch attributes of the same records, so their trainable mechanisms are
sequentially composed. Thus both topologies receive the same global target
epsilon rather than a full epsilon per VFL silo. The no-DP control is serialized
as `epsilon=null`, not described as a privacy budget of zero.

## Commands

```bash
bash scripts/run.sh exp07 --list
bash scripts/run.sh exp07 --smoke-only --config-index 0 --device cpu
bash scripts/run.sh exp07 --config-index 0 --device cuda
bash scripts/run.sh exp07-analyze
```

