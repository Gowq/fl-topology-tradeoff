# Second multimodal HAR dataset for Exp. 07

Date: 2026-07-18

## Recommendation

Use **MHEALTH** as the second multimodal HAR dataset. It is the best practical
choice for an immediately reproducible experiment: the official UCI record
defines it as a benchmark for multimodal body sensing and provides 10 subjects,
12 activities, 3 synchronized body-worn devices, 23 input channels, no missing
values, a 72.1 MB download, DOI `10.24432/C5TW22`, and an explicit CC BY 4.0
license ([UCI dataset record](https://archive.ics.uci.edu/dataset/319/mhealth%2Bdatas)).
The associated original publication is Baños et al., *mHealthDroid* (2014),
DOI `10.1007/978-3-319-13105-4_14`
([University of Granada record](https://produccioncientifica.ugr.es/documentos/61a5263e37d5b2018338f2de)).

As a conservative primary-source usage signal, the UCI page showed 36,633
views on the access date, while PAMAP2 showed 46,081 and UCI HAR 240,985. This
supports calling MHEALTH an established public benchmark, but not claiming it
is the single most-used HAR dataset; page views are not citation counts. The
stronger wording "widely used" should therefore be accompanied by the official
benchmark record above rather than an unsupported bibliometric claim.

The earlier local shortlist was real: the manuscript and slides explicitly
named **PAMAP2 or MHEALTH**, while the slides singled out PAMAP2 as the next
step. That was a candidate list rather than a binding dataset decision. The
relevant local evidence is in
`mestrado/quali/overleaf/full/main.tex`,
`mestrado/quali/overleaf/manuscrito/monografia/paper_main_ieee.tex`, and
`mestrado/quali/slides/slides.md`.

## Why MHEALTH fits HFL and VFL

The official schema admits a defensible three-party VFL split without inventing
feature ownership:

| VFL party | Official channels |
|---|---|
| Chest device | accelerometer xyz + ECG leads 1 and 2 (5) |
| Left-ankle device | accelerometer, gyroscope, magnetometer xyz (9) |
| Right-wrist device | accelerometer, gyroscope, magnetometer xyz (9) |

All modalities are sampled at 50 Hz, labels and timestamps are aligned, and
each subject has a separate log file. These properties support both a
device-owned vertical partition and a subject-owned horizontal partition
without cross-device resampling
([official UCI schema and protocol](https://archive.ics.uci.edu/dataset/319/mhealth%2Bdatas)).

For HFL, one subject can represent one natural client, so the natural ceiling is
`N=10`. A subject-independent evaluation should reserve subjects for validation
and test; a clean sensitivity design is therefore a fixed nested training cohort
such as `N in {2, 4, 8}`, with the same held-out subjects at every `N`. Any
`N>10` result would require splitting a person into synthetic shards and must
not be described as a client-population scaling result.

For VFL, the natural number of parties is three. Splitting each device again by
sensor type can create more parties, but that changes the semantics and balance
of feature ownership; it is an ablation, not a natural `N` sweep. HFL client
count and VFL party count should therefore be reported as distinct axes rather
than implying that their `N` values denote the same entity.

## Required control for the topology claim

Replicating on MHEALTH addresses **dependence on OPPORTUNITY**, but a second
multimodal dataset alone does not separate topology from semantic sensor
redundancy. Exp. 07 should use identical windows and labels across these
conditions:

1. centralized/early-fusion control with all 23 channels;
2. HFL split by subject;
3. VFL split by the three physical devices above;
4. redundancy ablation: compare the physical-device split with a matched random
   channel split and/or leave-one-device-out evaluations.

If the VFL advantage persists under matched random partitions, that supports a
topology contribution. If it is concentrated in the physical-device split or
disappears when one device is removed, the result is conditioned on multimodal
redundancy. This qualification should be explicit regardless of outcome.

## Alternatives checked

| Dataset | Primary-source assessment | Decision |
|---|---|---|
| **PAMAP2** | 9 subjects, 18 recorded activities, 3 IMUs at wrist/chest/ankle (100 Hz), heart rate at about 9 Hz, 52 sensor attributes, missing values, 656.3 MB, CC BY 4.0 ([UCI](https://archive.ics.uci.edu/dataset/231/pamap2+physical+activity+monitoring)); introduced specifically as a public benchmark ([DFKI publication record](https://www.dfki.de/web/forschung/projekte-publikationen/publikation/6359)). | Scientifically strong and the prior leading candidate, but lower natural `N`, mixed sampling rates, missing-data handling, and a download about nine times larger make it less suitable for rapid reproducible implementation than MHEALTH. |
| **RealWorld HAR** | 15 subjects, 8 activities, 7 simultaneous body positions and 6 sensor types, about 3.5 GB ([University of Mannheim](https://www.uni-mannheim.de/dws/research/projects/activity-recognition/dataset/dataset-realworld/)); original paper DOI [`10.1109/PERCOM.2016.7456521`](https://doi.org/10.1109/PERCOM.2016.7456521). | Best natural `N` and richest location split, but rejected for this experiment: on 2026-07-18 the official ZIP returned HTTP 502, the official page contained a broken `TODO: LINK` citation, and no dataset license was stated there. Reconsider only with an approved stable mirror and clarified reuse terms. |
| **UCI HAR** | 30 subjects and a stable 58.2 MB CC BY 4.0 archive, but only one waist-mounted smartphone with accelerometer and gyroscope ([UCI](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)). | Excellent for natural high-`N` HFL, weak as a genuinely multi-device VFL/generalization test. |
| **WESAD** | 15 subjects and rich physiological/motion modalities from chest and wrist devices, but its targets are neutral, stress, and amusement rather than physical activities ([authors' dataset page](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html), [UCI record](https://archive.ics.uci.edu/dataset/465/wesad+wearable+stress+and+affect+detection)). | Rejected because changing from HAR to affect recognition introduces a task confound; reuse is also limited to scientific, non-commercial purposes by the authors' disclaimer. |

## Reproducibility requirements

- Download the official UCI archive and pin its checksum in the preparation
  script.
- Preserve the 50 Hz raw alignment before windowing; define window length,
  overlap, label handling, and null-class policy in the experiment manifest.
- Split validation/test by subject before creating windows to prevent overlap
  leakage.
- Report macro-F1, per-class support, all seeds, and both natural-client and
  synthetic-ablation results with unambiguous labels.
