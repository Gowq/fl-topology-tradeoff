# Exp08 VFL Frontier Grid Results

Final GridUNESP output for the corrected VFL coordinator branch:
`fix/vfl-dp-composition`.

- Slurm array: `4654304_[0-3]`
- Completed: 2026-06-12 01:26:12 -03
- Source on Grid: `~/grid/fl-topology-tradeoff/results/`
- Raw parts: `exp08_robustagg_dp_frontier_part3{a,b,c,d}.json`
- Canonical merged file in this repo:
  `../vfl_coordinator.json`

Summary against the existing HFL frontier results:

| Metric | Value |
|---|---:|
| VFL configs | 64 |
| VFL seed runs | 192 |
| VFL mean F1 | 0.0984 |
| VFL Intermediate mean F1 | 0.1019 |
| VFL Late mean F1 | 0.0949 |
| HFL mean F1 across four aggregators | 0.0541 |
| Best-HFL paired cells won by VFL | 55/64 |
| Parameter-space cells won by VFL | 48/48 |
| Label Flip cells won by VFL | 7/16 |
| Mean paired delta vs best HFL | +0.0251 |
| One-sided paired sign-test p-value | 1.77e-09 |

Attack-level paired deltas against the best HFL aggregator per cell:

| Attack | VFL wins | Mean delta F1 |
|---|---:|---:|
| Sign Flip | 16/16 | +0.0476 |
| Scaling | 16/16 | +0.0248 |
| Free-Rider | 16/16 | +0.0397 |
| Label Flip | 7/16 | -0.0116 |

Interpretation: the corrected run preserves the mechanism-scoped claim. VFL is
systematically stronger for parameter-space attacks, while Label Flip remains
the semantic-channel boundary where the passive coordinator no longer dominates.
