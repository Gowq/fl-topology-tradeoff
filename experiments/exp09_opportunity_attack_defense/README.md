# Experiment 09 — OPPORTUNITY Attack/Defense Transfer

This experiment transfers the new Exp. 07 robustness scenario back to the
original OPPORTUNITY benchmark while preserving its canonical train/test
partition and intermediate-fusion architecture.

The 210 configurations contain:

- HFL model-replacement and clean-label sensor-backdoor attacks;
- HFL FedAvg, FLTrust, and FoolsGold aggregation;
- VFL clean-label sensor backdoor with its coordinator (HFL update aggregators
  are not meaningful for vertically partitioned feature silos);
- attack ratios 25% and 50%, epsilon 0, 3, and 20, and five seeds;
- 25 rounds, one local epoch, `delta=1e-5`, and gradient norm 5.

Dataset-specific backdoor adaptation is explicit. OPPORTUNITY class 0 is the
dominant null class, so target class 1 is used. The trigger adds four training
standard deviations to three channels in the lower-body sensor group during
the final 20% of a window. This preserves the semantics of the normalized
MHEALTH ankle trigger without using a raw-unit amplitude across incompatible
sensors. FLTrust uses subject 1 / ADL3 as a disjoint trusted root; training
continues to use Drill/ADL1/ADL2 and testing uses subject 2 / ADL4/ADL5.

```bash
python experiments/exp09_opportunity_attack_defense/code/run_exp09.py --list
python experiments/exp09_opportunity_attack_defense/code/run_exp09.py --config-index 0
python experiments/exp09_opportunity_attack_defense/code/analyze_exp09.py \
  experiments/exp09_opportunity_attack_defense/results
```
