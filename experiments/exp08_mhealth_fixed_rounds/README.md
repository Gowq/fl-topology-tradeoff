# Experiment 08 — MHEALTH Fixed-Round Replication

This experiment reproduces the diagnostic protocol from Exp. 06 on MHEALTH.
It isolates topology and fusion under a deliberately high privacy budget:

- HFL and VFL;
- intermediate and late fusion;
- epsilon 100 and 200 (`delta=1e-5`, max gradient norm 5);
- seeds 42, 123, and 456;
- exactly 25 communication rounds and three local epochs, without early stopping.

The 24 configurations use the subject-independent MHEALTH split and selected
hyperparameters from Exp. 07. The best round is reported post hoc together with
the final round; training never stops early based on test performance.

```bash
python experiments/exp08_mhealth_fixed_rounds/code/run_exp08.py --list
python experiments/exp08_mhealth_fixed_rounds/code/run_exp08.py --config-index 0
python experiments/exp08_mhealth_fixed_rounds/code/analyze_exp08.py \
  experiments/exp08_mhealth_fixed_rounds/results
```
