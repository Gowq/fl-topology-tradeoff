# Exp05 Attack Early-Stopping Audit

This diagnostic extends Exp05 to test whether the stopping protocol changes the
observed attack resistance of HFL and VFL.

It intentionally does not repeat the full Exp03 matrix. The compact sweep covers:

- Topologies and fusions: HFL/VFL with Intermediate and Late fusion.
- Attack channels: Label Flip as semantic corruption, Sign Flip as parameter-space corruption.
- Attack ratios: 25% and 75%.
- Privacy budgets: epsilon 100 and 200.
- Stop protocols: patience-based early stopping and fixed 25-round training.
- Seeds: 42, 123, 456.

Run locally or on Pegasus smoke:

```bash
python experiments/exp02_dp_frontier/code/05_FL_AttackEarlyStop_Audit_v25.py --smoke-only
```

Grid entry point:

```bash
sbatch scripts/grid/run_exp05_attack_earlystop_audit.sh
```
