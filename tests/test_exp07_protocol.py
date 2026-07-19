import ast
import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


CODE = Path(__file__).parents[1] / "experiments/exp07_mhealth_generalization/code"
GRID_SCRIPTS = Path(__file__).parents[1] / "scripts/grid"
sys.path.insert(0, str(CODE))

from mhealth_data import (
    MODALITY_COLUMNS,
    apply_sensor_trigger,
    select_client_subjects,
    window_subject,
)
from analyze_exp07 import completeness, summarize, tail_shape_flags
from protocol import build_protocol, smoke_protocol
from tune_exp07 import tuning_jobs


class ProtocolTests(unittest.TestCase):
    def test_grid_gpu_jobs_run_inside_a_slurm_job_step(self):
        for script_name in (
            "run_exp07_mhealth.sh",
            "run_exp07_mhealth_tuning.sh",
            "run_exp07_mhealth_tuning_smoke.sh",
        ):
            with self.subTest(script=script_name):
                source = (GRID_SCRIPTS / script_name).read_text(encoding="utf-8")
                self.assertIn("srun --unbuffered bash scripts/run.sh", source)

    def test_private_training_enables_train_mode_before_opacus_wrap(self):
        source = (CODE / "run_exp07.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        private_train = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "private_train"
        )
        calls = {
            ast.unparse(node.func): node.lineno
            for node in ast.walk(private_train)
            if isinstance(node, ast.Call)
        }
        self.assertLess(calls["model.train"], calls["engine.make_private"])

    def test_tuning_is_partitioned_into_unique_single_seed_jobs(self):
        jobs = tuning_jobs()
        self.assertEqual(len(jobs), 72)
        self.assertEqual(len({job["job_id"] for job in jobs}), 72)
        self.assertEqual({job["topology"] for job in jobs}, {"hfl", "vfl"})

    def test_config_ids_are_unique_and_all_arms_are_present(self):
        configs = build_protocol()
        self.assertEqual(len(configs), 510)
        self.assertEqual(len(configs), len({config.config_id for config in configs}))
        self.assertEqual(
            {config.arm for config in configs},
            {"generalization", "scale", "robustness", "tail", "redundancy"},
        )
        scale = [config for config in configs if config.arm == "scale"]
        self.assertEqual({config.client_count for config in scale}, {2, 4, 8})
        self.assertTrue(
            all((config.attack == "none") == (config.attack_ratio == 0.0) for config in scale)
        )

    def test_smoke_covers_both_topologies_and_modern_aggregator(self):
        configs = smoke_protocol()
        self.assertEqual({config.topology for config in configs}, {"hfl", "vfl"})
        self.assertIn("fltrust", {config.aggregator for config in configs})
        self.assertEqual(
            {config.topology for config in configs if config.epsilon > 0},
            {"hfl", "vfl"},
        )

    def test_analysis_aggregates_seeds_and_flags_tail_increase(self):
        self.assertEqual(completeness([])["expected"], 510)

        def result(seed, epsilon, f1):
            config = {
                "config_id": f"tail-{epsilon}-{seed}",
                "arm": "tail",
                "topology": "hfl",
                "epsilon": epsilon,
                "seed": seed,
                "client_count": 8,
                "attack": "none",
                "attack_ratio": 0.0,
                "aggregator": "fedavg",
            }
            return {
                "experiment": "exp07_mhealth_generalization",
                "config": config,
                "final": {"f1_macro": f1, "backdoor_asr": 0.0},
            }

        rows = [
            result(42, 50.0, 0.2),
            result(123, 50.0, 0.2),
            result(42, 100.0, 0.1),
            result(123, 100.0, 0.1),
        ]
        summary = summarize(rows)
        self.assertEqual(len(summary), 2)
        self.assertEqual(summary[0]["seed_count"], 2)
        self.assertEqual(len(tail_shape_flags(summary)), 1)


class MhealthTests(unittest.TestCase):
    def _raw(self, rows=32):
        raw = np.zeros((rows, 24), dtype=float)
        raw[:, 23] = 1
        for columns in MODALITY_COLUMNS.values():
            raw[:, columns] = np.arange(rows)[:, None]
        return raw

    def test_windowing_uses_documented_modalities_and_contiguous_labels(self):
        subject = window_subject(self._raw(), 1, window_size=8, stride=4)
        self.assertEqual(set(subject.modalities), set(MODALITY_COLUMNS))
        self.assertEqual(subject.modalities["left_ankle"].shape[1:], (9, 8))
        self.assertTrue(np.all(subject.labels == 0))

    def test_transition_windows_are_dropped(self):
        raw = self._raw(16)
        raw[3:8, 23] = 2
        subject = window_subject(raw, 1, window_size=8, stride=8)
        self.assertEqual(len(subject), 1)

    def test_client_selection_uses_real_subjects_and_is_deterministic(self):
        subjects = [window_subject(self._raw(), sid, 8, 4) for sid in range(1, 9)]
        first = select_client_subjects(subjects, client_count=4, seed=42)
        second = select_client_subjects(subjects, client_count=4, seed=42)
        self.assertEqual([s.subject_id for s in first], [s.subject_id for s in second])
        self.assertEqual(len({s.subject_id for s in first}), 4)
        small = select_client_subjects(subjects, client_count=2, seed=42)
        self.assertTrue({s.subject_id for s in small} <= {s.subject_id for s in first})

    def test_trigger_changes_only_ankle_gyroscope_tail(self):
        subject = window_subject(self._raw(), 1, 8, 4)
        row = {name: values[:1] for name, values in subject.modalities.items()}
        triggered = apply_sensor_trigger(row, amplitude=2.0, fraction=0.25)
        np.testing.assert_array_equal(triggered["chest"], row["chest"])
        expected = row["left_ankle"][:, 3:6, -2:] + 2.0
        np.testing.assert_array_equal(triggered["left_ankle"][:, 3:6, -2:], expected)


if __name__ == "__main__":
    unittest.main()
