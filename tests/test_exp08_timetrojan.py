import sys
import types
import unittest
from pathlib import Path

import numpy as np


CODE = Path(__file__).parents[1] / "experiments/exp08_timetrojan_topology/code"
sys.path.insert(0, str(CODE))
for name in ("protocol", "data", "timetrojan", "aggregation", "analyze_exp08", "models"):
    sys.modules.pop(name, None)

from analyze_exp08 import (
    condition_effects,
    dataset_summary,
    gate_report,
    secondary_aggregator_summary,
    secondary_aggregator_effects,
    topology_effects,
    uplift_rows,
)
from data import OPPORTUNITY_SPLIT, WindowSet, concatenate
from protocol import artifact_specs, build_protocol, smoke_protocol
from timetrojan import Artifact, append_poison
from run_exp08 import _artifact_path, _prepare_clients

try:
    import torch

    from aggregation import FoolsGoldState, aggregate
    from timetrojan import _fgsm_trigger
except ImportError:
    torch = None


class ProtocolTests(unittest.TestCase):
    def test_matrix_is_unique_and_has_preregistered_arm_sizes(self):
        configs = build_protocol()
        self.assertEqual(len(configs), 384)
        self.assertEqual(len({config.config_id for config in configs}), 384)
        self.assertEqual(sum(config.arm == "primary" for config in configs), 240)
        self.assertEqual(sum(config.arm == "secondary" for config in configs), 144)

    def test_artifacts_are_generated_once_per_dataset_and_primary_seed(self):
        specs = artifact_specs()
        self.assertEqual(len(specs), 10)
        self.assertEqual({spec.dataset for spec in specs}, {"mhealth", "opportunity"})
        self.assertEqual({spec.seed for spec in specs}, {42, 123, 456, 789, 2026})

    def test_no_dp_is_not_serialized_as_epsilon_zero(self):
        configs = build_protocol()
        self.assertIn(None, {config.epsilon for config in configs})
        self.assertNotIn(0.0, {config.epsilon for config in configs})
        self.assertEqual({config.poison_rate for config in configs}, {0.0, 0.05})

    def test_smoke_covers_the_required_surfaces(self):
        smoke = smoke_protocol()
        self.assertEqual({config.dataset for config in smoke}, {"mhealth", "opportunity"})
        self.assertEqual({config.topology for config in smoke}, {"hfl", "vfl"})
        self.assertEqual({config.fusion for config in smoke}, {"intermediate", "late"})
        self.assertIn(20.0, {config.epsilon for config in smoke})
        self.assertIn(100.0, {config.epsilon for config in smoke})
        self.assertEqual({"fltrust", "foolsgold"} & {config.aggregator for config in smoke},
                         {"fltrust", "foolsgold"})


class ArtifactReuseTests(unittest.TestCase):
    def test_clean_control_uses_paired_poison_artifact_for_trigger_eval(self):
        clean_test = WindowSet({"x": np.zeros((2, 1, 1), dtype=np.float32)}, np.array([0, 2]),
                               np.array([1, 2]), np.array([10, 10]))
        triggered = WindowSet({"x": np.ones((1, 1, 1), dtype=np.float32)}, np.array([1]),
                              np.array([2]), np.array([10]))
        artifact = Artifact(
            {"target_class": 1, "poison_rate": 0.05},
            triggered,
            triggered,
            triggered,
            "hash",
            np.array([], dtype=np.int64),
        )

        import run_exp08

        original = run_exp08.load_artifact
        try:
            run_exp08.load_artifact = lambda path, template: artifact
            config = types.SimpleNamespace(
                dataset="mhealth", seed=42, target_class=1, poison_rate=0.0,
                is_clean=True,
            )
            args = types.SimpleNamespace(artifact_dir=Path("artifacts"))
            clients, triggered_test, loaded = _prepare_clients(config, [], clean_test, args)
        finally:
            run_exp08.load_artifact = original

        self.assertEqual(clients, [])
        self.assertIs(triggered_test, triggered)
        self.assertIs(loaded, artifact)

    def test_artifact_path_uses_paired_poison_artifact_even_for_clean_control(self):
        config = types.SimpleNamespace(dataset="mhealth", seed=42, target_class=1, poison_rate=0.0)
        args = types.SimpleNamespace(artifact_dir=Path("artifacts"))
        self.assertEqual(
            _artifact_path(args, config).name,
            "timetrojan_fgsm__mhealth__target1__poison5__s42.npz",
        )

    def test_poison_is_appended_to_the_client_that_owned_the_parent_example(self):
        clients = [
            WindowSet({"x": np.zeros((2, 1, 3), dtype=np.float32)}, np.array([0, 2]),
                      np.array([10, 11]), np.array([1, 1])),
            WindowSet({"x": np.zeros((2, 1, 3), dtype=np.float32)}, np.array([3, 4]),
                      np.array([20, 21]), np.array([1, 1])),
        ]
        poisoned = WindowSet({"x": np.ones((1, 1, 3), dtype=np.float32)}, np.array([1]),
                             np.array([-1]), np.array([1]))
        artifact = Artifact({}, poisoned, poisoned, poisoned, "hash", np.array([20]))
        augmented = append_poison(clients, artifact)
        self.assertEqual([len(client) for client in augmented], [2, 3])
        self.assertEqual(augmented[1].labels.tolist(), [3, 4, 1])

    def test_concatenate_preserves_synthetic_poison_id(self):
        clean = WindowSet({"x": np.zeros((1, 1, 1), dtype=np.float32)}, np.array([0]),
                          np.array([10]), np.array([1]))
        poison = WindowSet({"x": np.ones((1, 1, 1), dtype=np.float32)}, np.array([1]),
                           np.array([-42]), np.array([1]))
        self.assertEqual(concatenate([clean, poison]).example_ids.tolist(), [10, -42])

    def test_opportunity_split_is_cross_subject_and_keeps_eight_clients(self):
        self.assertEqual(OPPORTUNITY_SPLIT["holdout"], "cross-subject")
        self.assertTrue(
            set(OPPORTUNITY_SPLIT["train_subjects"]).isdisjoint(OPPORTUNITY_SPLIT["test_subjects"])
        )
        self.assertEqual(sum(OPPORTUNITY_SPLIT["hfl_chunks_per_subject"]), 8)


@unittest.skipIf(torch is None, "torch is required for TimeTrojan and aggregation tests")
class TorchTests(unittest.TestCase):
    @staticmethod
    def _updates(values):
        return [{"w": torch.tensor([value], dtype=torch.float32)} for value in values]

    def test_timetrojan_fgsm_changes_at_most_k_time_positions_with_eta_bound(self):
        class ToyModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.classifier = types.SimpleNamespace(out_features=3)

            def forward(self, values):
                return torch.stack([
                    values[:, 0, :].sum(dim=1),
                    values[:, 1, :].sum(dim=1),
                    -values.sum(dim=(1, 2)),
                ], dim=1)

        values = np.zeros((2, 2, 12), dtype=np.float32)
        poisoned = _fgsm_trigger(ToyModel(), values, target_class=1, k=5, eta=2.0,
                                 device=torch.device("cpu"))
        changed_times = np.any(np.abs(poisoned - values) > 0, axis=1)
        self.assertTrue(np.all(changed_times.sum(axis=1) <= 5))
        self.assertLessEqual(float(np.max(np.abs(poisoned - values))), 2.0)

    def test_fltrust_uses_positive_cosine_trust(self):
        merged, metadata = aggregate(
            self._updates([1.0, -10.0]), [1, 1], "fltrust",
            root_update={"w": torch.tensor([1.0])},
        )
        self.assertAlmostEqual(float(merged["w"]), 1.0)
        self.assertEqual(metadata["trust_scores"][1], 0.0)

    def test_foolsgold_downweights_repeated_similar_updates(self):
        state = FoolsGoldState(3)
        weights = state.weights(self._updates([1.0, 1.0, -1.0]))
        self.assertLess(weights[0], weights[2])
        self.assertLess(weights[1], weights[2])


class AnalysisTests(unittest.TestCase):
    @staticmethod
    def _result(topology, poison_rate, asr, f1, seed=42):
        aggregator = "fedavg" if topology == "hfl" else "coordinator"
        config = {
            "config_id": f"{topology}-{poison_rate}-{seed}",
            "arm": "primary",
            "dataset": "mhealth",
            "topology": topology,
            "fusion": "intermediate",
            "epsilon": None,
            "poison_rate": poison_rate,
            "aggregator": aggregator,
            "seed": seed,
            "participant_count": 8,
            "target_class": 1,
        }
        return {
            "config": config,
            "final": {"asr_non_target": asr, "clean_f1_macro": f1},
            "attack": {"artifact_hash": "same"},
        }

    def test_delta_positive_means_vfl_more_resistant(self):
        rows = [
            self._result("hfl", 0.0, 0.10, 0.80),
            self._result("hfl", 0.05, 0.70, 0.77),
            self._result("vfl", 0.0, 0.10, 0.82),
            self._result("vfl", 0.05, 0.30, 0.81),
        ]
        effects = topology_effects(uplift_rows(rows))
        self.assertEqual(len(effects), 1)
        self.assertAlmostEqual(effects[0]["delta_topologia"], 0.40)
        self.assertEqual(effects[0]["interpretation"], "VFL_more_resistant")

    def test_empty_partial_results_do_not_pass_the_gate(self):
        gate = gate_report([], [], [])
        self.assertFalse(gate["passed_all_observed"])

    def test_dataset_summary_reports_paired_test_and_seed_warning(self):
        effects = [
            {
                "dataset": "mhealth",
                "fusion": "intermediate",
                "epsilon": None,
                "seed": seed,
                "delta_topologia": 0.1 + seed_index * 0.01,
            }
            for seed_index, seed in enumerate((42, 123, 456, 789, 2026))
        ]
        summary = dataset_summary(effects, bootstrap_samples=100)
        self.assertEqual(summary[0]["seed_count"], 5)
        self.assertEqual(summary[0]["paired_test"]["method"], "paired-wilcoxon-exact")
        self.assertIn("indicative", summary[0]["inference_warning"])
        self.assertEqual(len(condition_effects(effects, bootstrap_samples=100)), 1)

    def test_secondary_summary_is_descriptive_hfl_only(self):
        rows = [
            {
                "arm": "secondary",
                "dataset": "mhealth",
                "fusion": "late",
                "epsilon": None,
                "aggregator": "fltrust",
                "seed": seed,
                "asr_uplift": 0.2,
                "clean_f1_delta": -0.03,
            }
            for seed in (42, 123, 456)
        ]
        secondary = secondary_aggregator_effects(rows)
        summary = secondary_aggregator_summary(secondary)
        self.assertEqual(summary[0]["seed_count"], 3)
        self.assertIn("HFL-only", summary[0]["inference_warning"])


if __name__ == "__main__":
    unittest.main()
