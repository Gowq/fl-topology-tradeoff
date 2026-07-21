import sys
import unittest
from pathlib import Path

import numpy as np


CODE = Path(__file__).parents[1] / "experiments/exp07_corrected_attacks/code"
sys.path.insert(0, str(CODE))

from analyze_exp07 import degradation_rows, topology_effects
from attacks import attack_message, attack_update, label_poison_ids, malicious_indices
from data import MHEALTH_GROUPS, OPPORTUNITY_GROUPS, WindowSet, concatenate
from protocol import build_protocol, smoke_protocol

try:
    import torch

    from aggregation import aggregate
except ImportError:  # torch is absent on the review workstation
    torch = None


class ProtocolTests(unittest.TestCase):
    def test_matrix_is_unique_and_has_preregistered_arm_sizes(self):
        configs = build_protocol()
        self.assertEqual(len(configs), 2052)
        self.assertEqual(len({config.config_id for config in configs}), 2052)
        self.assertEqual(sum(config.arm == "primary" for config in configs), 1080)
        self.assertEqual(sum(config.arm == "secondary" for config in configs), 972)

    def test_no_dp_is_not_serialized_as_epsilon_zero(self):
        configs = build_protocol()
        self.assertIn(None, {config.epsilon for config in configs})
        self.assertNotIn(0.0, {config.epsilon for config in configs})
        self.assertTrue(all(config.attack == "none" or config.attack_ratio in {0.25, 0.75}
                            for config in configs))

    def test_smoke_covers_datasets_topologies_fusions_dp_and_robust_aggregation(self):
        smoke = smoke_protocol()
        self.assertEqual({config.dataset for config in smoke}, {"mhealth", "opportunity"})
        self.assertEqual({config.topology for config in smoke}, {"hfl", "vfl"})
        self.assertEqual({config.fusion for config in smoke}, {"intermediate", "late"})
        self.assertIn(None, {config.epsilon for config in smoke})
        self.assertIn(20.0, {config.epsilon for config in smoke})
        self.assertIn("krum", {config.aggregator for config in smoke})


class EquivalenceTests(unittest.TestCase):
    def test_both_datasets_define_exactly_eight_vfl_groups(self):
        self.assertEqual(len(MHEALTH_GROUPS), 8)
        self.assertEqual(len(OPPORTUNITY_GROUPS), 8)

    def test_malicious_assignments_have_exact_integer_pressure(self):
        for seed in (42, 123, 456, 789, 2026):
            self.assertEqual(len(malicious_indices(seed, 0.25)), 2)
            self.assertEqual(len(malicious_indices(seed, 0.75)), 6)

    def test_every_preregistered_seed_gets_a_distinct_malicious_set(self):
        assignments = {malicious_indices(seed, 0.25) for seed in (42, 123, 456, 789, 2026)}
        self.assertEqual(len(assignments), 5)

    def test_label_victims_are_the_records_owned_by_malicious_participants(self):
        clients = [
            WindowSet({"x": np.zeros((2, 1, 1))}, np.zeros(2), np.array([10 * i, 10 * i + 1]),
                      np.full(2, i))
            for i in range(8)
        ]
        selected = malicious_indices(42, 0.25)
        victims = label_poison_ids(clients, selected)
        self.assertEqual(len(victims), 4)
        self.assertEqual(
            victims,
            frozenset(int(value) for index in selected for value in clients[index].example_ids),
        )

    def test_parameter_attacks_target_updates_and_messages(self):
        update = {"weight": np.asarray([1.0, -2.0])}
        np.testing.assert_array_equal(attack_update(update, "sign_flip")["weight"],
                                      np.asarray([-1.0, 2.0]))
        np.testing.assert_array_equal(attack_update(update, "free_rider")["weight"],
                                      np.zeros(2))
        message = np.asarray([[1.0, -2.0]])
        np.testing.assert_array_equal(attack_message(message, "sign_flip"), -message)
        np.testing.assert_array_equal(attack_message(message, "free_rider"), message)

    def test_concatenation_preserves_global_example_ids(self):
        first = WindowSet({"x": np.zeros((2, 1, 1))}, np.zeros(2), np.array([10, 11]),
                          np.ones(2))
        second = WindowSet({"x": np.zeros((1, 1, 1))}, np.zeros(1), np.array([20]),
                           np.full(1, 2))
        self.assertEqual(concatenate([first, second]).example_ids.tolist(), [10, 11, 20])


@unittest.skipIf(torch is None, "torch is required for aggregator tests")
class AggregationTests(unittest.TestCase):
    @staticmethod
    def _updates(values):
        return [{"w": torch.tensor([value], dtype=torch.float32)} for value in values]

    def test_fedavg_weights_by_sample_count(self):
        merged, metadata = aggregate(self._updates([0.0, 1.0]), [3, 1], "fedavg", 0)
        self.assertAlmostEqual(float(merged["w"]), 0.25)
        self.assertTrue(metadata["theoretical_condition_met"])

    def test_coordinatewise_methods_flag_the_75_percent_regime_as_invalid(self):
        updates = self._updates([0.0] * 2 + [100.0] * 6)
        counts = [1] * 8
        _, median = aggregate(updates, counts, "median", 6)
        _, trimmed = aggregate(updates, counts, "trimmed_mean", 6)
        _, krum = aggregate(updates, counts, "krum", 6)
        self.assertFalse(median["theoretical_condition_met"])
        self.assertFalse(trimmed["theoretical_condition_met"])
        self.assertFalse(krum["theoretical_condition_met"])
        self.assertEqual(krum["neighbors"], 1)

    def test_robust_methods_reject_a_25_percent_minority(self):
        updates = self._updates([1.0] * 6 + [500.0] * 2)
        counts = [1] * 8
        merged, metadata = aggregate(updates, counts, "median", 2)
        self.assertTrue(metadata["theoretical_condition_met"])
        self.assertAlmostEqual(float(merged["w"]), 1.0)
        selected, krum = aggregate(updates, counts, "krum", 2)
        self.assertTrue(krum["theoretical_condition_met"])
        self.assertAlmostEqual(float(selected["w"]), 1.0)

    def test_unsupported_aggregator_is_rejected(self):
        with self.assertRaises(ValueError):
            aggregate(self._updates([1.0]), [1], "foolsgold", 0)


class AnalysisTests(unittest.TestCase):
    @staticmethod
    def _result(topology, attack, f1, seed=42):
        aggregator = "fedavg" if topology == "hfl" else "coordinator"
        config = {
            "config_id": f"{topology}-{attack}-{seed}", "arm": "primary",
            "dataset": "mhealth", "topology": topology, "fusion": "intermediate",
            "epsilon": None, "attack": attack,
            "attack_ratio": 0.0 if attack == "none" else 0.25,
            "aggregator": aggregator, "seed": seed, "participant_count": 8,
        }
        return {"config": config, "final": {"f1_macro": f1}}

    def test_effect_is_difference_between_clean_normalized_degradations(self):
        rows = [
            self._result("hfl", "none", 0.8), self._result("hfl", "sign_flip", 0.3),
            self._result("vfl", "none", 0.9), self._result("vfl", "sign_flip", 0.8),
        ]
        effects = topology_effects(degradation_rows(rows))
        self.assertEqual(len(effects), 1)
        self.assertAlmostEqual(effects[0]["topology_effect"], 0.4)


if __name__ == "__main__":
    unittest.main()
