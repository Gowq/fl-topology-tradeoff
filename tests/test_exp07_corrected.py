import sys
import types
import unittest
import importlib.util
from unittest import mock
from pathlib import Path

import numpy as np


CODE = Path(__file__).parents[1] / "experiments/exp07_corrected_attacks/code"


def load_exp07_module(filename):
    name = f"exp07_{filename}_for_test"
    spec = importlib.util.spec_from_file_location(name, CODE / f"{filename}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


exp07_protocol = load_exp07_module("protocol")
exp07_data = load_exp07_module("data")
exp07_attacks = load_exp07_module("attacks")
with mock.patch.dict(sys.modules, {"protocol": exp07_protocol}):
    analyze_exp07 = load_exp07_module("analyze_exp07")

degradation_rows = analyze_exp07.degradation_rows
topology_effects = analyze_exp07.topology_effects
attack_message = exp07_attacks.attack_message
attack_update = exp07_attacks.attack_update
label_poison_ids = exp07_attacks.label_poison_ids
malicious_indices = exp07_attacks.malicious_indices
MHEALTH_GROUPS = exp07_data.MHEALTH_GROUPS
OPPORTUNITY_GROUPS = exp07_data.OPPORTUNITY_GROUPS
WindowSet = exp07_data.WindowSet
concatenate = exp07_data.concatenate
build_protocol = exp07_protocol.build_protocol
smoke_protocol = exp07_protocol.smoke_protocol

try:
    import torch

    aggregate = load_exp07_module("aggregation").aggregate
except ImportError:  # torch is absent on the review workstation
    torch = None


def load_exp07_models():
    return load_exp07_module("models")


class ProtocolTests(unittest.TestCase):
    def test_matrix_is_unique_and_has_preregistered_arm_sizes(self):
        configs = build_protocol()
        self.assertEqual(len(configs), 2052)
        self.assertEqual(len({config.config_id for config in configs}), 2052)
        self.assertEqual(sum(config.arm == "primary" for config in configs), 1080)
        self.assertEqual(sum(config.arm == "secondary" for config in configs), 972)
        self.assertTrue(all(config.config_id.startswith("exp07v3__") for config in configs))

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

    def test_grid_blocks_cover_each_config_once_and_keep_arms_separate(self):
        primary = exp07_protocol.execution_blocks("primary", block_size=6)
        secondary = exp07_protocol.execution_blocks("secondary", block_size=6)
        flattened = [index for block in primary + secondary for index in block]
        self.assertEqual(flattened, list(range(2052)))
        self.assertTrue(all(len(block) == 6 for block in primary + secondary))
        self.assertEqual(len(primary), 180)
        self.assertEqual(len(secondary), 162)


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

    def test_both_datasets_use_cross_subject_holdout(self):
        self.assertEqual(exp07_data.MHEALTH_SPLIT["holdout"], "cross-subject")
        self.assertEqual(exp07_data.OPPORTUNITY_SPLIT["holdout"], "cross-subject")
        self.assertTrue(
            set(exp07_data.OPPORTUNITY_SPLIT["train_subjects"]).isdisjoint(
                exp07_data.OPPORTUNITY_SPLIT["test_subjects"]
            )
        )

    def test_opportunity_training_excludes_the_held_out_subject(self):
        def subject(_, subject_id, __):
            count = 6
            return WindowSet(
                {name: np.zeros((count, selection.stop - selection.start, 2), dtype=np.float32)
                 for name, (_, selection) in OPPORTUNITY_GROUPS.items()},
                np.zeros(count, dtype=np.int64),
                np.arange(subject_id * 100, subject_id * 100 + count),
                np.full(count, subject_id),
            )

        with mock.patch.object(exp07_data, "_opportunity_subject", side_effect=subject), \
                mock.patch.object(
                    exp07_data, "_normalize", side_effect=lambda train, test: (train, test)
                ):
            clients, test, _ = exp07_data._opportunity(Path("unused"))
        self.assertEqual(len(clients), 8)
        self.assertNotIn(2, {int(owner) for client in clients for owner in client.owners})
        self.assertEqual(set(test.owners.tolist()), {2})


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


@unittest.skipIf(torch is None, "torch is required for model tests")
class PersistentVFLAttackTests(unittest.TestCase):
    def test_sign_flip_persists_in_eval_forward(self):
        EightPartyFusion = load_exp07_models().EightPartyFusion

        model = EightPartyFusion({f"g{i}": 1 for i in range(8)}, "late", 3, dropout=0.0)
        values = {name: torch.randn(2, 1, 16) for name in model.branches}
        model.eval()
        clean = model(values).detach().clone()
        model.set_compromised((name for name in ("g0",)), "sign_flip")
        with mock.patch.dict(sys.modules, {"attacks": exp07_attacks}):
            attacked = model(values).detach().clone()
        self.assertFalse(torch.equal(clean, attacked))
        expected = model.messages(values, apply_attack=False)
        expected["g0"] = -expected["g0"]
        torch.testing.assert_close(attacked, model.fuse(expected))

    def test_free_rider_dp_count_matches_private_modules(self):
        models = load_exp07_models()

        fake_opacus = types.ModuleType("opacus")
        fake_grad_sample = types.ModuleType("opacus.grad_sample")
        fake_grad_sample.GradSampleModule = lambda module: module
        model = models.EightPartyFusion({f"g{i}": 1 for i in range(8)}, "intermediate", 3)
        with mock.patch.dict(sys.modules, {
            "opacus": fake_opacus,
            "opacus.grad_sample": fake_grad_sample,
        }):
            private_modules = models.wrap_private_modules(
                model, tuple(range(6)), "free_rider"
            )
        self.assertEqual(len(private_modules), 3)
        self.assertEqual(
            len(private_modules),
            sum(any(parameter.requires_grad for parameter in branch.parameters())
                for branch in model.branches.values()) + 1,
        )


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

    @staticmethod
    def _secondary_result(attack, f1, ratio=0.0, theoretical=True, seed=42):
        config = {
            "config_id": f"secondary-{attack}-{ratio}-{seed}", "arm": "secondary",
            "dataset": "mhealth", "topology": "hfl", "fusion": "intermediate",
            "epsilon": None, "attack": attack, "attack_ratio": ratio,
            "aggregator": "median", "seed": seed, "participant_count": 8,
        }
        return {
            "config": config,
            "round_metrics": [{"aggregation": {
                "theoretical_condition_met": theoretical,
            }}],
            "final": {"f1_macro": f1, "aggregation": {
                "theoretical_condition_met": theoretical,
            }},
        }

    def test_secondary_report_includes_degradation_and_theoretical_validity(self):
        rows = [
            self._secondary_result("none", 0.8),
            self._secondary_result("sign_flip", 0.5, 0.25, True),
            self._secondary_result("sign_flip", 0.2, 0.75, False),
        ]
        report = analyze_exp07.secondary_aggregator_effects(degradation_rows(rows))
        self.assertEqual(len(report), 2)
        self.assertAlmostEqual(report[0]["degradation"], -0.3)
        self.assertTrue(report[0]["theoretical_condition_met"])
        self.assertFalse(report[1]["theoretical_condition_met"])
        self.assertIn("three seeds", report[0]["inference_warning"])

    def test_condition_decomposition_does_not_hide_reversals(self):
        effects = [
            {"dataset": "mhealth", "fusion": "late", "epsilon": None,
             "attack": "sign_flip", "attack_ratio": 0.25, "seed": seed,
             "topology_effect": value}
            for seed, value in zip((42, 123, 456, 789, 2026), (0.1, 0.2, 0.3, 0.2, 0.1))
        ]
        cells = analyze_exp07.condition_effects(effects, bootstrap_samples=100)
        self.assertEqual(len(cells), 1)
        self.assertEqual(cells[0]["seed_count"], 5)
        self.assertEqual(cells[0]["attack"], "sign_flip")
        self.assertIn("indicative", cells[0]["inference_warning"])


if __name__ == "__main__":
    unittest.main()
