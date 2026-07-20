import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parents[1]
EXP08 = ROOT / "experiments/exp08_mhealth_fixed_rounds/code"
EXP09 = ROOT / "experiments/exp09_opportunity_attack_defense/code"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


exp08_protocol = load_module("exp08_protocol_test", EXP08 / "protocol.py")
exp09_protocol = load_module("exp09_protocol_test", EXP09 / "protocol.py")
opportunity_data = load_module("exp09_opportunity_data_test", EXP09 / "opportunity_data.py")


class Exp08ProtocolTests(unittest.TestCase):
    def test_matrix_exactly_replicates_exp06_axes(self):
        configs = exp08_protocol.build_protocol()
        self.assertEqual(len(configs), 24)
        self.assertEqual({config.topology for config in configs}, {"hfl", "vfl"})
        self.assertEqual({config.fusion for config in configs}, {"intermediate", "late"})
        self.assertEqual({config.epsilon for config in configs}, {100.0, 200.0})
        self.assertEqual({config.seed for config in configs}, {42, 123, 456})
        self.assertEqual(len({config.config_id for config in configs}), 24)

    def test_smoke_covers_every_topology_fusion_pair(self):
        configs = exp08_protocol.smoke_protocol()
        self.assertEqual(
            {(config.topology, config.fusion) for config in configs},
            {("hfl", "intermediate"), ("hfl", "late"),
             ("vfl", "intermediate"), ("vfl", "late")},
        )


class Exp09ProtocolTests(unittest.TestCase):
    def test_matrix_matches_exp07_robustness_transfer(self):
        configs = exp09_protocol.build_protocol()
        self.assertEqual(len(configs), 210)
        self.assertEqual(len({config.config_id for config in configs}), 210)
        hfl = [config for config in configs if config.topology == "hfl"]
        vfl = [config for config in configs if config.topology == "vfl"]
        self.assertEqual(len(hfl), 180)
        self.assertEqual(len(vfl), 30)
        self.assertEqual({config.aggregator for config in hfl}, {"fedavg", "fltrust", "foolsgold"})
        self.assertEqual({config.attack for config in hfl}, {"model_replacement", "sensor_backdoor"})
        self.assertEqual({config.aggregator for config in vfl}, {"coordinator"})
        self.assertEqual({config.attack for config in vfl}, {"sensor_backdoor"})

    def test_trigger_is_localized_and_scale_relative(self):
        values = {
            "body_sensors": np.zeros((30, 30), dtype=np.float32),
            "object_sensors": np.zeros((15, 30), dtype=np.float32),
            "ambient_sensors": np.zeros((10, 30), dtype=np.float32),
        }
        triggered = opportunity_data.apply_sensor_trigger(values, np.ones(3, dtype=np.float32))
        np.testing.assert_array_equal(triggered["object_sensors"], values["object_sensors"])
        np.testing.assert_array_equal(triggered["ambient_sensors"], values["ambient_sensors"])
        self.assertEqual(np.count_nonzero(triggered["body_sensors"]), 3 * 6)
        np.testing.assert_array_equal(triggered["body_sensors"][27:30, -6:], 4.0)

    def test_fltrust_root_is_disjoint_from_train_and_test_runs(self):
        self.assertTrue(set(opportunity_data.ROOT_RUNS).isdisjoint(opportunity_data.TRAIN_RUNS))
        self.assertTrue(set(opportunity_data.ROOT_RUNS).isdisjoint(opportunity_data.TEST_RUNS))
        self.assertNotEqual(opportunity_data.BACKDOOR_TARGET, 0)

    def test_cli_lists_work_without_gpu_dependencies(self):
        expected = ((EXP08 / "run_exp08.py", 24), (EXP09 / "run_exp09.py", 210))
        for script, count in expected:
            with self.subTest(script=script.name):
                completed = subprocess.run(
                    [sys.executable, str(script), "--list"], check=True,
                    capture_output=True, text=True,
                )
                self.assertEqual(completed.stdout.count('"config_id"'), count)

    def test_grid_launchers_use_gpu_job_steps_and_exact_arrays(self):
        cases = (
            ("run_exp08_mhealth_fixed.sh", "#SBATCH --array=0-23"),
            ("run_exp09_opportunity_attack_defense.sh", "#SBATCH --array=0-209"),
        )
        for filename, array in cases:
            source = (ROOT / "scripts/grid" / filename).read_text(encoding="utf-8")
            self.assertIn(array, source)
            self.assertIn("srun --unbuffered bash scripts/run.sh", source)
            self.assertIn("--begin=now+5minutes", source)


if __name__ == "__main__":
    unittest.main()
