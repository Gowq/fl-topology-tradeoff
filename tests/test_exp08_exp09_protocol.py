import importlib.util
import os
import subprocess
import sys
import tempfile
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

    def test_grid_launchers_batch_configs_and_isolate_singleton_retries(self):
        cases = (
            ("run_exp08_mhealth_fixed.sh", "#SBATCH --array=0-5%4", "EXP08_BATCH_SIZE=4", "exp08_r_"),
            ("run_exp09_opportunity_attack_defense.sh", "#SBATCH --array=0-20%4", "EXP09_BATCH_SIZE=10", "exp09_r_"),
        )
        for filename, array, batch_size, retry_prefix in cases:
            source = (ROOT / "scripts/grid" / filename).read_text(encoding="utf-8")
            self.assertIn(array, source)
            self.assertIn(batch_size, source)
            self.assertIn("#SBATCH --requeue", source)
            self.assertIn("retries >= 12", source)
            self.assertIn("srun --unbuffered bash scripts/run.sh", source)
            self.assertIn("--begin=now+5minutes", source)
            self.assertIn(f'--job-name="{retry_prefix}$retry_key"', source)
            self.assertIn("config_indices+=", source)

    def test_failed_cuda_guard_retries_the_exact_batch_with_a_unique_name(self):
        cases = (
            (
                "run_exp08_mhealth_fixed.sh", "2", "exp08_r_8_9_10_11",
                "EXP08_CONFIG_BATCH=8:9:10:11",
            ),
            (
                "run_exp09_opportunity_attack_defense.sh", "20",
                "exp09_r_200_201_202_203_204_205_206_207_208_209",
                "EXP09_CONFIG_BATCH=200:201:202:203:204:205:206:207:208:209",
            ),
        )
        for filename, task_id, job_name, exported_batch in cases:
            with self.subTest(script=filename), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                fake_bin = root / "bin"
                conda_base = root / "conda"
                profile = conda_base / "etc/profile.d"
                fake_bin.mkdir()
                profile.mkdir(parents=True)
                (profile / "conda.sh").write_text("conda() { :; }\n", encoding="utf-8")
                (fake_bin / "conda").write_text(
                    f'#!/bin/bash\nprintf "%s\\n" "{conda_base}"\n', encoding="utf-8"
                )
                (fake_bin / "srun").write_text("#!/bin/bash\nexit 1\n", encoding="utf-8")
                capture = root / "sbatch.txt"
                (fake_bin / "sbatch").write_text(
                    '#!/bin/bash\nprintf "%s\\n" "$@" > "$SBATCH_CAPTURE"\n',
                    encoding="utf-8",
                )
                for executable in fake_bin.iterdir():
                    executable.chmod(0o755)
                environment = os.environ.copy()
                environment.update({
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "SBATCH_CAPTURE": str(capture),
                    "SLURM_SUBMIT_DIR": str(root),
                    "SLURM_ARRAY_TASK_ID": task_id,
                    "SLURM_JOB_ID": "123",
                })
                subprocess.run(
                    ["bash", str(ROOT / "scripts/grid" / filename)],
                    env=environment, check=True, capture_output=True, text=True,
                )
                arguments = capture.read_text(encoding="utf-8").splitlines()
                self.assertIn(f"--job-name={job_name}", arguments)
                self.assertIn("--dependency=singleton", arguments)
                export = next(arg for arg in arguments if arg.startswith("--export="))
                export_parts = export.removeprefix("--export=").split(",")
                self.assertIn(exported_batch, export_parts)
                self.assertTrue(
                    all(part == "ALL" or "=" in part for part in export_parts),
                    f"Slurm would parse stray environment names from {export!r}",
                )

                batch_variable, encoded_batch = exported_batch.split("=", 1)
                retry_environment = environment.copy()
                retry_environment.update({
                    batch_variable: encoded_batch,
                    "SLURM_ARRAY_TASK_ID": "0",
                })
                subprocess.run(
                    ["bash", str(ROOT / "scripts/grid" / filename)],
                    env=retry_environment, check=True, capture_output=True, text=True,
                )
                retry_arguments = capture.read_text(encoding="utf-8").splitlines()
                self.assertIn(f"--job-name={job_name}", retry_arguments)
                retry_export = next(
                    arg for arg in retry_arguments if arg.startswith("--export=")
                )
                self.assertIn(
                    exported_batch,
                    retry_export.removeprefix("--export=").split(","),
                )


if __name__ == "__main__":
    unittest.main()
