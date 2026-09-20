import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts import run_spa_ablation as runner


class SpaAblationRunnerTests(unittest.TestCase):
    def test_default_matrix_has_24_exact_jobs(self):
        opts = runner.parse_args([])
        jobs = runner.build_jobs(opts)
        self.assertEqual(len(jobs), 24)
        self.assertEqual({job.stage for job in jobs}, {f"S{i}" for i in range(8)})
        self.assertEqual({job.seed for job in jobs}, {1, 34, 20031025})
        self.assertTrue(all("--ratio" in job.command for job in jobs))
        self.assertTrue(all("0.1" in job.command for job in jobs))
        self.assertTrue(all("--skip-map" in job.command for job in jobs))

    def test_completion_requires_matching_valid_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            job = runner.Job(
                stage="S3",
                seed=34,
                config_path=Path("config.json"),
                output_dir=root,
                metrics_path=root / "metrics.json",
                command=("python",),
            )
            self.assertFalse(runner.metrics_complete(job, 0.1))
            job.metrics_path.write_text("not json")
            self.assertFalse(runner.metrics_complete(job, 0.1))
            base = {
                "run_name": "S3_seed34",
                "seed": 34,
                "ratio": 0.1,
                "oa": 90.0,
                "aa": 88.0,
                "kappa": 87.0,
                "class_accuracy": [80.0, 96.0],
            }
            for key, value in (
                ("run_name", "S2_seed34"),
                ("seed", 1),
                ("ratio", 0.05),
            ):
                payload = dict(base)
                payload[key] = value
                job.metrics_path.write_text(json.dumps(payload))
                self.assertFalse(runner.metrics_complete(job, 0.1))
            incomplete = dict(base)
            incomplete.pop("aa")
            job.metrics_path.write_text(json.dumps(incomplete))
            self.assertFalse(runner.metrics_complete(job, 0.1))
            job.metrics_path.write_text(json.dumps(base))
            self.assertTrue(runner.metrics_complete(job, 0.1))

    def test_aggregation_uses_population_standard_deviation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jobs = []
            for seed, oa in ((1, 90.0), (34, 94.0)):
                output = root / f"seed_{seed}"
                output.mkdir()
                metrics_path = output / "metrics.json"
                metrics_path.write_text(json.dumps({
                    "run_name": f"S0_seed{seed}", "seed": seed, "ratio": 0.1,
                    "oa": oa, "aa": oa - 1, "kappa": oa - 2,
                    "parameter_count": 419476,
                    "class_accuracy": [oa - 3, oa - 4],
                }))
                jobs.append(runner.Job("S0", seed, Path("c"), output, metrics_path, ("python",)))
            summary = runner.aggregate_results(jobs, root)
            self.assertEqual(summary["S0"]["oa"]["mean"], 92.0)
            self.assertEqual(summary["S0"]["oa"]["std"], 2.0)
            self.assertTrue((root / "runs.csv").is_file())
            self.assertTrue((root / "summary.csv").is_file())
            self.assertTrue((root / "summary.json").is_file())

    def test_dry_run_does_not_launch_or_create_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "not-created"
            with mock.patch.object(runner.subprocess, "Popen") as popen:
                result = runner.main(["--dry-run", "--root", str(root)])
            self.assertEqual(result, 0)
            popen.assert_not_called()
            self.assertFalse(root.exists())


if __name__ == "__main__":
    unittest.main()
