import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

import main


class MainExperimentCliTests(unittest.TestCase):
    def test_parser_and_seed_resolution(self):
        parser = main.build_parser()
        opts = parser.parse_args(["--seed", "34"])
        self.assertEqual(main.resolve_seeds(opts), [34])
        opts = parser.parse_args(["--seed", "34", "--num_run", "3"])
        with self.assertRaisesRegex(ValueError, "--seed"):
            main.resolve_seeds(opts)

    def test_explicit_config_wins(self):
        parser = main.build_parser()
        opts = parser.parse_args(["--config", "chosen.json", "--model", "ignored"])
        self.assertEqual(main.resolve_config_path(opts), Path("chosen.json"))

    def test_numpy_metrics_are_json_ready(self):
        metrics = {
            "Accuracy": np.float64(91.5),
            "AA": np.float32(88.25),
            "Kappa": np.float64(89.0),
            "class acc": np.array([80.0, 90.0]),
            "Confusion matrix": np.array([[4, 1], [0, 5]]),
        }
        self.assertEqual(
            main.json_ready_metrics(metrics),
            {
                "oa": 91.5,
                "aa": 88.25,
                "kappa": 89.0,
                "class_accuracy": [80.0, 90.0],
                "confusion_matrix": [[4, 1], [0, 5]],
            },
        )

    def test_atomic_json_write(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "nested" / "metrics.json"
            main.atomic_write_json(target, {"ok": np.int64(1)})
            self.assertEqual(json.loads(target.read_text()), {"ok": 1})
            self.assertFalse(target.with_suffix(target.suffix + ".tmp").exists())


if __name__ == "__main__":
    unittest.main()
