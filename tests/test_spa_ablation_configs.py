import json
import unittest
from pathlib import Path


EXPECTED = {
    "S0": ("snake_single", False, False, False, "none"),
    "S1": ("snake_bidir", False, False, False, "none"),
    "S2": ("cross4", False, False, False, "none"),
    "S3": ("ring_bidir", False, False, False, "none"),
    "S4": ("ring_bidir", True, False, False, "none"),
    "S5": ("ring_bidir", False, True, False, "none"),
    "S6": ("ring_bidir", False, True, True, "none"),
    "S7": ("ring_bidir", False, True, True, "compressed"),
}


class SpaAblationConfigTests(unittest.TestCase):
    def test_complete_exact_stage_matrix(self):
        root = Path(__file__).resolve().parents[1]
        for stage, expected in EXPECTED.items():
            with self.subTest(stage=stage):
                path = root / "config" / "spa_ablation" / (
                    f"basic_spa_spe_mamba_{stage.lower()}_ip.json"
                )
                with path.open(encoding="utf-8") as config_file:
                    config = json.load(config_file)
                model = config["model"]
                actual = (
                    model["spa_scan"],
                    model["spa_center_readout"],
                    model["spa_boundary_injection"],
                    model["spa_context_params"],
                    model["spa_state_fusion"],
                )
                self.assertEqual(actual, expected)
                self.assertIs(model["spa_shared_ssm"], True)
                self.assertEqual(model["spa_state_compress_dim"], 32)
                self.assertEqual(model["patch_size"], 7)
                self.assertEqual(model["in_chans"], 200)
                self.assertEqual(model["num_classes"], 16)
                schedule = config["schedule"]
                self.assertEqual(schedule["optimizer"]["name"], "AdamW")
                self.assertEqual(schedule["optimizer"]["params"]["lr"], 0.001)
                self.assertEqual(schedule["lr_decay"]["name"], "StepLR")
                self.assertEqual(schedule["lr_decay"]["params"], {"step_size": 30, "gamma": 0.9})


if __name__ == "__main__":
    unittest.main()
