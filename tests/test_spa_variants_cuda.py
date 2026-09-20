import unittest

import torch

from models.basic_spa_spe_mamba import BasicSpaSpeMamba, EditableMambaCore


STAGES = {
    "S0": {},
    "S1": {"spa_scan": "snake_bidir"},
    "S2": {"spa_scan": "cross4"},
    "S3": {"spa_scan": "ring_bidir"},
    "S4": {"spa_scan": "ring_bidir", "spa_center_readout": True},
    "S5": {"spa_scan": "ring_bidir", "spa_boundary_injection": True},
    "S6": {
        "spa_scan": "ring_bidir",
        "spa_boundary_injection": True,
        "spa_context_params": True,
    },
    "S7": {
        "spa_scan": "ring_bidir",
        "spa_boundary_injection": True,
        "spa_context_params": True,
        "spa_state_fusion": "compressed",
    },
}


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class SpaVariantCudaTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            probe = EditableMambaCore(
                4, d_state=2, ssm_ratio=1, d_conv=0
            ).cuda()
            probe(torch.randn(1, 1, 3, 4, device="cuda"))
        except (ImportError, OSError) as error:
            raise unittest.SkipTest(f"CUDA selective scan unavailable: {error}")
        except RuntimeError as error:
            message = str(error).lower()
            missing_backend = (
                "selective_scan" in message
                or "no kernel image" in message
                or "not compiled" in message
            )
            if missing_backend:
                raise unittest.SkipTest(f"CUDA selective scan unavailable: {error}")
            raise

    def test_default_s0_parameter_count(self):
        model = BasicSpaSpeMamba()
        self.assertEqual(sum(p.numel() for p in model.parameters()), 419476)

    def test_s0_s7_forward_backward_and_enabled_modules(self):
        expected_paths = {
            "S0": 1, "S1": 2, "S2": 4, "S3": 2,
            "S4": 2, "S5": 2, "S6": 2, "S7": 2,
        }
        for stage, flags in STAGES.items():
            with self.subTest(stage=stage):
                model = BasicSpaSpeMamba(**flags).cuda()
                spatial = model.blocks[0].spatial
                self.assertEqual(spatial.num_paths, expected_paths[stage])
                self.assertEqual(hasattr(model, "center_pool_beta"), stage == "S4")
                self.assertEqual(spatial.boundary_reliability is not None, stage in {"S5", "S6", "S7"})
                self.assertEqual(spatial.context_conv is not None, stage in {"S6", "S7"})
                self.assertEqual(spatial.state_fusion is not None, stage == "S7")

                inputs = torch.randn(1, 1, 200, 7, 7, device="cuda")
                output = model(inputs)
                self.assertEqual(tuple(output.shape), (1, 16))
                self.assertTrue(torch.isfinite(output).all())
                output.sum().backward()

                if spatial.direction_logits is not None:
                    self.assertIsNotNone(spatial.direction_logits.grad)
                if stage == "S4":
                    self.assertIsNotNone(model.center_pool_beta.grad)
                if spatial.boundary_reliability is not None:
                    self.assertIsNotNone(spatial.boundary_reliability.mlp[0].weight.grad)
                if spatial.context_conv is not None:
                    self.assertIsNotNone(spatial.context_conv.weight.grad)
                if spatial.state_fusion is not None:
                    self.assertIsNotNone(spatial.state_fusion.down.weight.grad)


if __name__ == "__main__":
    unittest.main()
