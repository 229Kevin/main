import unittest

import torch

from models.basic_spa_spe_mamba import EditableMambaCore, SpatialMultiPathMamba


class ZeroPosition(torch.nn.Module):
    def forward(self, inputs):
        return torch.zeros_like(inputs)


class MultiPathTests(unittest.TestCase):
    def test_counts_and_single_parameter_group(self):
        counts = {"snake_single": 1, "snake_bidir": 2, "cross4": 4, "ring_bidir": 2}
        for scan, count in counts.items():
            module = SpatialMultiPathMamba(
                7, 8, spa_scan=scan, d_state=4, ssm_ratio=1, d_conv=3
            )
            self.assertEqual(module.num_paths, count)
            self.assertEqual(module.mamba.num_directions, 1)
            self.assertEqual(module.mamba.x_proj_weight.shape[0], 1)

    def test_alignment_before_fusion(self):
        module = SpatialMultiPathMamba(
            7, 8, spa_scan="cross4", d_state=4, ssm_ratio=1, d_conv=3
        )
        source = torch.arange(49, dtype=torch.float32).view(1, 7, 7, 1)
        source = source.repeat(1, 1, 1, 8)
        module.dynamic_pos = ZeroPosition()
        module.norm = torch.nn.Identity()
        module.mamba.forward_shared = (
            lambda sequences, parameter_features=None, reliability=None: sequences
        )
        self.assertTrue(torch.equal(module(source), source))


@unittest.skipUnless(torch.cuda.is_available(), "CUDA selective scan unavailable")
class SharedCudaScanTests(unittest.TestCase):
    def test_single_path_matches_existing_forward(self):
        torch.manual_seed(7)
        core = EditableMambaCore(
            8, num_directions=1, d_state=4, ssm_ratio=1, d_conv=3
        ).cuda().eval()
        inputs = torch.randn(2, 1, 9, 8, device="cuda")
        self.assertTrue(
            torch.allclose(
                core(inputs), core.forward_shared(inputs), atol=1e-5, rtol=1e-5
            )
        )


if __name__ == "__main__":
    unittest.main()
