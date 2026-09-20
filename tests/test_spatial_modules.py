import unittest

import torch

from models.basic_spa_spe_mamba import (
    BasicSpaSpeMamba,
    BoundaryReliability,
    CompressedSpatialStateFusion,
    EditableMambaCore,
    SpatialMultiPathMamba,
    center_mean_pool,
)


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


class GuidanceTests(unittest.TestCase):
    def test_reliability_range_center_and_readout(self):
        reliability = BoundaryReliability(8)(torch.randn(3, 7, 7, 8))
        self.assertTrue(torch.all(reliability >= 0.1))
        self.assertTrue(torch.all(reliability <= 1.0))
        self.assertTrue(torch.equal(reliability[:, 3, 3], torch.ones(3)))
        inputs = torch.randn(2, 7, 7, 8)
        actual = center_mean_pool(inputs, torch.nn.Parameter(torch.zeros(())))
        expected = 0.5 * inputs[:, 3, 3] + 0.5 * inputs.mean((1, 2))
        self.assertTrue(torch.equal(actual, expected))

    def test_context_shape(self):
        module = SpatialMultiPathMamba(
            7,
            8,
            spa_scan="ring_bidir",
            spa_context_params=True,
            d_state=4,
            ssm_ratio=1,
            d_conv=3,
        )
        context = module._parameter_context(torch.randn(2, 7, 7, 8))
        self.assertEqual(tuple(context.shape), (2, 2, 49, module.mamba.d_inner))


class StateFusionTests(unittest.TestCase):
    def test_shape_and_gradients(self):
        module = CompressedSpatialStateFusion(8, 4, 5)
        states = torch.randn(2, 2, 8, 4, 49, requires_grad=True)
        fused = module(states, torch.tensor([0.25, 0.75]))
        self.assertEqual(tuple(fused.shape), (2, 8, 4, 49))
        fused.sum().backward()
        self.assertIsNotNone(module.down.weight.grad)
        self.assertIsNotNone(module.up.weight.grad)

    def test_invalid_mode(self):
        with self.assertRaisesRegex(ValueError, "ring_bidir"):
            BasicSpaSpeMamba(spa_scan="cross4", spa_state_fusion="compressed")
        with self.assertRaisesRegex(ValueError, "positive"):
            BasicSpaSpeMamba(
                spa_scan="ring_bidir",
                spa_state_fusion="compressed",
                spa_state_compress_dim=0,
            )
        with self.assertRaisesRegex(ValueError, "spa_shared_ssm"):
            BasicSpaSpeMamba(spa_shared_ssm=False)


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

    def test_explicit_state_observation_matches_shared_scan(self):
        torch.manual_seed(11)
        core = EditableMambaCore(
            8, num_directions=1, d_state=4, ssm_ratio=1, d_conv=3
        ).cuda().eval()
        inputs = torch.randn(2, 1, 9, 8, device="cuda")
        internals = core.scan_states_shared(inputs)
        observed = core.observe_shared(
            internals.states[:, 0],
            internals.state_c,
            internals.u,
            internals.gate,
        )
        self.assertTrue(
            torch.allclose(
                core.forward_shared(inputs), observed, atol=2e-4, rtol=2e-4
            )
        )

    def test_compressed_state_fusion_forward_backward(self):
        module = SpatialMultiPathMamba(
            7,
            8,
            spa_scan="ring_bidir",
            spa_boundary_injection=True,
            spa_context_params=True,
            spa_state_fusion="compressed",
            spa_state_compress_dim=5,
            d_state=4,
            ssm_ratio=1,
            d_conv=3,
        ).cuda()
        inputs = torch.randn(2, 7, 7, 8, device="cuda", requires_grad=True)
        spectral_tokens = torch.randn(2, 7, 7, 8, device="cuda")
        output = module(inputs, spectral_tokens)
        self.assertEqual(tuple(output.shape), (2, 7, 7, 8))
        self.assertTrue(torch.isfinite(output).all())
        output.sum().backward()
        self.assertIsNotNone(module.state_fusion.down.weight.grad)


if __name__ == "__main__":
    unittest.main()
