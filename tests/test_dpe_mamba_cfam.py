"""Numerical and configuration contracts for the tuned D20/V139 model."""
import json
import math
from pathlib import Path
import sys
import unittest

import torch
from torch import nn

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from models.DPE_Mamba_CFAM import (
    BasicSpaSpeMamba, BidirectionalFusionState, CenterPairedCFAM,
    DualMambaDPE, SpectralOSSMWithBFS, _DPEAttention,
)

torch.set_num_threads(2)


class ConstantCorrection(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full_like(x, self.value)


class TunedModelContracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(34)

    def test_center_protection_still_obeys_rms_bound_when_branches_cancel(self):
        dpe = DualMambaDPE(8)
        with torch.no_grad():
            dpe.dwconv.weight.zero_()
            dpe.dwconv.bias.zero_()
        dpe.core = ConstantCorrection(-100)
        dpe.channel = ConstantCorrection(100)
        x = torch.ones(2, 7, 7, 8)
        delta = dpe(x) - x
        self.assertLessEqual(delta.square().mean().sqrt().item(), 0.05001)
        self.assertGreater(delta[:, 3, 3].abs().sum().item(), 0)
        self.assertEqual(delta[:, 0, 0].abs().sum().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'FP16 kernels require CUDA')
    def test_half_precision_dpe_attention_and_gradients_remain_finite(self):
        device = torch.device('cuda', torch.cuda.device_count() - 1)
        attention = _DPEAttention(8).to(device).half()
        with torch.no_grad():
            attention.qkv.weight.zero_()
            attention.qkv.bias.fill_(1000)
        x = torch.ones(2, 8, 3, 3, device=device, dtype=torch.float16, requires_grad=True)
        output = attention(x)
        self.assertTrue(torch.isfinite(output).all().item())
        output.float().square().mean().backward()
        self.assertTrue(torch.isfinite(x.grad).all().item())

    @unittest.skipUnless(torch.cuda.is_available(), 'FP16 kernels require CUDA')
    def test_half_precision_raw_cross_attention_remains_finite(self):
        device = torch.device('cuda', torch.cuda.device_count() - 1)
        fusion = CenterPairedCFAM(3, 8).to(device).half()
        with torch.no_grad():
            for projection in (fusion.raw_cross_query, fusion.raw_cross_key):
                projection.weight.zero_()
                projection.weight[:, 0] = 512
        raw = torch.zeros(2, 8, 3, 3, device=device, dtype=torch.float16)
        raw[:, 0] = 1
        spa = raw.permute(0, 2, 3, 1).contiguous()
        spe = spa[:, 1, 1].unsqueeze(1)
        output = fusion.cross_attend_raw(raw, spa, spe)
        self.assertTrue(torch.isfinite(output).all().item())

    def test_single_block_honors_configured_drop_path(self):
        config = json.loads((PROJECT / 'config/basic_spa_spe_mamba_ip.json').read_text())['model']
        model = BasicSpaSpeMamba(**config)
        self.assertAlmostEqual(getattr(model.blocks[0].drop_path, 'drop_prob', 0), 0.1)

    def test_bare_defaults_match_exported_mamba_preset(self):
        block = BasicSpaSpeMamba().blocks[0]
        self.assertTrue(block.spatial.mamba.disable_z)
        self.assertEqual(block.spatial.mamba.d_conv, 0)
        self.assertFalse(block.spectral.mamba.bfs_enabled)

    def test_tuning_options_reach_both_modules(self):
        model = BasicSpaSpeMamba(
            dpe_options={'num_heads': 8, 'radius': 2.0, 'gain_init': 0.02},
            cfam_options={'raw_width_ratio': 2.0, 'raw_encoder_drop_rate': 0.1,
                          'cross_heads': 8, 'pool_heads': 4, 'raw_gain_init': 0.1,
                          'raw_rms_limit': 1.0, 'spatial_prior_radius': 1.5},
        )
        block = model.blocks[0]
        self.assertEqual(block.dpe.core.heads, 8)
        self.assertEqual(block.dpe.core.radius, 2.0)
        torch.testing.assert_close(0.5 * block.dpe.gain.tanh(), torch.full_like(block.dpe.gain, 0.02))
        fusion = block.cross_fusion
        self.assertEqual(fusion.raw_encoder[0].out_channels, 128)
        self.assertEqual(fusion.raw_encoder_dropout.p, 0.1)
        self.assertEqual(fusion.raw_cross_heads, 8)
        self.assertEqual(fusion.raw_heads, 4)
        torch.testing.assert_close(fusion.raw_gate.tanh(), torch.full_like(fusion.raw_gate, 0.1))
        self.assertEqual(fusion.raw_rms_limit, 1.0)
        self.assertEqual(fusion.spatial_prior_radius, 1.5)

    def test_configured_multihead_pooling_forward_backward_and_float64(self):
        for dtype in (torch.float32, torch.float64):
            with self.subTest(dtype=dtype):
                fusion = CenterPairedCFAM(3, 16, raw_in_chans=103,
                                          raw_width_ratio=2, pool_heads=4).to(dtype)
                with torch.no_grad():
                    fusion.raw_cross_project.weight.copy_(0.1 * torch.eye(16, dtype=dtype))
                inputs = [torch.randn(2, 3, 3, 16, dtype=dtype, requires_grad=True),
                          torch.randn(2, 3, 3, 16, dtype=dtype, requires_grad=True),
                          torch.randn(2, 1, 16, dtype=dtype, requires_grad=True),
                          torch.randn(2, 103, 3, 3, dtype=dtype, requires_grad=True)]
                output = fusion(*inputs)
                self.assertEqual(output.dtype, dtype)
                self.assertEqual(output.shape, inputs[0].shape)
                self.assertTrue(torch.isfinite(output).all().item())
                output.square().mean().backward()
                for value in inputs:
                    self.assertTrue(torch.isfinite(value.grad).all().item())
                self.assertGreater(inputs[-1].grad.abs().sum().item(), 0)

    @unittest.skipUnless(torch.cuda.is_available(), 'Autocast check requires CUDA')
    def test_multihead_pooling_under_cuda_autocast(self):
        device = torch.device('cuda', torch.cuda.device_count() - 1)
        fusion = CenterPairedCFAM(3, 16, pool_heads=4).to(device)
        with torch.no_grad():
            fusion.raw_cross_project.weight.copy_(0.1 * torch.eye(16, device=device))
        inputs = [torch.randn(2, 3, 3, 16, device=device, requires_grad=True),
                  torch.randn(2, 3, 3, 16, device=device, requires_grad=True),
                  torch.randn(2, 1, 16, device=device, requires_grad=True),
                  torch.randn(2, 200, 3, 3, device=device, requires_grad=True)]
        with torch.autocast('cuda', dtype=torch.float16):
            output = fusion(*inputs)
            loss = output.float().square().mean()
        self.assertTrue(torch.isfinite(output).all().item())
        loss.backward()
        for value in inputs:
            self.assertTrue(torch.isfinite(value.grad).all().item())

    def test_unknown_model_option_is_rejected(self):
        with self.assertRaisesRegex(TypeError, 'ssm_rato'):
            BasicSpaSpeMamba(ssm_rato=7)

    def test_unimplemented_legacy_readouts_fail_explicitly(self):
        for option in ('context_enabled', 'center_readout', 'pool_readout'):
            with self.subTest(option=option), self.assertRaisesRegex(ValueError, 'V139|unsupported'):
                CenterPairedCFAM(3, 8, **{option: True})

    def test_raw_encoder_uses_configured_input_bands(self):
        fusion = BasicSpaSpeMamba(in_chans=103).blocks[0].cross_fusion
        self.assertEqual(fusion.raw_encoder[0].in_channels, 103)
        self.assertEqual(fusion.raw_band_gate[-1].out_features, 103)

    def test_even_spectral_kernel_is_rejected_before_forward(self):
        for kernel in (2, 4, -1, 2.5):
            with self.subTest(kernel=kernel), self.assertRaisesRegex(ValueError, 'spectral|d_conv'):
                SpectralOSSMWithBFS(d_conv=kernel, bfs_enabled=False)

    def test_bfs_rejects_zero_groups_with_a_clear_error(self):
        with self.assertRaisesRegex(ValueError, 'groups'):
            BidirectionalFusionState(2, 16, groups=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'Full spatial scan requires CUDA')
    def test_profile_preserves_random_state_and_model_mode(self):
        device = torch.device('cuda', torch.cuda.device_count() - 1)
        model = BasicSpaSpeMamba(dims=(8,)).to(device).train()
        before_cpu = torch.get_rng_state().clone()
        before_cuda = torch.cuda.get_rng_state(device).clone()
        before_state = {key: value.clone() for key, value in model.state_dict().items()}
        model.flops()
        self.assertTrue(torch.equal(torch.get_rng_state(), before_cpu))
        self.assertTrue(torch.equal(torch.cuda.get_rng_state(device), before_cuda))
        self.assertTrue(model.training)
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before_state[key], rtol=0, atol=0)

    def test_invalid_attention_and_gain_options_are_rejected(self):
        for options in ({'num_heads': 3}, {'radius': 0}, {'gain_init': 0.5},
                        {'gain_bound': -1}, {'radius': math.nan}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                DualMambaDPE(8, **options)
        for options in ({'cross_heads': 3}, {'pool_heads': 0}, {'raw_width_ratio': 0},
                        {'raw_encoder_drop_rate': 1.1}, {'raw_rms_limit': 0},
                        {'raw_gain_init': 1}, {'spatial_prior_radius': math.nan}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                CenterPairedCFAM(3, 8, **options)


if __name__ == '__main__':
    unittest.main()
