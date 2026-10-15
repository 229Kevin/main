"""Checks for mean-pooled X/S/P concatenation without fusion residual/enhancement.

Run from the repository root:
    python -B -m unittest discover -s tests -p 'test_dpe_mamba_newcfam.py' -v
No full training run or persistent checkpoint is created.
"""
import importlib
import io
import json
from pathlib import Path
import sys
import unittest

import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from models import DPE_Mamba_CFAM as previous

torch.set_num_threads(2)


class NewCFAMContracts(unittest.TestCase):
    def setUp(self):
        self.assertTrue((PROJECT / 'models/DPE_Mamba_newCFAM.py').is_file(),
                        'The standalone new CFAM model has not been implemented')
        self.module = importlib.import_module('models.DPE_Mamba_newCFAM')
        self.assertTrue(hasattr(self.module, 'MeanConcatFusion'),
                        'Mean-pooled concatenation fusion has not been implemented')
        torch.manual_seed(34)

    def config(self, dataset):
        return json.loads((PROJECT / f'config/basic_spa_spe_mamba_{dataset}.json').read_text())['model']

    def test_patch_embedding_covers_all_bands_without_grouped_convolution(self):
        for dataset in ('ip', 'up'):
            with self.subTest(dataset=dataset):
                config = self.config(dataset)
                model = self.module.BasicSpaSpeMamba(**config)
                self.assertTrue(hasattr(model, 'patch_embedding'),
                                'Conv2d/GroupNorm/SiLU patch embedding has not been implemented')
                conv, norm, activation = model.patch_embedding
                self.assertEqual(conv.in_channels, config['in_chans'])
                self.assertEqual(conv.out_channels, 64)
                self.assertEqual(conv.kernel_size, (1, 1))
                self.assertEqual(conv.groups, 1)
                self.assertIsInstance(norm, torch.nn.GroupNorm)
                self.assertEqual(norm.num_groups, 4)
                self.assertIsInstance(activation, torch.nn.SiLU)
                output = model.patch_embedding(torch.randn(2, config['in_chans'], 7, 7))
                self.assertEqual(output.shape, (2, 64, 7, 7))
                self.assertTrue(torch.isfinite(output).all())

    def test_backbone_operators_match_original_given_same_weights(self):
        old_config = self.config('ip')
        old_config['fusion_type'] = 'center_cfam_residual'
        old = previous.BasicSpaSpeMamba(**old_config)
        new = self.module.BasicSpaSpeMamba(**self.config('ip'))
        inputs = torch.randn(2, 7, 7, 64)
        for name in ('dpe', 'spectral'):
            old_module = getattr(old.blocks[0], name)
            new_module = getattr(new.blocks[0], name)
            new_module.load_state_dict(old_module.state_dict(), strict=True)
            torch.testing.assert_close(new_module(inputs), old_module(inputs), rtol=0, atol=0)
        new.blocks[0].spatial.load_state_dict(old.blocks[0].spatial.state_dict(), strict=True)
        if torch.cuda.is_available():
            torch.testing.assert_close(new.blocks[0].spatial.cuda()(inputs.cuda()),
                                       old.blocks[0].spatial.cuda()(inputs.cuda()), rtol=0, atol=0)

    def test_seeded_initialization_is_repeatable(self):
        torch.manual_seed(34)
        first = self.module.BasicSpaSpeMamba(**self.config('ip'))
        rng = torch.get_rng_state().clone()
        torch.manual_seed(34)
        second = self.module.BasicSpaSpeMamba(**self.config('ip'))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in first.state_dict().items():
            torch.testing.assert_close(second.state_dict()[name], value, rtol=0, atol=0)

    def test_all_three_features_and_projection_receive_gradients(self):
        fusion = self.module.MeanConcatFusion(7, 64)
        features = [torch.randn(2, 7, 7, 64, requires_grad=True),
                    torch.randn(2, 7, 7, 64, requires_grad=True),
                    torch.randn(2, 1, 64, requires_grad=True)]
        output = fusion(*features)
        self.assertEqual(output.shape, (2, 64))
        output.square().mean().backward()
        for value in features:
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        for name, parameter in fusion.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)
            self.assertGreater(parameter.grad.abs().sum().item(), 0, name)

    def test_mean_pooling_uses_all_positions_and_has_no_residual_or_normalization(self):
        fusion = self.module.MeanConcatFusion(3, 2)
        with torch.no_grad():
            fusion.fusion_proj.weight.copy_(torch.tensor([
                [1., 0., 2., 0., 3., 0.],
                [0., 1., 0., 2., 0., 3.],
            ]))
            fusion.fusion_proj.bias.copy_(torch.tensor([.5, -.5]))
        x = torch.zeros(1, 3, 3, 2)
        s = torch.zeros_like(x)
        x[0, 0, 0] = torch.tensor([9., 18.])
        s[0, 2, 2] = torch.tensor([27., 36.])
        p = torch.tensor([[[5., 6.]]])
        # Means are X=[1,2], S=[3,4]; projection gives [22.5,27.5].
        torch.testing.assert_close(fusion(x, s, p), torch.tensor([[22.5, 27.5]]))
        with torch.no_grad():
            fusion.fusion_proj.weight.zero_()
            fusion.fusion_proj.bias.zero_()
        torch.testing.assert_close(fusion(x, s, p), torch.zeros(1, 2), rtol=0, atol=0)

    def test_pooled_fusion_rejects_multiple_spatial_blocks(self):
        config = self.config('ip')
        config['fusion_type'] = 'mean_concat'
        config['depths'] = [2]
        with self.assertRaises(ValueError):
            self.module.BasicSpaSpeMamba(**config)

    def test_fusion_stays_below_twenty_thousand_parameters(self):
        model = self.module.BasicSpaSpeMamba(**self.config('ip'))
        self.assertLess(sum(p.numel() for p in model.blocks[0].cross_fusion.parameters()), 20000)
        self.assertLess(sum(p.numel() for p in model.parameters()), 150000)

    def test_invalid_feature_shapes_are_rejected(self):
        fusion = self.module.MeanConcatFusion(7, 64)
        x = torch.randn(2, 7, 7, 64)
        with self.assertRaises(ValueError):
            fusion(x, torch.randn(2, 5, 5, 64), torch.randn(2, 1, 64))
        with self.assertRaises(ValueError):
            fusion(x, x, torch.randn(2, 64))
        with self.assertRaises(ValueError):
            fusion(x, x, torch.randn(1, 1, 64))

    def test_training_factory_selects_new_version_for_both_configs(self):
        from models.get_model import get_model
        for dataset in ('ip', 'up'):
            with self.subTest(dataset=dataset):
                model = get_model(self.config(dataset), 'basic_spa_spe_mamba', dataset, 7)
                self.assertIsInstance(model, self.module.BasicSpaSpeMamba)
                self.assertIsInstance(model.blocks[0].cross_fusion, self.module.MeanConcatFusion)

    @unittest.skipUnless(torch.cuda.is_available(), 'Spatial OSSM requires CUDA')
    def test_complete_models_backward_and_checkpoint_roundtrip(self):
        from models.get_model import get_model
        device = torch.device('cuda', torch.cuda.device_count() - 1)
        for dataset in ('ip', 'up'):
            with self.subTest(dataset=dataset):
                config = self.config(dataset)
                model = get_model(config, 'basic_spa_spe_mamba', dataset, 7).to(device).train()
                x = torch.randn(2, 1, config['in_chans'], 7, 7, device=device)
                captures = {}
                h1 = model.patch_embedding.register_forward_hook(
                    lambda module, args, result: captures.update(raw=args[0].detach(), embedded=result.detach()))
                h2 = model.blocks[0].dpe.register_forward_pre_hook(
                    lambda module, args: captures.update(dpe_input=args[0].detach()))
                output = model(x)
                h1.remove()
                h2.remove()
                torch.testing.assert_close(captures['raw'], x.squeeze(1), rtol=0, atol=0)
                torch.testing.assert_close(captures['dpe_input'],
                                           captures['embedded'].permute(0, 2, 3, 1), rtol=0, atol=0)
                self.assertEqual(output.shape, (2, config['num_classes']))
                torch.nn.functional.cross_entropy(output, torch.tensor([0, 1], device=device)).backward()
                for name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                model.eval()
                buffer = io.BytesIO()
                torch.save(model.state_dict(), buffer)
                buffer.seek(0)
                restored = self.module.BasicSpaSpeMamba(**config).to(device).eval()
                restored.load_state_dict(torch.load(buffer, map_location=device), strict=True)
                with torch.no_grad():
                    expected = model(x)
                    self.assertTrue(torch.isfinite(expected).all())
                    torch.testing.assert_close(restored(x), expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
