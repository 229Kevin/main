"""Input and training contracts for the minimal HSI baseline."""

import sys
import unittest
import contextlib
import io
import json
import ast
import math
from pathlib import Path
from unittest.mock import patch

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.zwk_hsi import ZWKHSI


class ZWKHSITests(unittest.TestCase):
    def test_stem_feeds_psa_directly_when_cnn_is_disabled(self):
        torch.manual_seed(42)
        for channels in (12, 64):
            with self.subTest(channels=channels):
                model = ZWKHSI(num_bands=8, num_classes=3, embed_dim=channels,
                               cnn_enabled=False)
                self.assertIsInstance(model.cnn, torch.nn.Identity)
                self.assertFalse(any(isinstance(module, torch.nn.Conv3d)
                                     for module in model.modules()))
                captured = {}
                hooks = [
                    model.stem.register_forward_hook(
                        lambda module, inputs, output: captured.update(stem=output)),
                    model.psa.register_forward_pre_hook(
                        lambda module, inputs: captured.update(psa_input=inputs[0])),
                    model.psa.register_forward_hook(
                        lambda module, inputs, output: captured.update(psa=output)),
                    model.pool.register_forward_pre_hook(
                        lambda module, inputs: captured.update(pool_input=inputs[0])),
                ]
                try:
                    features = torch.randn(2, 1, 8, 15, 15, requires_grad=True)
                    logits = model(features)
                finally:
                    for hook in hooks:
                        hook.remove()
                self.assertIs(captured['psa_input'], captured['stem'])
                self.assertEqual(tuple(captured['psa'].shape), (2, channels, 15, 15))
                torch.testing.assert_close(
                    captured['pool_input'],
                    captured['stem'] + model.psa_residual_scale * captured['psa'],
                    rtol=0, atol=0)
                self.assertEqual(tuple(logits.shape), (2, 3))
                torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
                self.assertGreater(features.grad.abs().sum().item(), 0)
                for name, parameter in model.named_parameters():
                    with self.subTest(parameter=name):
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_cnn_switch_requires_boolean(self):
        with self.assertRaisesRegex(ValueError, 'cnn_enabled'):
            ZWKHSI(num_bands=8, num_classes=3, cnn_enabled='false')

    def test_psa_residual_scale_is_optimized_and_restored_from_checkpoint(self):
        torch.manual_seed(7)
        model = ZWKHSI(num_bands=8, num_classes=3)
        alpha = model.psa_residual_scale
        self.assertIsInstance(alpha, torch.nn.Parameter)
        self.assertTrue(alpha.requires_grad)
        self.assertAlmostEqual(alpha.item(), 0.1, places=7)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        self.assertTrue(any(parameter is alpha for group in optimizer.param_groups
                            for parameter in group['params']))
        before = alpha.detach().clone()
        features = torch.randn(2, 1, 8, 13, 13)
        loss = torch.nn.functional.cross_entropy(model(features), torch.tensor([0, 2]))
        loss.backward()
        self.assertIsNotNone(alpha.grad)
        self.assertTrue(torch.isfinite(alpha.grad))
        self.assertGreater(alpha.grad.abs().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(alpha, before))
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = ZWKHSI(num_bands=8, num_classes=3, psa_residual_scale=0.9)
        restored.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=True))
        torch.testing.assert_close(restored.psa_residual_scale, alpha, rtol=0, atol=0)
        model.eval()
        restored.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(features), model(features), rtol=0, atol=0)

    def test_psa_residual_scale_is_frozen_when_psa_is_disabled(self):
        model = ZWKHSI(num_bands=8, num_classes=3, psa_enabled=False)
        self.assertFalse(model.psa_residual_scale.requires_grad)

    def test_psa_normalizes_each_spatial_position_before_generating_qkv(self):
        from models.zwk_hsi import SingleLayerPSA

        psa = SingleLayerPSA(dim=4, num_heads=2, pre_norm=True).double()
        features = torch.arange(24, dtype=torch.float64).reshape(1, 4, 2, 3)
        captured = []
        hook = psa.qkv.register_forward_pre_hook(
            lambda module, inputs: captured.append(inputs[0].detach().clone()))
        try:
            with torch.no_grad():
                output = psa(features)
        finally:
            hook.remove()
        mean = features.mean(dim=1, keepdim=True)
        variance = (features - mean).square().mean(dim=1, keepdim=True)
        expected = (features - mean) / torch.sqrt(variance + 1e-5)
        torch.testing.assert_close(captured[0], expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(tuple(output.shape), (1, 4, 2, 3))

    def test_normalized_center_weighted_psa_supports_two_and_four_heads(self):
        from models.zwk_hsi import SingleLayerPSA

        for heads in (2, 4):
            with self.subTest(heads=heads):
                psa = SingleLayerPSA(dim=64, num_heads=heads, pre_norm=True)
                features = torch.randn(2, 64, 13, 13, requires_grad=True)
                output = psa(features)
                self.assertEqual(tuple(output.shape), (2, 64, 13, 13))
                self.assertTrue(torch.isfinite(output).all())
                output.square().mean().backward()
                self.assertTrue(torch.isfinite(features.grad).all())
                self.assertGreater(features.grad.abs().sum().item(), 0)
                for name, parameter in psa.named_parameters():
                    with self.subTest(parameter=name):
                        self.assertIsNotNone(parameter.grad)
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    def test_psa_channel_relations_use_center_weighted_cosine_and_unweighted_values(self):
        from models.zwk_hsi import SingleLayerPSA

        # At distance sqrt(2), this sigma gives Gaussian weight 1/4.
        psa = SingleLayerPSA(dim=2, num_heads=1, center_weighted=True,
                             center_sigma=math.sqrt(1 / math.log(4)), pre_norm=False).double()
        with torch.no_grad():
            for convolution in (psa.qkv, psa.qkv_dwconv, psa.project_out):
                convolution.weight.zero_()
                convolution.bias.zero_()
            for channel in range(6):
                psa.qkv.weight[channel, channel % 2, 0, 0] = 1
                psa.qkv_dwconv.weight[channel, 0, 1, 1] = 1
            psa.project_out.weight[0, 0, 0, 0] = 1
            psa.project_out.weight[1, 1, 0, 0] = 1
        features = torch.zeros(1, 2, 3, 3, dtype=torch.float64)
        features[0, :, 1, 1] = 1
        features[0, 0, 0, 0] = 1
        features[0, 1, 0, 0] = -1
        # Weighted cross-channel cosine: (1 - 1/4) / (1 + 1/4) = 0.6.
        expected_attention = torch.tensor([[1., .6], [.6, 1.]], dtype=torch.float64).softmax(-1)
        expected = (expected_attention @ features.flatten(2)).reshape(1, 2, 3, 3)
        torch.testing.assert_close(psa(features), expected, rtol=1e-12, atol=1e-12)

    def test_psa_center_sigma_must_be_positive_and_finite(self):
        from models.zwk_hsi import SingleLayerPSA

        for sigma in (0., -1., float('inf'), float('nan')):
            with self.subTest(sigma=sigma), self.assertRaises(ValueError):
                SingleLayerPSA(dim=8, num_heads=2, center_sigma=sigma)

    def test_psa_depthwise_convolution_uses_contiguous_three_by_three_neighborhood(self):
        from models.zwk_hsi import SingleLayerPSA

        convolution = SingleLayerPSA(dim=8, num_heads=2).qkv_dwconv
        with torch.no_grad():
            convolution.weight.fill_(1)
            convolution.bias.zero_()
        features = torch.zeros(1, 24, 13, 13)
        features[0, 0, 6, 6] = 1
        expected = torch.zeros(1, 24, 13, 13)
        expected[0, 0, 5:8, 5:8] = 1
        with torch.no_grad():
            output = convolution(features)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)

    def test_pool_initially_receives_cnn_features_plus_one_tenth_psa_output(self):
        model = ZWKHSI(num_bands=8, num_classes=3).eval()
        features = {}
        hooks = [
            model.cnn.register_forward_hook(
                lambda module, inputs, output: features.update(cnn=output)),
            model.psa.register_forward_hook(
                lambda module, inputs, output: features.update(psa=output)),
            model.pool.register_forward_pre_hook(
                lambda module, inputs: features.update(pool_input=inputs[0])),
        ]
        try:
            with torch.no_grad():
                model(torch.randn(2, 1, 8, 13, 13))
        finally:
            for hook in hooks:
                hook.remove()
        self.assertGreater(features['psa'].abs().sum().item(), 0)
        torch.testing.assert_close(features['pool_input'],
                                   features['cnn'] + 0.1 * features['psa'],
                                   rtol=0, atol=0)

    def test_unweighted_psa_matches_hypyramamba_with_local_convolution_and_gradients(self):
        from einops import rearrange
        from models.zwk_hsi import SingleLayerPSA

        source = Path('/home/zwk/CVPR/HSI/HyPyraMamba/HypraMamba/model.py')
        tree = ast.parse(source.read_text())
        attention = next(node for node in tree.body
                         if isinstance(node, ast.ClassDef) and node.name == 'PyramidAttention')
        scope = {'torch': torch, 'nn': torch.nn, 'rearrange': rearrange}
        exec(compile(ast.Module(body=[attention], type_ignores=[]), str(source), 'exec'), scope)
        reference = scope['PyramidAttention'](8, 2, True).double()
        # Keep original attention math, adapting only the QKV depthwise convolution.
        reference.qkv_dwconv = torch.nn.Conv2d(
            24, 24, kernel_size=3, stride=1, padding=1, groups=24, bias=True,
        ).double()
        actual = SingleLayerPSA(8, 2, True, center_weighted=False, pre_norm=False).double()
        actual.load_state_dict(reference.state_dict(), strict=True)
        x = torch.randn(2, 8, 5, 7, dtype=torch.float64, requires_grad=True)
        xx = x.detach().clone().requires_grad_()
        expected = reference(x)
        output = actual(xx)
        torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
        reference_gradients = torch.autograd.grad(expected.square().mean(), (x, *reference.parameters()))
        actual_gradients = torch.autograd.grad(output.square().mean(), (xx, *actual.parameters()))
        for expected_gradient, actual_gradient in zip(reference_gradients, actual_gradients):
            torch.testing.assert_close(actual_gradient, expected_gradient, rtol=1e-12, atol=1e-12)

    def test_psa_runs_once_between_cnn_and_pool_without_downsampling(self):
        model = ZWKHSI(num_bands=8, num_classes=3).eval()
        events = []
        attention_shapes = []
        hooks = [module.register_forward_hook(
            lambda module, inputs, output, name=name:
                events.append((name, tuple(inputs[0].shape), tuple(output.shape))))
            for name, module in (('cnn', model.cnn), ('psa', model.psa), ('pool', model.pool))]
        hooks.append(model.psa.qkv.register_forward_pre_hook(
            lambda module, inputs: attention_shapes.append(tuple(inputs[0].shape))))
        try:
            with torch.no_grad(), \
                 patch.object(torch.nn.functional, 'avg_pool2d', side_effect=AssertionError('PSA downsampled')), \
                 patch.object(torch.nn.functional, 'interpolate', side_effect=AssertionError('PSA resampled')):
                logits = model(torch.randn(2, 1, 8, 13, 13))
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual([event[0] for event in events], ['cnn', 'psa', 'pool'])
        self.assertEqual(events[1][1:], ((2, 64, 13, 13), (2, 64, 13, 13)))
        self.assertEqual(attention_shapes, [(2, 64, 13, 13)])
        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_psa_can_be_disabled_and_backbone_weights_are_preserved(self):
        torch.manual_seed(123)
        baseline = ZWKHSI(num_bands=8, num_classes=3, psa_enabled=False)
        torch.manual_seed(123)
        with_psa = ZWKHSI(num_bands=8, num_classes=3, psa_enabled=True)
        self.assertIsInstance(baseline.psa, torch.nn.Identity)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(with_psa.state_dict()[name], value, rtol=0, atol=0)
        baseline.eval()
        x = torch.randn(2, 1, 8, 13, 13)
        with torch.no_grad():
            features = baseline.pool(baseline.cnn(baseline.stem(x.squeeze(1)).unsqueeze(1))).flatten(1)
            expected = baseline.classifier(features)
            torch.testing.assert_close(baseline(x), expected, rtol=0, atol=0)

    def test_psa_rejects_invalid_head_count(self):
        from models.zwk_hsi import SingleLayerPSA
        for heads in (0, 3):
            with self.subTest(heads=heads), self.assertRaises(ValueError):
                SingleLayerPSA(dim=8, num_heads=heads)

    def test_residual_ffn_preserves_features_and_gradients_when_branch_is_zero(self):
        model = ZWKHSI(num_bands=8, num_classes=3,
                       conv2d_channels=6, ffn_hidden_dim=10, psa_num_heads=2).eval()
        residual_ffn = model.classifier[0]
        with torch.no_grad():
            for parameter in residual_ffn.ffn.parameters():
                parameter.zero_()
        features = torch.tensor([
            [1., 2., 3., 4., 5., 6.],
            [-2., 0., 1., 3., 4., 7.],
        ], requires_grad=True)
        output = residual_ffn(features)
        torch.testing.assert_close(output, features, rtol=0, atol=0)
        output.sum().backward()
        torch.testing.assert_close(features.grad, torch.ones(2, 6), rtol=0, atol=0)
        with torch.no_grad():
            logits = model.classifier(features)
        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_cnn_uses_3d_then_2d_with_depth_folded_into_channels(self):
        model = ZWKHSI(num_bands=8, num_classes=3, embed_dim=4,
                       conv3d_channels=2, conv2d_channels=6, psa_num_heads=2).eval()
        events = []

        def record(module, inputs, output):
            events.append((type(module), tuple(inputs[0].shape), tuple(output.shape)))

        tracked = (torch.nn.Conv3d, torch.nn.Conv2d, torch.nn.BatchNorm3d,
                   torch.nn.BatchNorm2d, torch.nn.PReLU)
        hooks = [module.register_forward_hook(record) for module in model.cnn.modules()
                 if isinstance(module, tracked)]
        try:
            with torch.no_grad():
                output = model.cnn(torch.randn(2, 1, 4, 5, 7))
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual([event[0] for event in events], [
            torch.nn.Conv3d, torch.nn.BatchNorm3d, torch.nn.PReLU,
            torch.nn.Conv2d, torch.nn.BatchNorm2d, torch.nn.PReLU,
        ])
        self.assertEqual(events[3][1], (2, 8, 5, 7))
        self.assertEqual(tuple(output.shape), (2, 6, 5, 7))
        with torch.no_grad():
            logits = model(torch.randn(2, 1, 8, 5, 7))
        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_both_convolution_channel_counts_must_be_positive(self):
        for channels_3d, channels_2d in ((0, 64), (8, 0)):
            with self.subTest(conv3d=channels_3d, conv2d=channels_2d):
                with self.assertRaises(ValueError):
                    ZWKHSI(num_bands=8, num_classes=3,
                           conv3d_channels=channels_3d, conv2d_channels=channels_2d)

    def test_cnn_batch_norm_updates_during_training_and_is_stable_in_eval(self):
        model = ZWKHSI(num_bands=8, num_classes=3)
        norms = [module for module in model.cnn.modules()
                 if isinstance(module, (torch.nn.BatchNorm3d, torch.nn.BatchNorm2d))]
        self.assertEqual(len(norms), 2)
        patch = torch.randn(2, 1, 8, 7, 7)
        model(patch)
        for norm in norms:
            self.assertEqual(norm.num_batches_tracked.item(), 1)
        model.eval()
        with torch.no_grad():
            first = model(patch)
            second = model(patch)
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        for norm in norms:
            self.assertEqual(norm.num_batches_tracked.item(), 1)

    def test_patch_classifier_supports_different_bands_and_patch_sizes(self):
        for bands, classes, size in ((200, 16, 13), (103, 9, 13), (8, 3, 1)):
            with self.subTest(bands=bands, size=size):
                model = ZWKHSI(num_bands=bands, num_classes=classes).eval()
                with torch.no_grad():
                    logits = model(torch.randn(1, 1, bands, size, size))
                self.assertEqual(tuple(logits.shape), (1, classes))
                self.assertTrue(torch.isfinite(logits).all())

    def test_four_and_five_dimensional_inputs_match(self):
        model = ZWKHSI(num_bands=8, num_classes=3).eval()
        patch = torch.randn(2, 8, 5, 7)
        with torch.no_grad():
            torch.testing.assert_close(model(patch), model(patch.unsqueeze(1)))

    def test_cross_entropy_reaches_input_and_all_trainable_parameters(self):
        torch.manual_seed(42)
        model = ZWKHSI(num_bands=8, num_classes=3)
        patch = torch.randn(2, 1, 8, 7, 7, requires_grad=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        before = model.classifier[-1].weight.detach().clone()
        loss = torch.nn.functional.cross_entropy(model(patch), torch.tensor([0, 2]))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue(torch.isfinite(patch.grad).all())
        self.assertGreater(patch.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.classifier[-1].weight))

    def test_malformed_patch_shapes_are_rejected(self):
        model = ZWKHSI(num_bands=8, num_classes=3)
        for shape in ((2, 8, 7), (2, 2, 8, 7, 7), (2, 9, 7, 7)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.randn(*shape))

    def test_factory_configs_support_a_training_step(self):
        from models.get_model import get_model
        from utils.scheduler import load_scheduler

        project = Path(__file__).resolve().parents[1]
        for dataset, bands, classes in (("ip", 200, 16), ("up", 103, 9)):
            with self.subTest(dataset=dataset):
                config = json.loads((project / f"config/zwk_hsi_{dataset}.json").read_text())
                model = get_model(config["model"], "zwk_hsi", dataset, 15)
                self.assertIsInstance(model, ZWKHSI)
                self.assertEqual(model.cnn_enabled, config["model"].get("cnn_enabled", True))
                if model.cnn_enabled:
                    self.assertEqual(model.cnn.conv3d[0].out_channels, 8)
                    self.assertEqual(model.cnn.conv2d[0].out_channels, 64)
                else:
                    self.assertIsInstance(model.cnn, torch.nn.Identity)
                    self.assertEqual(model.stem[0].out_channels, 64)
                self.assertEqual(model.psa_enabled, config["model"]["psa_enabled"])
                if model.psa_enabled:
                    self.assertEqual(model.psa.center_weighted, config["model"]["psa_center_weighted"])
                    self.assertEqual(model.psa.pre_norm, config["model"]["psa_pre_norm"])
                    self.assertEqual(model.psa.num_heads, config["model"]["psa_num_heads"])
                else:
                    self.assertIsInstance(model.psa, torch.nn.Identity)
                optimizer, scheduler = load_scheduler(config["schedule"], "zwk_hsi", model)
                self.assertIsInstance(optimizer, torch.optim.AdamW)
                self.assertIsInstance(scheduler, torch.optim.lr_scheduler.StepLR)
                logits = model(torch.randn(2, 1, bands, 15, 15))
                self.assertEqual(tuple(logits.shape), (2, classes))
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, classes - 1]))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                self.assertTrue(torch.isfinite(loss))

    def test_flops_interface_preserves_training_state_and_weights(self):
        model = ZWKHSI(num_bands=8, num_classes=3).train()
        before = {name: value.clone() for name, value in model.state_dict().items()}
        rng_before = torch.get_rng_state().clone()
        with contextlib.redirect_stdout(io.StringIO()) as output:
            gflops = model.flops(shape=(1, 8, 7, 7))
        self.assertGreater(gflops, 0)
        self.assertIn("GFLOPs", output.getvalue())
        self.assertTrue(model.training)
        self.assertTrue(torch.equal(rng_before, torch.get_rng_state()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    torch.set_num_threads(2)
    unittest.main()
