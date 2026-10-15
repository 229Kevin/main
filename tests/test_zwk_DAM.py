"""Numerical and spatial contracts for the standalone direction-aware module."""

import importlib
import importlib.util
import io
import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class DAMTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('models.zwk_DAM'),
                             'The standalone DAM module has not been implemented')
        return importlib.import_module('models.zwk_DAM')

    def test_paths_and_direction_labels_match_hand_checked_grid(self):
        orders, inverses, directions = self.module().build_scan_indices(3, 3)
        self.assertEqual(orders[0].tolist(), [0, 1, 2, 5, 4, 3, 6, 7, 8])
        self.assertEqual(orders[2].tolist(), [0, 3, 6, 7, 4, 1, 2, 5, 8])
        self.assertEqual(directions[0].tolist(), [0, 4, 4, 2, 3, 3, 2, 4, 4])
        self.assertEqual(directions[1].tolist(), [0, 3, 3, 1, 4, 4, 1, 3, 3])
        for path in range(4):
            torch.testing.assert_close(orders[path][inverses[path]], torch.arange(9))

    def test_paths_cover_rectangular_and_degenerate_grids_without_jumps(self):
        for height, width in ((3, 5), (4, 6), (1, 5), (5, 1), (1, 1)):
            with self.subTest(shape=(height, width)):
                orders, inverses, _ = self.module().build_scan_indices(height, width)
                for order, inverse in zip(orders, inverses):
                    torch.testing.assert_close(order.sort().values, torch.arange(height * width))
                    torch.testing.assert_close(order[inverse], torch.arange(height * width))
                    rows, cols = order // width, order % width
                    steps = (rows[1:] - rows[:-1]).abs() + (cols[1:] - cols[:-1]).abs()
                    self.assertTrue((steps == 1).all())

    def test_reference_recurrence_matches_closed_form_and_backpropagates(self):
        scan = self.module().selective_scan_reference
        u = torch.tensor([[[[1., 2., 3.]]]], dtype=torch.float64, requires_grad=True)
        delta = torch.full_like(u, 0.2)
        a = torch.tensor([[[-1.]]], dtype=torch.float64)
        b = torch.full((1, 1, 1, 3), 2., dtype=torch.float64)
        c = torch.full_like(b, 3.)
        skip = torch.tensor([[0.5]], dtype=torch.float64)
        decay = math.exp(-0.2)
        h1 = 0.4
        h2 = decay * h1 + 0.8
        h3 = decay * h2 + 1.2
        expected = torch.tensor([[[[3*h1+0.5, 3*h2+1., 3*h3+1.5]]]], dtype=torch.float64)
        output = scan(u, delta, a, b, c, skip)
        torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
        output.sum().backward()
        self.assertTrue(torch.isfinite(u.grad).all())
        self.assertTrue((u.grad > 0).all())

    def test_module_supports_patch_sizes_and_checkpoint_after_cache_changes(self):
        cls = self.module().ZWKDAM
        model = cls(d_model=12, scan_backend='reference').eval()
        for shape in ((2, 12, 15, 15), (1, 12, 4, 6), (1, 12, 1, 1)):
            with torch.no_grad():
                output = model(torch.randn(*shape))
            self.assertEqual(tuple(output.shape), shape)
            self.assertTrue(torch.isfinite(output).all())
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = cls(d_model=12, scan_backend='reference').eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        features = torch.randn(1, 12, 3, 5)
        torch.testing.assert_close(restored(features), model(features), rtol=0, atol=0)

    def test_zero_compensation_matches_disabled_branch_and_nonzero_changes_output(self):
        cls = self.module().ZWKDAM
        torch.manual_seed(1)
        enabled = cls(d_model=8, scan_backend='reference').eval()
        disabled = cls(d_model=8, direction_enabled=False, scan_backend='reference').eval()
        disabled.load_state_dict(enabled.state_dict())
        features = torch.randn(2, 8, 5, 7)
        expected = disabled(features)
        torch.testing.assert_close(enabled(features), expected, rtol=0, atol=0)
        with torch.no_grad():
            enabled.direction_table.normal_(std=0.4)
        self.assertGreater((enabled(features) - expected).abs().max().item(), 1e-5)

    def test_training_reaches_all_parameters_including_zero_initialized_directions(self):
        torch.manual_seed(7)
        model = self.module().ZWKDAM(scan_backend='reference')
        features = torch.randn(2, 64, 15, 15, requires_grad=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        before = model.direction_table.detach().clone()
        output = model(features)
        loss = (output * torch.randn_like(output)).mean()
        loss.backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertTrue((model.direction_table.grad.abs().sum(1) > 0).all())
        self.assertGreater(features.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, model.direction_table))

    def test_invalid_configuration_and_input_are_rejected(self):
        cls = self.module().ZWKDAM
        for options in ({'d_model': 0}, {'d_state': 0}, {'ssm_ratio': float('nan')},
                        {'ssm_ratio': 0}, {'dt_rank': 0}, {'d_conv': 2},
                        {'dt_min': 0}, {'dt_max': float('inf')},
                        {'dt_min': 0.2, 'dt_max': 0.1}, {'direction_enabled': 'false'},
                        {'scan_backend': 'unknown'}, {'dropout': 1.}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                cls(**options)
        model = cls(d_model=8)
        for shape in ((1, 8, 3), (1, 9, 3, 3), (1, 8, 0, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.randn(*shape))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_matches_reference_outputs_and_all_parameter_gradients(self):
        cls = self.module().ZWKDAM
        torch.manual_seed(3)
        reference = cls(d_model=12, scan_backend='reference').cuda()
        accelerated = cls(d_model=12, scan_backend='cuda').cuda()
        with torch.no_grad():
            reference.direction_table.normal_(std=0.15)
        accelerated.load_state_dict(reference.state_dict())
        x = torch.randn(2, 12, 5, 7, device='cuda', requires_grad=True)
        xx = x.detach().clone().requires_grad_(True)
        weight = torch.randn_like(x)
        expected, actual = reference(x), accelerated(xx)
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-5)
        (expected * weight).sum().backward()
        (actual * weight).sum().backward()
        torch.testing.assert_close(xx.grad, x.grad, rtol=3e-3, atol=2e-4)
        for (name, parameter), (_, other) in zip(reference.named_parameters(), accelerated.named_parameters()):
            with self.subTest(parameter=name):
                torch.testing.assert_close(other.grad, parameter.grad, rtol=3e-3, atol=2e-4)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA is unavailable')
    def test_cuda_autocast_and_full_patch_training_are_finite(self):
        model = self.module().ZWKDAM(scan_backend='cuda').cuda()
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model.zero_grad(set_to_none=True)
                features = torch.randn(2, 64, 15, 15, device='cuda', requires_grad=True)
                with torch.autocast('cuda', dtype=dtype):
                    output = model(features)
                    loss = output.float().square().mean()
                self.assertEqual(tuple(output.shape), tuple(features.shape))
                self.assertTrue(torch.isfinite(output).all())
                loss.backward()
                for parameter in model.parameters():
                    self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
