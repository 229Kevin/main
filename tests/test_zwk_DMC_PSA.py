"""DMC+PSA ablation: routing, retained baseline, training and registration."""

import copy
import importlib
import io
import json
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.zwk_DMC import ZWKDMC
from models.zwk_hsi import SingleLayerPSA


class DMCPSATests(unittest.TestCase):
    def classifier(self):
        cls = getattr(importlib.import_module('models.zwk_DAM'), 'ZWKDMCPSA', None)
        self.assertIsNotNone(cls, 'DMC+PSA classifier must be provided in zwk_DAM.py')
        return cls

    def options(self):
        return dict(num_bands=8, num_classes=3, feature_dim=8, groups=2, region_size=5)

    def test_residual_psa_fuses_exactly_once_before_pooling_without_dam(self):
        model = self.classifier()(**self.options()).eval()
        from models.zwk_DAM import ZWKDAM
        self.assertIsInstance(model.psa, SingleLayerPSA)
        self.assertFalse(any(isinstance(module, ZWKDAM) for module in model.modules()))
        seen = {}
        handles = [
            model.dmc.register_forward_hook(lambda m, a, out: seen.update(dmc=out)),
            model.psa.register_forward_pre_hook(lambda m, a: seen.update(psa_input=a[0])),
            model.psa.register_forward_hook(lambda m, a, out: seen.update(psa=out)),
            model.pool.register_forward_pre_hook(lambda m, a: seen.update(pool_input=a[0])),
            model.ffn.register_forward_pre_hook(lambda m, a: seen.update(ffn_input=a[0])),
        ]
        try:
            with torch.no_grad():
                output = model(torch.randn(2, 1, 8, 15, 15))
        finally:
            for handle in handles:
                handle.remove()
        self.assertIs(seen['psa_input'], seen['dmc'])
        self.assertEqual(tuple(seen['psa'].shape), (2, 8, 15, 15))
        torch.testing.assert_close(seen['pool_input'], seen['dmc'] + seen['psa'], rtol=0, atol=0)
        torch.testing.assert_close(seen['ffn_input'], seen['pool_input'].mean((2, 3)))
        self.assertEqual(tuple(output.shape), (2, 3))

    def test_new_branch_preserves_baseline_initial_weights_and_rng(self):
        cls = self.classifier()
        for enabled in (False, True):
            torch.manual_seed(43)
            baseline = ZWKDMC(**self.options()).eval()
            baseline_rng = torch.get_rng_state().clone()
            torch.manual_seed(43)
            model = cls(**self.options(), psa_enabled=enabled).eval()
            torch.testing.assert_close(torch.get_rng_state(), baseline_rng, rtol=0, atol=0)
            for name, value in baseline.state_dict().items():
                torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
            if not enabled:
                model.load_state_dict(baseline.state_dict(), strict=True)
                for shape in ((2, 8, 13, 13), (2, 1, 8, 5, 7)):
                    features = torch.randn(*shape)
                    with torch.no_grad():
                        torch.testing.assert_close(model(features), baseline(features), rtol=0, atol=0)

    def test_zero_psa_output_reproduces_baseline_logits(self):
        torch.manual_seed(3)
        baseline = ZWKDMC(**self.options()).eval()
        torch.manual_seed(3)
        model = self.classifier()(**self.options()).eval()
        with torch.no_grad():
            for parameter in model.psa.parameters():
                parameter.zero_()
            features = torch.randn(2, 1, 8, 15, 15)
            torch.testing.assert_close(model(features), baseline(features), rtol=0, atol=0)

    def test_four_and_five_dimensional_inputs_match_and_invalid_inputs_fail(self):
        model = self.classifier()(**self.options()).eval()
        for height, width in ((13, 13), (15, 15), (5, 7)):
            features = torch.randn(2, 8, height, width)
            with torch.no_grad():
                torch.testing.assert_close(model(features), model(features.unsqueeze(1)), rtol=0, atol=0)
        for shape in ((2, 9, 3, 3), (2, 2, 8, 3, 3), (2, 8, 3), (2, 8, 0, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                model(torch.randn(*shape))
        for options in ({'psa_enabled': 'false'}, {'psa_num_heads': 3},
                        {'psa_center_sigma': 0.}, {'psa_pre_norm': 'true'}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.classifier()(**self.options(), **options)

    def test_all_parameters_receive_gradients_and_checkpoint_roundtrips(self):
        torch.manual_seed(11)
        cls = self.classifier()
        model = cls(**self.options())
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        features = torch.randn(2, 1, 8, 15, 15, requires_grad=True)
        before = model.psa.project_out.weight.detach().clone()
        loss = torch.nn.functional.cross_entropy(model(features), torch.tensor([0, 2]))
        loss.backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0.)
        optimizer.step()
        self.assertFalse(torch.equal(model.psa.project_out.weight, before))
        checkpoint = io.BytesIO()
        torch.save(model.state_dict(), checkpoint)
        checkpoint.seek(0)
        restored = cls(**self.options()).eval()
        restored.load_state_dict(torch.load(checkpoint, weights_only=True), strict=True)
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(features), model(features), rtol=0, atol=0)

    def test_deployment_conversion_preserves_psa_and_predictions(self):
        model = self.classifier()(**self.options()).double().eval()
        features = torch.randn(2, 1, 8, 5, 7, dtype=torch.float64)
        with torch.no_grad():
            expected = model(features)
            model.switch_to_deploy()
            torch.testing.assert_close(model(features), expected, rtol=1e-10, atol=1e-10)
        restored = self.classifier()(**self.options(), deploy=True).double().eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(features), model(features), rtol=0, atol=0)

    def test_factory_config_flops_and_optimizer_support_real_training_entry(self):
        from models.get_model import get_model
        from utils.scheduler import load_scheduler
        root = Path(__file__).resolve().parents[1]
        config = json.loads((root / 'config/zwk_dmc_psa_ip.json').read_text())
        baseline_config = json.loads((root / 'config/zwk_dmc_ip.json').read_text())
        self.assertEqual({k: v for k, v in config['model'].items() if not k.startswith('psa_')}, baseline_config['model'])
        self.assertEqual(config['schedule'], baseline_config['schedule'])
        model = get_model(config['model'], 'zwk_dmc_psa', 'ip', 15)
        self.assertIsInstance(model, self.classifier())
        self.assertEqual(sum(p.numel() for p in model.parameters()), 95547)
        before = copy.deepcopy(model.state_dict())
        rng = torch.get_rng_state().clone()
        self.assertGreater(model.flops(shape=(1, 200, 15, 15), verbose=False),
                           ZWKDMC(**baseline_config['model']).flops(verbose=False))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        # Baseline construction above consumes RNG; measure the profiler separately.
        rng = torch.get_rng_state().clone()
        model.flops(shape=(1, 200, 15, 15), verbose=False)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        optimizer, scheduler = load_scheduler(config['schedule'], 'zwk_dmc_psa', model)
        output = model(torch.randn(2, 1, 200, 15, 15))
        self.assertEqual(tuple(output.shape), (2, 16))
        torch.nn.functional.cross_entropy(output, torch.tensor([0, 15])).backward()
        optimizer.step()
        scheduler.step()

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_mixed_precision_keeps_all_branch_gradients_finite(self):
        for dtype in (torch.float16, torch.bfloat16):
            model = self.classifier()(**self.options()).cuda().train()
            features = torch.randn(2, 1, 8, 15, 15, device='cuda', requires_grad=True)
            with torch.autocast('cuda', dtype=dtype):
                output = model(features)
                loss = torch.nn.functional.cross_entropy(output, torch.tensor([0, 2], device='cuda'))
            loss.backward()
            self.assertTrue(torch.isfinite(features.grad).all())
            for name, parameter in model.named_parameters():
                with self.subTest(dtype=dtype, parameter=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
