"""Contracts for the serial residual DMC -> PSA -> DAM patch classifier."""

import copy
import importlib
import io
import json
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.zwk_DAM import ZWKDAM, ZWKDMCPSA


class DMCPSADAMTests(unittest.TestCase):
    def classifier(self):
        cls = getattr(importlib.import_module('models.zwk_DAM'), 'ZWKDMCPSADAM', None)
        self.assertIsNotNone(cls, 'Complete DMC+PSA+DAM classifier is missing')
        return cls

    def options(self):
        return dict(num_bands=8, num_classes=3, feature_dim=8, groups=2,
                    region_size=5, dam_d_state=4, dam_dt_rank=2)

    def test_dam_receives_residual_psa_features_and_fuses_exactly_once(self):
        model = self.classifier()(**self.options()).eval()
        seen = {}
        handles = [
            model.dmc.register_forward_hook(lambda m, a, out: seen.update(dmc=out)),
            model.psa.register_forward_hook(lambda m, a, out: seen.update(psa=out)),
            model.dam.register_forward_pre_hook(lambda m, a: seen.update(dam_input=a[0])),
            model.dam.register_forward_hook(lambda m, a, out: seen.update(dam=out)),
            model.pool.register_forward_pre_hook(lambda m, a: seen.update(pool=a[0])),
            model.ffn.register_forward_pre_hook(lambda m, a: seen.update(ffn=a[0])),
        ]
        try:
            with torch.no_grad():
                output = model(torch.randn(2, 1, 8, 5, 7))
        finally:
            for handle in handles:
                handle.remove()
        torch.testing.assert_close(seen['dam_input'], seen['dmc'] + seen['psa'], rtol=0, atol=0)
        torch.testing.assert_close(seen['pool'], seen['dam_input'] + seen['dam'], rtol=0, atol=0)
        torch.testing.assert_close(seen['ffn'], seen['pool'].mean((2, 3)))
        self.assertEqual(tuple(output.shape), (2, 3))
        self.assertEqual(sum(isinstance(m, ZWKDAM) for m in model.modules()), 1)
        self.assertEqual(tuple(seen['dam'].shape), tuple(seen['dam_input'].shape))
        self.assertEqual(seen['dam'].dtype, seen['dam_input'].dtype)

    def test_dam_switch_preserves_initialization_rng_and_disabled_logits(self):
        cls = self.classifier()
        for psa_enabled in (False, True):
            baseline_options = {k: v for k, v in self.options().items() if not k.startswith('dam_')}
            torch.manual_seed(31)
            baseline = ZWKDMCPSA(**baseline_options, psa_enabled=psa_enabled).eval()
            rng = torch.get_rng_state().clone()
            for enabled in (False, True):
                torch.manual_seed(31)
                model = cls(**self.options(), psa_enabled=psa_enabled, dam_enabled=enabled).eval()
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                for name, value in baseline.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
                if not enabled:
                    model.load_state_dict(baseline.state_dict(), strict=True)
                    self.assertEqual(model.no_weight_decay(), set())
                    with torch.no_grad():
                        features = torch.randn(2, 8, 5, 7)
                        torch.testing.assert_close(model(features), baseline(features), rtol=0, atol=0)

    def test_zero_dam_output_reproduces_existing_psa_model(self):
        torch.manual_seed(4)
        base_options = {k: v for k, v in self.options().items() if not k.startswith('dam_')}
        baseline = ZWKDMCPSA(**base_options).eval()
        torch.manual_seed(4)
        model = self.classifier()(**self.options()).eval()
        with torch.no_grad():
            model.dam.out_proj.weight.zero_()
            features = torch.randn(2, 1, 8, 5, 7)
            torch.testing.assert_close(model(features), baseline(features), rtol=0, atol=0)

    def test_gradients_reach_directions_and_checkpoint_loads_after_cache_changes(self):
        torch.manual_seed(9)
        cls = self.classifier()
        model = cls(**self.options())
        features = torch.randn(2, 1, 8, 5, 7, requires_grad=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        before = model.dam.direction_table.detach().clone()
        torch.nn.functional.cross_entropy(model(features), torch.tensor([0, 2])).backward()
        for name, parameter in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
                self.assertGreater(parameter.grad.abs().sum().item(), 0.)
        self.assertTrue(torch.isfinite(features.grad).all())
        optimizer.step()
        self.assertFalse(torch.equal(before, model.dam.direction_table))
        model.eval()
        with torch.no_grad():
            model(torch.randn(1, 1, 8, 3, 5))
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = cls(**self.options()).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(model(features), restored(features), rtol=0, atol=0)

    def test_configs_factory_optimizer_groups_and_flops_work_for_ip_and_up(self):
        from models.get_model import get_model
        from utils.scheduler import load_scheduler
        root = Path(__file__).resolve().parents[1]
        for dataset, bands, classes, params in (('ip', 200, 16, 117475), ('up', 103, 9, 110812)):
            with self.subTest(dataset=dataset):
                config = json.loads((root / f'config/zwk_dmc_psa_dam_{dataset}.json').read_text())
                baseline_config = json.loads((root / f'config/zwk_dmc_psa_{dataset}.json').read_text())
                self.assertEqual({k: v for k, v in config['model'].items() if not k.startswith('dam_')}, baseline_config['model'])
                self.assertEqual(config['schedule'], baseline_config['schedule'])
                model = get_model(config['model'], 'zwk_dmc_psa_dam', dataset, 15)
                self.assertIsInstance(model, self.classifier())
                self.assertEqual(sum(p.numel() for p in model.parameters()), params)
                state, rng = copy.deepcopy(model.state_dict()), torch.get_rng_state().clone()
                flops = model.flops(shape=(1, bands, 15, 15), verbose=False)
                torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
                for name, value in model.state_dict().items():
                    torch.testing.assert_close(value, state[name], rtol=0, atol=0)
                self.assertIsNone(model.dam._cached_shape)
                self.assertGreater(flops, ZWKDMCPSA(**baseline_config['model']).flops(verbose=False))
                optimizer, scheduler = load_scheduler(config['schedule'], 'zwk_dmc_psa_dam', model)
                excluded = {'dam.A_log', 'dam.D_skip'}
                self.assertEqual(model.no_weight_decay(), excluded)
                all_ids = [id(p) for group in optimizer.param_groups for p in group['params']]
                self.assertEqual(len(all_ids), len(set(all_ids)))
                self.assertEqual(set(all_ids), {id(p) for p in model.parameters()})
                for name, parameter in model.named_parameters():
                    group = next(g for g in optimizer.param_groups if any(p is parameter for p in g['params']))
                    self.assertEqual(group['weight_decay'], 0. if name in excluded else .01)
                output = model(torch.randn(2, 1, bands, 3, 5))
                self.assertEqual(tuple(output.shape), (2, classes))
                torch.nn.functional.cross_entropy(output, torch.tensor([0, classes - 1])).backward()
                optimizer.step()
                scheduler.step()

    def test_deployment_and_deployed_checkpoint_preserve_full_model(self):
        cls = self.classifier()
        model = cls(**self.options()).double().eval()
        features = torch.randn(2, 1, 8, 3, 5, dtype=torch.float64)
        with torch.no_grad():
            expected = model(features)
            model.switch_to_deploy()
            torch.testing.assert_close(model(features), expected, rtol=1e-10, atol=1e-10)
        restored = cls(**self.options(), deploy=True).double().eval()
        restored.load_state_dict(model.state_dict(), strict=True)
        with torch.no_grad():
            torch.testing.assert_close(restored(features), model(features), rtol=0, atol=0)

    def test_invalid_dam_options_are_rejected(self):
        for options in ({'dam_enabled': 'true'}, {'dam_d_state': 0},
                        {'dam_ssm_ratio': 0.}, {'dam_direction_enabled': 'false'},
                        {'dam_scan_backend': 'unknown'}):
            settings = dict(self.options(), **options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.classifier()(**settings)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_matches_reference_and_amp_training_uses_accelerated_scan(self):
        cls = self.classifier()
        torch.manual_seed(8)
        reference = cls(**self.options(), dam_scan_backend='reference').cuda().eval()
        accelerated = cls(**self.options(), dam_scan_backend='cuda').cuda().eval()
        accelerated.load_state_dict(reference.state_dict(), strict=True)
        features = torch.randn(2, 1, 8, 5, 7, device='cuda')
        with torch.no_grad():
            torch.testing.assert_close(accelerated(features), reference(features), rtol=2e-4, atol=2e-5)
        self.assertIsNotNone(accelerated.dam._cuda_scan)
        accelerated.train()
        for dtype in (torch.float16, torch.bfloat16):
            accelerated.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=dtype):
                output = accelerated(features)
                loss = torch.nn.functional.cross_entropy(output, torch.tensor([0, 2], device='cuda'))
            loss.backward()
            for name, parameter in accelerated.named_parameters():
                with self.subTest(dtype=dtype, parameter=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
