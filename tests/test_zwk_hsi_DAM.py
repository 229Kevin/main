"""Contracts for parallel fusion and direct serial PSA/DAM in the classifier."""

import io
import json
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models.zwk_hsi import ZWKHSI


class ParallelDAMTests(unittest.TestCase):
    def test_serial_sum_preserves_serial_flow_and_adds_each_output_once(self):
        for psa_enabled,dam_enabled in ((True,True),(True,False),(False,True),(False,False)):
            with self.subTest(psa=psa_enabled,dam=dam_enabled):
                model = ZWKHSI(num_bands=8,num_classes=3,psa_enabled=psa_enabled,
                               dam_enabled=dam_enabled,fusion_mode='serial_sum').eval()
                captured = {}
                handles = [module.register_forward_hook(
                    lambda module,inputs,output,name=name: captured.update({name:output}))
                    for name,module in (('cnn',model.cnn),('psa',model.psa),('dam',model.dam))]
                handles.append(model.dam.register_forward_pre_hook(
                    lambda module,inputs: captured.update(dam_input=inputs[0])))
                handles.append(model.pool.register_forward_pre_hook(
                    lambda module,inputs: captured.update(pool_input=inputs[0])))
                try:
                    with torch.no_grad():
                        logits = model(torch.randn(2,1,8,15,15))
                finally:
                    for handle in handles:handle.remove()
                expected = captured['cnn']
                if psa_enabled: expected = expected + captured['psa']
                if dam_enabled:
                    self.assertIs(captured['dam_input'],captured['psa'] if psa_enabled else captured['cnn'])
                    expected = expected + captured['dam']
                torch.testing.assert_close(captured['pool_input'],expected,rtol=0,atol=0)
                self.assertEqual(tuple(logits.shape),(2,3))
                self.assertIsNone(model.psa_residual_scale)
                self.assertIsNone(model.dam_residual_scale)

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_serial_sum_cuda_training_and_checkpoint(self):
        torch.manual_seed(7)
        model = ZWKHSI(num_bands=8,num_classes=3,dam_enabled=True,fusion_mode='serial_sum').cuda()
        optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
        features = torch.randn(2,1,8,15,15,device='cuda',requires_grad=True)
        with torch.autocast('cuda',dtype=torch.float16):
            logits = model(features)
            loss = torch.nn.functional.cross_entropy(logits,torch.tensor([0,2],device='cuda'))
        loss.backward()
        for name,p in model.named_parameters():
            if p.requires_grad:
                with self.subTest(parameter=name):
                    self.assertIsNotNone(p.grad)
                    self.assertTrue(torch.isfinite(p.grad).all())
                    self.assertGreater(p.grad.abs().sum().item(),0.)
        self.assertGreater(features.grad.abs().sum().item(),0.)
        optimizer.step()
        buffer = io.BytesIO()
        torch.save(model.state_dict(),buffer)
        buffer.seek(0)
        restored = ZWKHSI(num_bands=8,num_classes=3,dam_enabled=True,fusion_mode='serial_sum').cuda().eval()
        restored.load_state_dict(torch.load(buffer,map_location='cuda',weights_only=True))
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(features),model(features),rtol=0,atol=0)
        self.assertGreater(restored.flops(shape=(1,8,15,15),verbose=False),0.)

    def test_serial_dam_receives_psa_output_and_head_receives_only_dam(self):
        model = ZWKHSI(num_bands=8, num_classes=3, dam_enabled=True,
                       fusion_mode='serial').eval()
        captured = {}
        hooks = [
            model.psa.register_forward_hook(
                lambda module, inputs, output: captured.update(psa=output)),
            model.dam.register_forward_pre_hook(
                lambda module, inputs: captured.update(dam_input=inputs[0])),
            model.dam.register_forward_hook(
                lambda module, inputs, output: captured.update(dam=output)),
            model.pool.register_forward_pre_hook(
                lambda module, inputs: captured.update(pool_input=inputs[0])),
            model.classifier.register_forward_pre_hook(
                lambda module, inputs: captured.update(head_input=inputs[0])),
        ]
        try:
            with torch.no_grad():
                logits = model(torch.randn(2, 1, 8, 15, 15))
        finally:
            for hook in hooks:
                hook.remove()
        self.assertIs(captured['dam_input'], captured['psa'])
        self.assertIs(captured['pool_input'], captured['dam'])
        self.assertEqual(tuple(captured['dam'].shape), (2, 64, 15, 15))
        torch.testing.assert_close(captured['head_input'], captured['dam'].mean((2,3)),
                                   rtol=1e-6, atol=1e-7)
        self.assertEqual(tuple(logits.shape), (2, 3))
        self.assertIsNone(model.psa_residual_scale)
        self.assertIsNone(model.dam_residual_scale)

    def test_serial_has_same_layers_and_ignores_residual_scale_settings(self):
        torch.manual_seed(23)
        serial = ZWKHSI(num_bands=8, num_classes=3, dam_enabled=True,
                        fusion_mode='serial',psa_residual_scale=0.,dam_residual_scale=0.).eval()
        torch.manual_seed(23)
        different_scales = ZWKHSI(num_bands=8, num_classes=3, dam_enabled=True,
                                  fusion_mode='serial',psa_residual_scale=8.,dam_residual_scale=9.).eval()
        torch.manual_seed(23)
        parallel = ZWKHSI(num_bands=8, num_classes=3, dam_enabled=True).eval()
        for name,value in serial.state_dict().items():
            torch.testing.assert_close(value,parallel.state_dict()[name],rtol=0,atol=0)
        convolutions = lambda model: {n:tuple(m.weight.shape) for n,m in model.named_modules()
                                     if isinstance(m,(torch.nn.Conv2d,torch.nn.Conv3d))}
        self.assertEqual(convolutions(serial),convolutions(parallel))
        x = torch.randn(2,1,8,5,7)
        with torch.no_grad():
            torch.testing.assert_close(serial(x),different_scales(x),rtol=0,atol=0)
        with self.assertRaises(ValueError):
            ZWKHSI(num_bands=8,num_classes=3,fusion_mode='wrong')

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_serial_cuda_training_autocast_and_checkpoint_roundtrip(self):
        for dtype in (torch.float16,torch.bfloat16):
            with self.subTest(dtype=dtype):
                model = ZWKHSI(num_bands=8,num_classes=3,dam_enabled=True,
                               fusion_mode='serial').cuda().train()
                optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
                features = torch.randn(2,1,8,15,15,device='cuda',requires_grad=True)
                with torch.autocast('cuda',dtype=dtype):
                    logits = model(features)
                    loss = torch.nn.functional.cross_entropy(logits,torch.tensor([0,2],device='cuda'))
                loss.backward()
                self.assertTrue(torch.isfinite(features.grad).all())
                for name,p in model.named_parameters():
                    if p.requires_grad:
                        with self.subTest(parameter=name):
                            self.assertIsNotNone(p.grad)
                            self.assertTrue(torch.isfinite(p.grad).all())
                optimizer.step()
                buffer = io.BytesIO()
                torch.save(model.state_dict(),buffer)
                buffer.seek(0)
                restored = ZWKHSI(num_bands=8,num_classes=3,dam_enabled=True,
                                  fusion_mode='serial').cuda().eval()
                restored.load_state_dict(torch.load(buffer,map_location='cuda',weights_only=True))
                model.eval()
                with torch.no_grad():
                    torch.testing.assert_close(restored(features),model(features),rtol=0,atol=0)
                self.assertGreater(restored.flops(shape=(1,8,15,15),verbose=False),0.)

    def test_both_branches_receive_identical_cnn_features_and_fuse_once(self):
        model = ZWKHSI(num_bands=8, num_classes=3, dam_enabled=True).eval()
        captured = {}
        hooks = [
            model.cnn.register_forward_hook(
                lambda module, inputs, output: captured.update(cnn=output)),
            model.psa.register_forward_pre_hook(
                lambda module, inputs: captured.update(psa_input=inputs[0])),
            model.dam.register_forward_pre_hook(
                lambda module, inputs: captured.update(dam_input=inputs[0])),
            model.psa.register_forward_hook(
                lambda module, inputs, output: captured.update(psa=output)),
            model.dam.register_forward_hook(
                lambda module, inputs, output: captured.update(dam=output)),
            model.pool.register_forward_pre_hook(
                lambda module, inputs: captured.update(fused=inputs[0])),
        ]
        try:
            with torch.no_grad():
                logits = model(torch.randn(2, 1, 8, 15, 15))
        finally:
            for hook in hooks:
                hook.remove()
        self.assertIs(captured['psa_input'], captured['cnn'])
        self.assertIs(captured['dam_input'], captured['cnn'])
        for name in ('psa', 'dam', 'fused'):
            self.assertEqual(tuple(captured[name].shape), (2, 64, 15, 15))
            self.assertEqual(captured[name].dtype, captured['cnn'].dtype)
            self.assertEqual(captured[name].device, captured['cnn'].device)
        torch.testing.assert_close(captured['fused'], captured['cnn']
                                   + model.psa_residual_scale * captured['psa']
                                   + model.dam_residual_scale * captured['dam'],
                                   rtol=0, atol=0)
        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_zero_dam_scale_reproduces_baseline_and_preserves_initialization(self):
        torch.manual_seed(23)
        baseline = ZWKHSI(num_bands=8, num_classes=3, dam_enabled=False).eval()
        rng = torch.get_rng_state().clone()
        torch.manual_seed(23)
        parallel = ZWKHSI(num_bands=8, num_classes=3, dam_enabled=True,
                          dam_residual_scale=0.).eval()
        torch.testing.assert_close(torch.get_rng_state(), rng)
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(parallel.state_dict()[name], value, rtol=0, atol=0)
        base_convs = {n: (type(m), tuple(m.weight.shape)) for n,m in baseline.named_modules()
                      if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d))}
        retained_convs = {n: (type(m), tuple(m.weight.shape)) for n,m in parallel.named_modules()
                          if isinstance(m, (torch.nn.Conv2d, torch.nn.Conv3d))
                          and not n.startswith('dam.')}
        self.assertEqual(retained_convs, base_convs)
        for shape in ((2, 8, 15, 15), (2, 1, 8, 5, 7)):
            features = torch.randn(*shape)
            with torch.no_grad():
                torch.testing.assert_close(parallel(features), baseline(features), rtol=0, atol=0)

    def test_invalid_dam_switch_and_scale_are_rejected(self):
        for options in ({'dam_enabled': 'false'}, {'dam_residual_scale': float('nan')},
                        {'dam_residual_scale': float('inf')}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ZWKHSI(num_bands=8, num_classes=3, **options)

    def test_fusion_rejects_broadcastable_wrong_spatial_shape(self):
        model = ZWKHSI(num_bands=8, num_classes=3, dam_enabled=True).eval()
        model.dam = torch.nn.AdaptiveAvgPool2d(1)
        with self.assertRaisesRegex(RuntimeError, 'DAM'):
            model(torch.randn(2, 1, 8, 15, 15))

    def test_parallel_branches_adapt_to_nonstandard_channel_count(self):
        model = ZWKHSI(num_bands=8, num_classes=3, embed_dim=12, cnn_enabled=False,
                       dam_enabled=True).eval()
        self.assertEqual(model.dam.d_model, 12)
        features = torch.randn(2, 8, 5, 7)
        with torch.no_grad():
            logits = model(features)
        self.assertEqual(tuple(logits.shape), (2, 3))

    def test_cross_entropy_updates_both_branches_and_checkpoint_roundtrip(self):
        torch.manual_seed(7)
        model = ZWKHSI(num_bands=8, num_classes=3, dam_enabled=True)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        before = model.dam_residual_scale.detach().clone()
        before_direction = model.dam.direction_table.detach().clone()
        self.assertFalse(model.dam_residual_scale.requires_grad)
        self.assertFalse(any(parameter is model.dam_residual_scale
                             for group in optimizer.param_groups for parameter in group['params']))
        features = torch.randn(2, 1, 8, 15, 15, requires_grad=True)
        loss = torch.nn.functional.cross_entropy(model(features), torch.tensor([0, 2]))
        loss.backward()
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                with self.subTest(parameter=name):
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
                    self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertGreater(features.grad.abs().sum().item(), 0)
        optimizer.step()
        torch.testing.assert_close(model.dam_residual_scale, before, rtol=0, atol=0)
        self.assertFalse(torch.equal(before_direction, model.dam.direction_table))
        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        buffer.seek(0)
        restored = ZWKHSI(num_bands=8, num_classes=3, dam_enabled=True).eval()
        restored.load_state_dict(torch.load(buffer, weights_only=True))
        model.eval()
        with torch.no_grad():
            torch.testing.assert_close(restored(features), model(features), rtol=0, atol=0)

    def test_factory_configs_include_dam_and_exclude_only_its_ssm_decay(self):
        from models.get_model import get_model
        from utils.scheduler import load_scheduler

        project = Path(__file__).resolve().parents[1]
        for dataset, classes in (('ip', 16), ('up', 9)):
            with self.subTest(dataset=dataset):
                config = json.loads((project / f'config/zwk_hsi_{dataset}.json').read_text())
                model = get_model(config['model'], 'zwk_hsi', dataset, 15)
                self.assertTrue(model.cnn_enabled and model.psa_enabled and model.dam_enabled)
                self.assertEqual(model.fusion_mode,config['model'].get('fusion_mode','parallel'))
                optimizer, _ = load_scheduler(config['schedule'], 'zwk_hsi', model)
                groups = {id(p): group['weight_decay'] for group in optimizer.param_groups
                          for p in group['params']}
                self.assertEqual(groups[id(model.dam.A_log)], 0.)
                self.assertEqual(groups[id(model.dam.D_skip)], 0.)
                self.assertEqual(groups[id(model.psa.qkv.weight)], 0.01)
                self.assertEqual(groups[id(model.dam.in_proj.weight)], 0.01)
                logits = model(torch.randn(2, 1, config['model']['num_bands'], 15, 15))
                self.assertEqual(tuple(logits.shape), (2, classes))
                loss = torch.nn.functional.cross_entropy(logits, torch.tensor([0, classes-1]))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                self.assertTrue(torch.isfinite(loss))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_autocast_preserves_fusion_shape_dtype_and_finite_gradients(self):
        model = ZWKHSI(num_bands=8, num_classes=3, dam_enabled=True).cuda()
        for dtype in (torch.float16, torch.bfloat16):
            with self.subTest(dtype=dtype):
                model.zero_grad(set_to_none=True)
                captures = {}
                hooks = [module.register_forward_hook(
                    lambda module, inputs, output, name=name: captures.update({name:output}))
                    for name,module in (('cnn',model.cnn),('psa',model.psa),('dam',model.dam))]
                hooks.append(model.pool.register_forward_pre_hook(
                    lambda module, inputs: captures.update(fused=inputs[0])))
                try:
                    with torch.autocast('cuda', dtype=dtype):
                        output = model(torch.randn(2, 1, 8, 15, 15, device='cuda'))
                        loss = torch.nn.functional.cross_entropy(output, torch.tensor([0, 2],device='cuda'))
                    loss.backward()
                finally:
                    for hook in hooks:
                        hook.remove()
                for name,value in captures.items():
                    with self.subTest(branch=name):
                        self.assertEqual(tuple(value.shape), (2, 64, 15, 15))
                        self.assertEqual(value.dtype, captures['cnn'].dtype)
                        self.assertTrue(value.is_cuda and torch.isfinite(value).all())
                for parameter in model.parameters():
                    if parameter.requires_grad:
                        self.assertTrue(torch.isfinite(parameter.grad).all())

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_cuda_flops_entry_runs_without_mutating_weights_or_rng(self):
        model = ZWKHSI(num_bands=8, num_classes=3, dam_enabled=True).cuda().train()
        before = {name: value.clone() for name,value in model.state_dict().items()}
        cpu_rng, cuda_rng = torch.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
        gflops = model.flops(shape=(1, 8, 15, 15), verbose=False)
        self.assertGreater(gflops, 0.)
        self.assertTrue(model.training)
        torch.testing.assert_close(torch.get_rng_state(), cpu_rng)
        torch.testing.assert_close(torch.cuda.get_rng_state(), cuda_rng)
        for name,value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
