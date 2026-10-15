"""No-DPE ablation: identical retained weights, protocol and bypassed data flow."""
import importlib
import io
import json
from pathlib import Path
import sys
import unittest
import torch

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from models import DPE_Mamba_newCFAM as reference
from models.get_model import get_model

torch.set_num_threads(2)

class BypassDPE(torch.nn.Module):
    def forward(self, x, raw_patch=None):
        return x

class NoDPEContracts(unittest.TestCase):
    def setUp(self):
        self.assertTrue((PROJECT / 'models/newDPE_mamba_CFAM.py').is_file(),
                        'The independent no-DPE model has not been implemented')
        self.module = importlib.import_module('models.newDPE_mamba_CFAM')

    def config(self, model, dataset):
        return json.loads((PROJECT / f'config/{model}_{dataset}.json').read_text())

    def test_branch_defaults_match_fullossm_reference(self):
        from models import basic_spa_spe_mamba_fullossm_sharedcenter_cas2f_bfs as fullossm
        baseline = fullossm.BasicSpaSpeMamba()
        model = self.module.BasicSpaSpeMamba()
        for name in ('spatial', 'spectral'):
            old = getattr(baseline.blocks[0], name)
            new = getattr(model.blocks[0], name)
            self.assertEqual(set(old.state_dict()), set(new.state_dict()))
            new.load_state_dict(old.state_dict(), strict=True)
            if name == 'spectral' or torch.cuda.is_available():
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                old = old.to(device).eval()
                new = new.to(device).eval()
                x = torch.randn(2, 7, 7, 64, device=device)
                with torch.no_grad():
                    torch.testing.assert_close(new(x), old(x), rtol=1e-5, atol=1e-7)
        self.assertEqual(model.blocks[0].spatial.mamba.disable_z,
                         baseline.blocks[0].spatial.mamba.disable_z)
        self.assertEqual(model.blocks[0].spatial.mamba.d_conv,
                         baseline.blocks[0].spatial.mamba.d_conv)
        self.assertEqual(model.blocks[0].spectral.mamba.bfs_enabled,
                         baseline.blocks[0].spectral.mamba.bfs_enabled)

    def test_retained_initial_weights_and_rng_match_dpe_reference(self):
        for dataset in ('ip', 'up'):
            for seed in (1, 34, 20031025, 9, 10):
                with self.subTest(dataset=dataset, seed=seed):
                    config = self.config('mamba_no_dpe', dataset)['model']
                    torch.manual_seed(seed)
                    old = reference.BasicSpaSpeMamba(**config)
                    rng = torch.get_rng_state().clone()
                    torch.manual_seed(seed)
                    new = self.module.BasicSpaSpeMamba(**config)
                    self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                    expected = {k: v for k, v in old.state_dict().items()
                                if not k.startswith('blocks.0.dpe.')}
                    self.assertEqual(set(new.state_dict()), set(expected))
                    for name, value in expected.items():
                        torch.testing.assert_close(new.state_dict()[name], value, rtol=0, atol=0, msg=name)
                    self.assertEqual(sum(p.numel() for p in old.parameters()) -
                                     sum(p.numel() for p in new.parameters()), 34120)

    def test_factory_and_configs_preserve_baseline_and_schedule(self):
        from utils.scheduler import load_scheduler
        for ds in ('ip', 'up'):
            old_config = self.config('basic_spa_spe_mamba', ds)
            new_config = self.config('mamba_no_dpe', ds)
            expected = self.config('basic_spa_spe_mamba', ds)
            expected['model']['spatial_forward_type'] = 'v3'
            expected['model']['spatial_ssm_conv'] = 3
            self.assertEqual(expected, new_config)
            old = get_model(old_config['model'], 'basic_spa_spe_mamba', ds, 7)
            new = get_model(new_config['model'], 'mamba_no_dpe', ds, 7)
            self.assertIsInstance(old, reference.BasicSpaSpeMamba)
            self.assertIsInstance(new, self.module.BasicSpaSpeMamba)
            self.assertIsInstance(new.blocks[0].cross_fusion, self.module.MeanConcatFusion)
            self.assertFalse(hasattr(new.blocks[0], 'dpe'))
            opt, sched = load_scheduler(new_config['schedule'], 'mamba_no_dpe', new)
            self.assertIsInstance(opt, torch.optim.AdamW)
            self.assertEqual(opt.param_groups[0]['lr'], .001)
            self.assertIsInstance(sched, torch.optim.lr_scheduler.StepLR)
            self.assertEqual(sched.step_size, 30)
            self.assertEqual(sched.gamma, .9)

    @unittest.skipUnless(torch.cuda.is_available(), 'Spatial OSSM requires CUDA')
    def test_training_presets_execute_spatial_conv_and_z_gate(self):
        for ds in ('ip', 'up'):
            config = self.config('mamba_no_dpe', ds)['model']
            model = get_model(config, 'mamba_no_dpe', ds, 7).cuda().eval()
            spa = model.blocks[0].spatial
            self.assertFalse(spa.mamba.disable_z, 'Training config still disables the z gate')
            self.assertEqual(spa.mamba.d_conv, 3)
            self.assertEqual(spa.mamba.conv2d.groups, 128)
            self.assertFalse(model.blocks[0].spectral.mamba.bfs_enabled)
            calls = []
            hook = spa.mamba.conv2d.register_forward_hook(lambda m, a, out: calls.append(out.shape))
            x = torch.randn(2, 7, 7, 64, device='cuda')
            with torch.no_grad():
                normal = spa(x)
            hook.remove()
            self.assertEqual(calls, [torch.Size([2, 128, 7, 7])])
            self.assertGreater(normal.abs().sum().item(), 0)
            def zero_gate(module, args, out):
                content, gate = out.chunk(2, dim=-1)
                return torch.cat((content, torch.zeros_like(gate)), dim=-1)
            hook = spa.mamba.in_proj.register_forward_hook(zero_gate)
            with torch.no_grad():
                muted = spa(x)
            hook.remove()
            torch.testing.assert_close(muted, torch.zeros_like(muted), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'Spatial OSSM requires CUDA')
    def test_logits_match_reference_with_dpe_bypassed(self):
        for ds in ('ip', 'up'):
            config = self.config('mamba_no_dpe', ds)['model']
            torch.manual_seed(1)
            old = reference.BasicSpaSpeMamba(**config).cuda().eval()
            torch.manual_seed(1)
            new = self.module.BasicSpaSpeMamba(**config).cuda().eval()
            old.blocks[0].dpe = BypassDPE()
            x = torch.randn(3, 1, config['in_chans'], 7, 7, device='cuda')
            with torch.no_grad():
                torch.testing.assert_close(new(x), old(x), rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), 'Spatial OSSM requires CUDA')
    def test_direct_embedding_to_branches_backward_and_checkpoint(self):
        for ds in ('ip', 'up'):
            config = self.config('mamba_no_dpe', ds)['model']
            model = get_model(config, 'mamba_no_dpe', ds, 7).cuda().train()
            x = torch.randn(2, 1, config['in_chans'], 7, 7, device='cuda')
            seen = {}
            hooks = [model.patch_embedding.register_forward_hook(
                lambda m, args, out: seen.update(embedding=out.detach().permute(0, 2, 3, 1))),
                model.blocks[0].spatial.register_forward_pre_hook(
                    lambda m, args: seen.update(spatial=args[0].detach())),
                model.blocks[0].spectral.register_forward_pre_hook(
                    lambda m, args: seen.update(spectral=args[0].detach()))]
            output = model(x)
            for h in hooks: h.remove()
            for name in ('spatial', 'spectral'):
                torch.testing.assert_close(seen[name], seen['embedding'], rtol=0, atol=0)
            self.assertEqual(output.shape, (2, config['num_classes']))
            torch.nn.functional.cross_entropy(output, torch.tensor([0, 1], device='cuda')).backward()
            for name, p in model.named_parameters():
                self.assertIsNotNone(p.grad, name)
                self.assertTrue(torch.isfinite(p.grad).all(), name)
            model.eval()
            buf = io.BytesIO(); torch.save(model.state_dict(), buf); buf.seek(0)
            restored = self.module.BasicSpaSpeMamba(**config).cuda().eval()
            restored.load_state_dict(torch.load(buf, map_location='cuda'), strict=True)
            with torch.no_grad():
                torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)

if __name__ == '__main__':
    unittest.main()
