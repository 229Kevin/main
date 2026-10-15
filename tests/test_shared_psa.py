"""Shared PSA: original-source equivalence and branch integration."""
import ast
import importlib
import io
import json
from pathlib import Path
import sys
import unittest
import torch
from einops import rearrange

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from models.get_model import get_model
SOURCE = Path('/home/zwk/CVPR/HSI/HyPyraMamba/HypraMamba/model.py')
torch.set_num_threads(2)

class SharedPSAContracts(unittest.TestCase):
    def setUp(self):
        self.module = importlib.import_module('models.newDPE_mamba_CFAM')
        self.assertTrue(hasattr(self.module, 'PyramidRefinedChannelAttention'),
                        'The original HyPyraMamba PSA has not been implemented')

    def config(self, model, ds):
        return json.loads((PROJECT / f'config/{model}_{ds}.json').read_text())

    def test_original_source_forward_and_gradients_match(self):
        # Execute only the two PSA classes from the original source as the oracle.
        tree = ast.parse(SOURCE.read_text())
        classes = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in
                   ('PyramidAttention', 'PyramidRefinedChannelAttention')]
        scope = {'nn': torch.nn, 'torch': torch, 'rearrange': rearrange}
        exec(compile(ast.Module(body=classes, type_ignores=[]), str(SOURCE), 'exec'), scope)
        torch.manual_seed(34)
        old = scope['PyramidRefinedChannelAttention'](64, 4, True, 1, 2).double()
        new = self.module.PyramidRefinedChannelAttention(64, 4, True, 1, 2).double()
        new.load_state_dict(old.state_dict(), strict=True)
        x = torch.randn(2, 64, 7, 7, dtype=torch.float64, requires_grad=True)
        xx = x.detach().clone().requires_grad_()
        y, yy = old(x), new(xx)
        torch.testing.assert_close(yy, y, rtol=0, atol=0)
        ga = torch.autograd.grad(y.square().mean(), (x, *old.parameters()), allow_unused=True)
        gb = torch.autograd.grad(yy.square().mean(), (xx, *new.parameters()), allow_unused=True)
        for a, b in zip(ga, gb):
            if a is None:
                self.assertIsNone(b)
            else:
                torch.testing.assert_close(b, a, rtol=0, atol=0)
                self.assertTrue(torch.isfinite(b).all())

    def test_psa_preserves_retained_weights_rng_and_training_schedule(self):
        for ds in ('ip', 'up'):
            off = self.config('mamba_no_dpe', ds)
            on = self.config('mamba_shared_psa', ds)
            expected = self.config('mamba_no_dpe', ds)
            expected['model']['psa_enabled'] = True
            self.assertEqual(expected, on)
            for seed in (1, 34, 20031025):
                torch.manual_seed(seed)
                baseline = get_model(off['model'], 'mamba_no_dpe', ds, 7)
                rng = torch.get_rng_state().clone()
                torch.manual_seed(seed)
                model = get_model(on['model'], 'mamba_shared_psa', ds, 7)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                for name, value in baseline.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[name], value, rtol=0, atol=0)
                self.assertEqual(sum(p.numel() for p in model.parameters()) -
                                 sum(p.numel() for p in baseline.parameters()), 179432)
                self.assertFalse(hasattr(model.blocks[0], 'dpe'))

    def test_single_scale_psa_accepts_small_patch_without_pooling(self):
        config = self.config('mamba_shared_psa', 'ip')['model']
        config.update(psa_enabled=True, patch_size=3)
        model = self.module.BasicSpaSpeMamba(**config)
        output = model.blocks[0].psa(torch.randn(2, 64, 3, 3))
        self.assertEqual(output.shape, (2, 64, 3, 3))

    @unittest.skipUnless(torch.cuda.is_available(), 'Spatial OSSM requires CUDA')
    def test_one_shared_psa_feeds_both_branches_and_keeps_embedding_in_fusion(self):
        for ds in ('ip', 'up'):
            config = self.config('mamba_shared_psa', ds)['model']
            model = get_model(config, 'mamba_shared_psa', ds, 7).cuda().train()
            block = model.blocks[0]; seen = {}; calls = []
            hooks = [model.patch_embedding.register_forward_hook(
                lambda m, a, o: seen.update(embedding=o.detach())),
                block.psa.register_forward_hook(
                    lambda m, a, o: (calls.append(1), seen.update(psa_input=a[0].detach(), psa_output=o.detach())) and None),
                block.spatial.register_forward_pre_hook(lambda m, a: seen.update(spa=a[0].detach())),
                block.spectral.register_forward_pre_hook(lambda m, a: seen.update(spe=a[0].detach())),
                block.cross_fusion.register_forward_pre_hook(
                    lambda m, a, kw: seen.update(fusion_x=kw['x_residual'].detach()), with_kwargs=True)]
            x = torch.randn(2, 1, config['in_chans'], 7, 7, device='cuda')
            out = model(x)
            for hook in hooks: hook.remove()
            self.assertEqual(len(calls), 1)
            torch.testing.assert_close(seen['psa_input'], seen['embedding'], rtol=0, atol=0)
            for name in ('spa', 'spe'):
                torch.testing.assert_close(seen[name], seen['psa_output'].permute(0, 2, 3, 1), rtol=0, atol=0)
            torch.testing.assert_close(seen['fusion_x'], seen['embedding'].permute(0, 2, 3, 1), rtol=0, atol=0)
            torch.nn.functional.cross_entropy(out, torch.tensor([0, 1], device='cuda')).backward()
            for name, p in model.named_parameters():
                # This outer temperature is unused by the original PSA source.
                if name == 'blocks.0.psa.temperature':
                    self.assertIsNone(p.grad)
                else:
                    self.assertIsNotNone(p.grad, name)
                    self.assertTrue(torch.isfinite(p.grad).all(), name)
            model.eval(); buf = io.BytesIO(); torch.save(model.state_dict(), buf); buf.seek(0)
            restored = self.module.BasicSpaSpeMamba(**config).cuda().eval()
            restored.load_state_dict(torch.load(buf, map_location='cuda'), strict=True)
            with torch.no_grad():
                torch.testing.assert_close(restored(x), model(x), rtol=1e-5, atol=1e-7)

if __name__ == '__main__':
    unittest.main()
