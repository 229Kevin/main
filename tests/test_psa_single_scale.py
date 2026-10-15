"""Single-scale PSA ablation: 7x7 only, preserving the retained model."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import unittest
import torch
from einops import rearrange

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from models import newDPE_mamba_CFAM as current
from models.get_model import get_model
torch.set_num_threads(2)

class SingleScalePSAContracts(unittest.TestCase):
    def config(self, ds):
        return json.loads((PROJECT/f'config/mamba_shared_psa_{ds}.json').read_text())['model']

    def test_factory_has_only_one_scale_and_no_reduced_resolution_attention(self):
        for ds in ('ip','up'):
            model=get_model(self.config(ds),'mamba_shared_psa',ds,7)
            psa=model.blocks[0].psa
            self.assertEqual(len(psa.attention_modules),1)
            self.assertEqual(len(psa.attention_layers),1)
            self.assertEqual(len(psa.attention_layers[0]),2)
            self.assertEqual(psa.project_out.in_channels,64)
            self.assertEqual(sum(p.numel() for p in psa.parameters()),59856)
            sizes=[]
            hooks=[m.register_forward_pre_hook(lambda module,args:sizes.append(tuple(args[0].shape[-2:])))
                   for m in psa.modules() if isinstance(m,current.PyramidAttention)]
            out=psa(torch.randn(2,64,7,7))
            for h in hooks:h.remove()
            self.assertEqual(sizes,[(7,7),(7,7),(7,7)])
            self.assertEqual(out.shape,(2,64,7,7))

    def test_removed_multiscale_option_is_rejected(self):
        with self.assertRaises(ValueError):
            current.PyramidRefinedChannelAttention(64,4,True,num_scales=3)

    def test_single_scale_matches_original_psa_with_one_scale(self):
        source=Path('/home/zwk/CVPR/HSI/HyPyraMamba/HypraMamba/model.py')
        tree=ast.parse(source.read_text())
        classes=[n for n in tree.body if isinstance(n,ast.ClassDef) and n.name in
                 ('PyramidAttention','PyramidRefinedChannelAttention')]
        scope={'nn':torch.nn,'torch':torch,'rearrange':rearrange}
        exec(compile(ast.Module(body=classes,type_ignores=[]),str(source),'exec'),scope)
        old=scope['PyramidRefinedChannelAttention'](64,4,True,1,2).double()
        new=current.PyramidRefinedChannelAttention(64,4,True).double()
        new.load_state_dict(old.state_dict(),strict=True)
        x=torch.randn(2,64,7,7,dtype=torch.float64,requires_grad=True)
        xx=x.detach().clone().requires_grad_()
        a,b=old(x),new(xx)
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        ga=torch.autograd.grad(a.square().mean(),(x,*old.parameters()),allow_unused=True)
        gb=torch.autograd.grad(b.square().mean(),(xx,*new.parameters()),allow_unused=True)
        for a,b in zip(ga,gb):
            if a is None:self.assertIsNone(b)
            else:torch.testing.assert_close(a,b,rtol=0,atol=0)

    def test_retained_initial_weights_match_three_scale_backup(self):
        backup=Path('/home/zwk/cfam_experiments_20261003/before_psa_single_scale_20261012_205157/models/newDPE_mamba_CFAM.py')
        spec=importlib.util.spec_from_file_location('models._psa_three_scale_reference',backup)
        previous=importlib.util.module_from_spec(spec);spec.loader.exec_module(previous)
        for ds in ('ip','up'):
            for seed in (1,34):
                config=self.config(ds)
                torch.manual_seed(seed);old=previous.BasicSpaSpeMamba(**config);rng=torch.get_rng_state()
                torch.manual_seed(seed);new=current.BasicSpaSpeMamba(**config)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                for name,value in old.state_dict().items():
                    if not name.startswith('blocks.0.psa.'):
                        torch.testing.assert_close(new.state_dict()[name],value,rtol=0,atol=0,msg=name)

if __name__=='__main__':unittest.main()
