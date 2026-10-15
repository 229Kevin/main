"""AEDC original-source equivalence and single-input spatial enhancement."""
import ast
import importlib
import io
import json
from pathlib import Path
import sys
import unittest
import torch

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT))
from models.get_model import get_model
torch.set_num_threads(2)

class AEDCContracts(unittest.TestCase):
    def setUp(self):
        self.module=importlib.import_module('models.newDPE_mamba_CFAM')
        self.assertTrue(hasattr(self.module,'DynamicConvBlock'),
                        'The original HyPyraMamba AEDC has not been implemented')

    def config(self, ds):
        return json.loads((PROJECT/f'config/mamba_shared_psa_aedc_{ds}.json').read_text())

    def test_original_source_output_dropout_and_gradients_match(self):
        path=Path('/home/zwk/CVPR/HSI/HyPyraMamba/HypraMamba/model.py')
        tree=ast.parse(path.read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='DynamicConvBlock')
        scope={'nn':torch.nn,'torch':torch}
        exec(compile(ast.Module(body=[cls],type_ignores=[]),str(path),'exec'),scope)
        old=scope['DynamicConvBlock'](64).double().train()
        new=self.module.DynamicConvBlock(64).double().train()
        new.load_state_dict(old.state_dict(),strict=True)
        x=torch.randn(2,64,7,7,dtype=torch.float64,requires_grad=True)
        xx=x.detach().clone().requires_grad_()
        torch.manual_seed(13);a=old(x)
        torch.manual_seed(13);b=new(xx)
        torch.testing.assert_close(b,a,rtol=0,atol=0)
        ga=torch.autograd.grad(a.square().mean(),(x,*old.parameters()))
        gb=torch.autograd.grad(b.square().mean(),(xx,*new.parameters()))
        for a,b in zip(ga,gb):
            torch.testing.assert_close(b,a,rtol=0,atol=0)
            self.assertTrue(torch.isfinite(b).all())
        self.assertEqual(sum(p.numel() for p in new.parameters()),3796)

    def test_configs_and_retained_initialization_are_preserved(self):
        for ds in ('ip','up'):
            old_config=json.loads((PROJECT/f'config/mamba_shared_psa_{ds}.json').read_text())
            new_config=self.config(ds)
            expected=json.loads((PROJECT/f'config/mamba_shared_psa_{ds}.json').read_text())
            expected['model']['aedc_enabled']=True
            expected['model']['fusion_type']='cas2f'
            self.assertEqual(expected,new_config)
            for seed in (1,34,20031025):
                off=dict(new_config['model'],aedc_enabled=False)
                torch.manual_seed(seed);baseline=self.module.BasicSpaSpeMamba(**off);rng=torch.get_rng_state()
                torch.manual_seed(seed);model=get_model(new_config['model'],'mamba_shared_psa_aedc',ds,7)
                self.assertTrue(torch.equal(rng,torch.get_rng_state()))
                for name,value in baseline.state_dict().items():
                    torch.testing.assert_close(model.state_dict()[name],value,rtol=0,atol=0,msg=name)
                self.assertEqual(sum(p.numel() for p in model.parameters())-
                                 sum(p.numel() for p in baseline.parameters()),3796)
                self.assertIsInstance(model.aedc,self.module.DynamicConvBlock)

    def test_vector_fusion_cannot_enable_spatial_aedc(self):
        config=self.config('ip')['model']
        config['fusion_type']='mean_concat'
        with self.assertRaises(ValueError):self.module.BasicSpaSpeMamba(**config)

    @unittest.skipUnless(torch.cuda.is_available(),'Spatial OSSM requires CUDA')
    def test_aedc_receives_block_map_before_norm_and_all_experts_receive_gradients(self):
        for ds in ('ip','up'):
            config=self.config(ds)['model']
            model=get_model(config,'mamba_shared_psa_aedc',ds,7).cuda().train()
            seen={}
            hooks=[model.blocks[0].register_forward_hook(lambda m,a,o:seen.update(block=o.detach())),
                   model.aedc.register_forward_hook(lambda m,a,o:seen.update(aedc_in=a[0].detach(),aedc_out=o.detach(),arg_count=len(a))),
                   model.norm.register_forward_pre_hook(lambda m,a:seen.update(norm_in=a[0].detach()))]
            x=torch.randn(2,1,config['in_chans'],7,7,device='cuda')
            out=model(x)
            for h in hooks:h.remove()
            self.assertEqual(seen['arg_count'],1)
            self.assertEqual(seen['aedc_in'].shape,(2,64,7,7))
            torch.testing.assert_close(seen['aedc_in'],seen['block'].permute(0,3,1,2),rtol=0,atol=0)
            torch.testing.assert_close(seen['norm_in'],seen['aedc_out'].permute(0,2,3,1),rtol=0,atol=0)
            self.assertEqual(out.shape,(2,config['num_classes']))
            torch.nn.functional.cross_entropy(out,torch.tensor([0,1],device='cuda')).backward()
            for name,p in model.aedc.named_parameters():
                self.assertIsNotNone(p.grad,name)
                self.assertTrue(torch.isfinite(p.grad).all(),name)
            for conv in model.aedc.convs:self.assertGreater(conv.weight.grad.abs().sum().item(),0)
            model.eval();buffer=io.BytesIO();torch.save(model.state_dict(),buffer);buffer.seek(0)
            restored=self.module.BasicSpaSpeMamba(**config).cuda().eval()
            restored.load_state_dict(torch.load(buffer,map_location='cuda'),strict=True)
            with torch.no_grad():torch.testing.assert_close(restored(x),model(x),rtol=1e-5,atol=1e-7)

if __name__=='__main__':unittest.main()
