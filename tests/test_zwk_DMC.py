"""Numerical and end-to-end contracts for the reconstructed DMC classifier."""

import copy
import importlib
import importlib.util
import io
import json
import math
from pathlib import Path
import sys
import unittest

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class DMCTests(unittest.TestCase):
    def module(self):
        self.assertIsNotNone(importlib.util.find_spec('models.zwk_DMC'),
                             'The reconstructed DMC module is missing')
        return importlib.import_module('models.zwk_DMC')

    def test_grouped_spatial_kernels_apply_to_the_correct_neighbors(self):
        apply = self.module().apply_grouped_dynamic_kernel
        features = torch.tensor([[[[1.,2.,3.],[4.,5.,6.]],
                                  [[10.,20.,30.],[40.,50.,60.]]]],dtype=torch.float64)
        weights = torch.zeros(1,2,6,9,dtype=torch.float64)
        weights[:,0,:,4] = 1.
        weights[:,1,:,5] = 1.
        weights[:,1,[2,5],5] = 0.
        weights[:,1,[2,5],4] = 1.
        expected = torch.tensor([[[[1.,2.,3.],[4.,5.,6.]],
                                  [[20.,30.,30.],[50.,60.,60.]]]],dtype=torch.float64)
        torch.testing.assert_close(apply(features,weights,3),expected,rtol=0,atol=0)

    def test_kernel_softmax_excludes_padding_and_preserves_constants(self):
        module = self.module()
        generator = module.GlobalKernelGenerator(4,groups=2,region_size=3,kernel_size=3)
        with torch.no_grad():
            for parameter in generator.parameters(): parameter.zero_()
        features = torch.ones(2,4,3,5)
        weights = generator(features)
        self.assertEqual(tuple(weights.shape),(2,2,15,9))
        torch.testing.assert_close(weights.sum(-1),torch.ones(2,2,15))
        self.assertEqual(torch.count_nonzero(weights[0,0,0]).item(),4)
        self.assertEqual(torch.count_nonzero(weights[0,0,6]).item(),9)
        torch.testing.assert_close(module.apply_grouped_dynamic_kernel(features,weights,3),features)

    def test_distant_region_changes_center_kernel_without_changing_local_patch(self):
        generator = self.module().GlobalKernelGenerator(4,groups=2,region_size=3,kernel_size=3)
        with torch.no_grad():
            for parameter in generator.parameters(): parameter.zero_()
            generator.value_projection.weight.copy_(torch.eye(4).reshape(4,4,1,1))
            generator.kernel_mlp[0].weight[0,2] = 1.
            generator.kernel_mlp[2].weight[4,0] = 1.
        first = torch.ones(1,4,15,15)
        second = first.clone()
        second[:,:,:3,:3] = 6.
        torch.testing.assert_close(first[:,:,6:9,6:9],second[:,:,6:9,6:9])
        difference = (generator(first)[:,:,112]-generator(second)[:,:,112]).abs().max()
        self.assertGreater(difference.item(),0.01)

    def test_cross_attention_reads_region_values_with_uniform_scores(self):
        # Catches omitting AV or normalizing over pixels instead of regions.
        generator = self.module().GlobalKernelGenerator(2,groups=2,region_size=2,kernel_size=3).double()
        self.assertIsNotNone(getattr(generator,'value_projection',None),
                             'Dynamic kernels must retrieve regional Value content')
        with torch.no_grad():
            for parameter in generator.parameters(): parameter.zero_()
            generator.value_projection.weight.copy_(torch.eye(2,dtype=torch.float64).reshape(2,2,1,1))
            generator.kernel_mlp[0].weight[0,1] = 1.
            generator.kernel_mlp[2].weight[4,0] = 1.
        features = torch.tensor([[[[0.,0.,0.],[0.,1.,2.],[0.,3.,4.]],
                                  [[2.,2.,2.],[2.,2.,2.],[2.,2.,2.]]]],dtype=torch.float64)
        weights = generator(features)
        # The first group's 2x2 pooled values are .25, .75, 1, 2.5.
        # Zero scores retrieve their mean 1.125; group two retrieves 2.
        for group,context in enumerate((1.125,2.)):
            gelu = context * .5 * (1. + math.erf(context / math.sqrt(2.)))
            center = math.exp(gelu)
            expected = torch.full((9,),1. / (8. + center),dtype=torch.float64)
            expected[4] = center / (8. + center)
            torch.testing.assert_close(weights[0,group,4],expected,rtol=1e-12,atol=1e-12)

    def test_scaled_query_key_scores_select_the_correct_group_context(self):
        # Catches missing sqrt(d) scaling, head mixing, or incorrect Q/K/V axes.
        generator = self.module().GlobalKernelGenerator(4,groups=2,region_size=2,kernel_size=3).double()
        self.assertIsNotNone(getattr(generator,'value_projection',None))
        with torch.no_grad():
            for parameter in generator.parameters(): parameter.zero_()
            for projection in (generator.query_projection,generator.key_projection,generator.value_projection):
                projection.weight.copy_(torch.eye(4,dtype=torch.float64).reshape(4,4,1,1))
            generator.kernel_mlp[0].weight[0,2] = 1.
            generator.kernel_mlp[2].weight[4,0] = 1.
        features = torch.zeros(1,4,2,2,dtype=torch.float64)
        features[0,0,1,1] = 2.
        features[0,2,1,1] = 1.
        weights = generator(features)
        for group,value in enumerate((2.,1.)):
            score_exp = math.exp(value * value / math.sqrt(2.))
            context = value * score_exp / (3. + score_exp)
            center = math.exp(context * .5 * (1. + math.erf(context / math.sqrt(2.))))
            expected = torch.zeros(9,dtype=torch.float64)
            expected[[0,1,3]] = 1. / (3. + center)
            expected[4] = center / (3. + center)
            torch.testing.assert_close(weights[0,group,3],expected,rtol=1e-12,atol=1e-12)

    def test_kernel_generator_conditions_on_raw_local_features(self):
        # With Q/K/V disabled, only concatenated local features can change a kernel.
        generator = self.module().GlobalKernelGenerator(2,groups=2,region_size=2,kernel_size=3)
        self.assertIsNotNone(getattr(generator,'kernel_mlp',None))
        with torch.no_grad():
            for parameter in generator.parameters(): parameter.zero_()
            generator.kernel_mlp[0].weight[0,0] = 1.
            generator.kernel_mlp[2].weight[4,0] = 1.
        first = torch.ones(1,2,3,3)
        second = first.clone()
        second[0,0,1,1] = 2.
        first_weights,second_weights = generator(first),generator(second)
        self.assertGreater((first_weights[0,0,4]-second_weights[0,0,4]).abs().max().item(),.01)
        torch.testing.assert_close(first_weights[0,1],second_weights[0,1],rtol=0,atol=0)

    def test_region_count_and_neighborhood_size_are_independent(self):
        module = self.module()
        for height,width,region_size,kernel_size in ((3,5,2,3),(1,1,2,5),(4,6,3,1)):
            with self.subTest(shape=(height,width),region_size=region_size,kernel_size=kernel_size):
                generator = module.GlobalKernelGenerator(4,groups=2,region_size=region_size,kernel_size=kernel_size)
                features = torch.ones(2,4,height,width)
                weights = generator(features)
                self.assertEqual(tuple(weights.shape),(2,2,height*width,kernel_size**2))
                torch.testing.assert_close(weights.sum(-1),torch.ones(2,2,height*width))
                torch.testing.assert_close(module.apply_grouped_dynamic_kernel(features,weights,kernel_size),features)

    def test_dynamic_kernel_input_gradients_match_finite_differences(self):
        torch.manual_seed(31)
        generator = self.module().GlobalKernelGenerator(4,groups=2,region_size=2,kernel_size=3).double()
        features = torch.randn(1,4,2,3,dtype=torch.float64,requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(generator,(features,),fast_mode=True))

    def test_repconv_deployment_matches_multibranch_eval(self):
        cls = self.module().RepConv
        for groups,dilation in ((1,1),(2,2)):
            with self.subTest(groups=groups,dilation=dilation):
                block = cls(4,groups=groups,dilation=dilation).double().train()
                for _ in range(3): block(torch.randn(2,4,5,7,dtype=torch.float64))
                block.eval()
                features = torch.randn(2,4,5,7,dtype=torch.float64)
                expected = block(features)
                block.switch_to_deploy()
                actual = block(features)
                torch.testing.assert_close(actual,expected,rtol=1e-10,atol=1e-10)
                self.assertEqual(sum(isinstance(m,torch.nn.Conv2d) for m in block.modules()),1)
        with self.assertRaises(RuntimeError): cls(4).train().switch_to_deploy()

    def test_classifier_shapes_and_four_five_dimensional_input_parity(self):
        model = self.module().ZWKDMC(num_bands=8,num_classes=3,feature_dim=8,groups=2).eval()
        for height,width in ((13,13),(15,15),(5,7)):
            features = torch.randn(2,8,height,width)
            with torch.no_grad():
                output = model(features)
                torch.testing.assert_close(output,model(features.unsqueeze(1)),rtol=0,atol=0)
            self.assertEqual(tuple(output.shape),(2,3))

    def test_cross_entropy_reaches_all_parameters_and_checkpoint_roundtrip(self):
        torch.manual_seed(7)
        cls = self.module().ZWKDMC
        model = cls(num_bands=8,num_classes=3,feature_dim=8,groups=2)
        features = torch.randn(2,1,8,15,15,requires_grad=True)
        optimizer = torch.optim.AdamW(model.parameters(),lr=0.001)
        before = model.dmc.kernel_generator.kernel_mlp[2].weight.detach().clone()
        loss = torch.nn.functional.cross_entropy(model(features),torch.tensor([0,2]))
        loss.backward()
        self.assertGreater(features.grad.abs().sum().item(),0.)
        for name,p in model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(p.grad)
                self.assertTrue(torch.isfinite(p.grad).all())
                self.assertGreater(p.grad.abs().sum().item(),0.)
        optimizer.step()
        self.assertFalse(torch.equal(before,model.dmc.kernel_generator.kernel_mlp[2].weight))
        buffer = io.BytesIO()
        torch.save(model.state_dict(),buffer)
        buffer.seek(0)
        restored = cls(num_bands=8,num_classes=3,feature_dim=8,groups=2).eval()
        restored.load_state_dict(torch.load(buffer,weights_only=True))
        model.eval()
        with torch.no_grad(): torch.testing.assert_close(model(features),restored(features),rtol=0,atol=0)

    def test_deployed_network_matches_training_architecture_in_eval(self):
        model = self.module().ZWKDMC(num_bands=8,num_classes=3,feature_dim=8,groups=2).eval()
        features = torch.randn(2,1,8,13,13)
        with torch.no_grad(): expected = model(features)
        model.switch_to_deploy()
        with torch.no_grad(): torch.testing.assert_close(model(features),expected,rtol=1e-5,atol=1e-6)

    def test_invalid_configuration_and_input_are_rejected(self):
        cls = self.module().ZWKDMC
        for options in ({'num_bands':0},{'feature_dim':7,'groups':4},{'groups':0},
                        {'region_size':0},{'kernel_size':2},{'local_dilation':0},
                        {'ffn_hidden_dim':0},{'dropout':1.}):
            with self.subTest(options=options),self.assertRaises(ValueError): cls(**options)
        model = cls(num_bands=8,num_classes=3,feature_dim=8,groups=2)
        for shape in ((2,8,3),(2,9,3,3),(2,2,8,3,3),(2,8,0,3)):
            with self.subTest(shape=shape),self.assertRaises(ValueError): model(torch.randn(*shape))

    def test_factory_config_and_flops_support_training_entry(self):
        from models.get_model import get_model
        from utils.scheduler import load_scheduler

        path = Path(__file__).resolve().parents[1]/'config/zwk_dmc_ip.json'
        config = json.loads(path.read_text())
        model = get_model(config['model'],'zwk_dmc','ip',15)
        self.assertIsInstance(model,self.module().ZWKDMC)
        before = {name:p.clone() for name,p in model.state_dict().items()}
        rng = torch.get_rng_state().clone()
        self.assertGreater(model.flops(shape=(1,200,15,15),verbose=False),0.)
        for name,p in model.state_dict().items(): torch.testing.assert_close(p,before[name],rtol=0,atol=0)
        torch.testing.assert_close(torch.get_rng_state(),rng)
        optimizer,scheduler = load_scheduler(config['schedule'],'zwk_dmc',model)
        logits = model(torch.randn(2,1,200,15,15))
        self.assertEqual(tuple(logits.shape),(2,16))
        torch.nn.functional.cross_entropy(logits,torch.tensor([0,15])).backward()
        optimizer.step()
        scheduler.step()

    @unittest.skipUnless(torch.cuda.is_available(),'CUDA unavailable')
    def test_cuda_mixed_precision_training_and_cpu_eval_parity(self):
        torch.manual_seed(9)
        model = self.module().ZWKDMC(num_bands=8,num_classes=3,feature_dim=8,groups=2).eval()
        gpu = copy.deepcopy(model).cuda()
        features = torch.randn(2,1,8,15,15)
        with torch.no_grad(): torch.testing.assert_close(gpu(features.cuda()).cpu(),model(features),rtol=2e-4,atol=2e-5)
        gpu.train()
        for dtype in (torch.float16,torch.bfloat16):
            gpu.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=dtype):
                logits = gpu(features.cuda())
                loss = torch.nn.functional.cross_entropy(logits,torch.tensor([0,2],device='cuda'))
            loss.backward()
            for p in gpu.parameters(): self.assertTrue(torch.isfinite(p.grad).all())


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
