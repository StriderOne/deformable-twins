import unittest
import numpy as np
import torch
from lab.calibration.residual_flow import SpringGraph, ResidualBlock, GraphResidualPosterior, spatial_edges


class FlowTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(12)
        self.graph=SpringGraph([[0,1],[1,2],[2,3],[4,5]],[[0,2],[1,3]]).double()
        self.settings=dict(lower=100.,upper=100000.,initial_std=1.5,blocks=2,width=4,
                           contraction=.9,trace_min_terms=2,trace_continuation=.95,spatial_neighbors=1)

    def test_graph_contraction_and_neighbors(self):
        matrix=self.graph(torch.eye(4,dtype=torch.double))
        self.assertLessEqual(float(torch.linalg.matrix_norm(matrix,2)),1.+1e-12)
        points=np.array([[0,0,0],[1,0,0],[2,0,0],[3,0,0],[0,1,0],[1,1,0]])
        edges=np.array([[0,1],[1,2],[2,3],[4,5]])
        pairs=spatial_edges(points,edges,1)
        self.assertEqual(len(pairs),len(set(map(tuple,pairs))))
        for a,b in pairs:
            self.assertLess(a,b)
            self.assertFalse(np.isin(edges[a],edges[b]).any())

    def test_inverse_and_jacobian(self):
        block=ResidualBlock(4,.9).double()
        with torch.no_grad():
            block.last.weight.mul_(100)
        x=torch.randn(4,dtype=torch.double,requires_grad=True)
        target=block(x,self.graph).detach()
        restored=target.clone()
        for _ in range(100):
            restored=target-block.residual(restored,self.graph)
        torch.testing.assert_close(restored,x)
        jac=torch.autograd.functional.jacobian(lambda z:block(z,self.graph),x,create_graph=True)
        exact=torch.linalg.slogdet(jac)[1]
        exact_grad=torch.autograd.grad(exact,block.last.weight,retain_graph=True)[0]
        estimates=[]; gradients=[]
        for _ in range(128):
            estimate=block.logdet(x,self.graph,2,.95)
            gradients.append(torch.autograd.grad(estimate,block.last.weight)[0].detach())
            estimates.append(estimate.detach())
        estimates=torch.stack(estimates)
        self.assertLess(float((estimates.mean()-exact).detach().abs()),float(6*estimates.std()/128**.5)+1e-8)
        gradients=torch.stack(gradients)
        self.assertTrue(torch.all((gradients.mean(0)-exact_grad).abs() < 6*gradients.std(0)/128**.5+1e-7))

    def test_support_gradients_rng_and_checkpoint(self):
        q=GraphResidualPosterior(torch.full((4,),8.,dtype=torch.double),self.settings,self.graph)
        epsilon=q.draw_noise()
        z,penalty=q.sample_and_penalty(epsilon)
        (z.square().sum()+penalty).backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in q.parameters()))
        self.assertTrue(((z.exp()>100)&(z.exp()<100000)).all())
        state=torch.get_rng_state().clone()
        q.std_ratio()
        self.assertTrue(torch.equal(state,torch.get_rng_state()))
        other=GraphResidualPosterior(q.mu.detach()*0+8,self.settings,self.graph)
        other.load_state_dict(q.state_dict())
        torch.testing.assert_close(q.sample(epsilon),other.sample(epsilon))

    def test_frame_update_resume_and_graph_rejection(self):
        import copy
        import tempfile
        import time
        from pathlib import Path
        from lab.calibration.vi_frame_updates import train_pass
        from lab.calibration.phystwin_vi import checkpoint, restore_checkpoint
        from test_vi_frame_updates import ToyRollout
        q=GraphResidualPosterior(torch.full((4,),8.,dtype=torch.double),self.settings,self.graph)
        opt=torch.optim.Adam(q.parameters(),lr=.001)
        meta=dict(source_commit='test',topology_sha256='test',graph_sha256='graph',config=dict(
            data={},likelihood={},posterior=self.settings,execution={},optimizer={'max_iterations':10}))
        train_pass(ToyRollout(q.mu),q,opt,3,1,time.monotonic()+30)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'checkpoint.pt'
            checkpoint(path,q,opt,1,'vi',meta)
            train_pass(ToyRollout(q.mu),q,opt,3,1,time.monotonic()+30)
            expected=copy.deepcopy(q.state_dict())
            restore_checkpoint(path,q,opt,'vi',meta)
            train_pass(ToyRollout(q.mu),q,opt,3,1,time.monotonic()+30)
            for key,value in q.state_dict().items():
                torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
            with self.assertRaisesRegex(ValueError,'graph mismatch'):
                restore_checkpoint(path,q,opt,'vi',dict(meta,graph_sha256='different'))

    def test_pair_metrics(self):
        import tempfile
        import json
        from pathlib import Path
        from lab.calibration.flow_analysis import compare_samples
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)
            candidates=[('a',torch.zeros(4)),('b',torch.ones(4))]
            predictions={'a':np.zeros((3,2,3)),'b':np.ones((3,2,3))*2}
            rows=[dict(name=name,calibration={'rmse_m':.1},continuation={'rmse_m':.2}) for name in ('a','b')]
            compare_samples(candidates,predictions,rows,3,2,path)
            pair=json.loads((path/'sample-pairs.json').read_text())['pairs'][0]
            self.assertEqual(pair['object_log_stiffness_rms'],1.)
            self.assertEqual(pair['controller_log_stiffness_rms'],1.)
            self.assertEqual(pair['continuation_prediction_rms_m'],2.)

if __name__=='__main__': unittest.main()
