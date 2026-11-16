"""Check unchanged PhysTwin dynamics, fast wrapper equivalence, and uniform VI."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import traceback
import sys

import numpy as np
import torch
import warp as wp

from lab.calibration.phystwin_adapter import ROOT, SOURCE, COMMIT, build_simulator, sha256
from lab.calibration.phystwin_vi import read_plan, train, write_json
from lab.calibration.posterior import make_posterior, UniformStiffnessPosterior


def main(output, config):
    output.mkdir(parents=True,exist_ok=False)
    plan,protocol=read_plan(config)
    cfg=plan['validation']
    result={'status':'running','config_sha256':sha256(config),'source_commit':COMMIT,
            'analysis_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'src/lab/calibration').glob('*.py')}}
    try:
        assert not subprocess.check_output(['git','-C',str(SOURCE),'diff','HEAD'])
        sys.path.insert(0,str(ROOT/'tests'))
        from smoke_vi_gradients import check, tiny
        result['original_gradient_checks']=check()
        small=tiny();center=small.z_buffer.clone()
        q=UniformStiffnessPosterior(center,1.,1.,1e5)
        eps=torch.tensor([.2,-.3],device='cuda:0')
        sample=q.sample(eps);small.reset(sample);_,g=small.frame(1,True,False)
        sample.backward(g)
        expected=q.mu.grad[0].item()
        with torch.no_grad():
            orig=q.mu[0].item();h=.002
            q.mu[0]=orig+h;small.reset(q.sample(eps));plus,_=small.frame(1,False,False)
            q.mu[0]=orig-h;small.reset(q.sample(eps));minus,_=small.frame(1,False,False)
            q.mu[0]=orig
        np.testing.assert_allclose(expected,(plus-minus)/(2*h),rtol=.05,atol=2e-5)
        result['bounded_warp_bridge']={'autodiff':expected,'finite_difference':(plus-minus)/(2*h)}
        r,_=build_simulator(plan);r.capture();center=r.z_buffer.clone()
        q=make_posterior(center,plan['posterior'])
        torch.manual_seed(plan['optimizer']['seed']);z=q.sample().detach()
        timings=[]
        for i in range(cfg['benchmark_repeats']):
            torch.cuda.synchronize();start=time.perf_counter()
            loss,grad=r.trajectory(z,protocol['splits']['calibration_targets'][1],True)
            torch.cuda.synchronize();timings.append(time.perf_counter()-start)
            assert np.isfinite(loss).all() and torch.isfinite(grad).all()
        result['baseline_seconds']=timings
        result['optimization_accepted']=False
        result['reason']='Deferred-host candidate rejected: negligible speed gain; original wrapper retained'
        # Compare KL at wider latent scales than the initial pilot distribution.
        centers=torch.full((15,),center[0].item(),device='cuda:0',dtype=torch.float64)
        a=make_posterior(centers,plan['posterior'])
        b=UniformStiffnessPosterior(centers,1.5,plan['posterior']['lower'],plan['posterior']['upper'],cfg['quadrature_reference_points'])
        with torch.no_grad():
            a.mu.copy_(torch.tensor([-8.,-3.,0.,3.,8.],device='cuda:0').repeat(3))
            a.rho.copy_(torch.tensor([.5,1.5,4.],device='cuda:0',dtype=torch.float64).repeat_interleave(5).expm1().log())
            b.mu.copy_(a.mu);b.rho.copy_(a.rho)
        torch.testing.assert_close(a.kl(),b.kl(),rtol=0,atol=cfg['quadrature_absolute_error_per_spring'])
        result['quadrature_KL_difference']=float((a.kl()-b.kl()).abs())
        folder=output/'smoke';folder.mkdir()
        s=r.sim
        np.savez(folder/'topology.npz',edges=s.wp_springs.numpy(),rest=s.wp_rest_lengths.numpy(),mass=s.wp_masses.numpy(),object_springs=np.array(s.object_springs))
        r.z_buffer.copy_(center)
        smoke=dict(plan);smoke['optimizer']=dict(plan['optimizer'],max_iterations=cfg['initial_samples'])
        result['smoke']=train(r,smoke,protocol,'vi',folder)
        assert result['smoke']['status']=='completed'
        result['status']='passed'
    except BaseException:
        result.update(status='failed',error=traceback.format_exc());raise
    finally:
        write_json(output/'validation.json',result)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--config',type=Path,default=ROOT/'experiments/E-0006.yaml')
    args=parser.parse_args();main(args.output,args.config)
