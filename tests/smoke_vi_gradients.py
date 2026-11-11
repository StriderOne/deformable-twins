"""Explicit GPU gradient checks; not an automatic hardware/scientific CI evaluation."""
import argparse
import json
from pathlib import Path
import tempfile
import time
import numpy as np
import torch
import warp as wp
from lab.calibration.phystwin_adapter import load_upstream, simulator_class, Rollout, build_simulator, setup_cuda
from lab.calibration.phystwin_vi import read_plan, ROOT, checkpoint
from lab.calibration.posterior import LogStiffnessPosterior


def tiny():
    setup_cuda()
    cfg,mod,_=load_upstream()
    cfg.device='cuda:0';cfg.use_graph=False
    def t(a,dtype=torch.float32): return torch.tensor(a,dtype=dtype,device='cuda:0')
    vertices=t([[0,0,1],[0.12,0,1],[0.24,0,1]])
    gt=t([[[0,0,1],[0.12,0,1]],[[0.001,0,1],[0.125,0,1]],[[0.002,0,1],[0.13,0,1]]])
    s=simulator_class(mod)(vertices,t([[0,1],[1,2]],torch.int32),t([0.1,0.1]),t([1,1,1]),
        dt=0.001,num_substeps=20,spring_Y=20.,collide_elas=0.2,collide_fric=0.2,
        dashpot_damping=0.1,drag_damping=0.1,num_object_points=2,num_original_points=2,
        num_surface_points=2,controller_points=t([[[0.24,0,1]],[[0.25,0,1]],[[0.26,0,1]]]),
        reverse_z=False,spring_Y_min=0.,spring_Y_max=1e5,gt_object_points=gt,
        gt_object_visibilities=t([[1,1],[1,0],[1,1]],torch.bool),
        gt_object_motions_valid=t([[1,1],[0,1],[1,1]],torch.bool))
    s.inverse_variance=10000.
    return Rollout(s)


def check():
    r=tiny();z=r.z_buffer.clone()
    r.reset(z);loss,grad=r.frame(1,True,False)
    # Mask excludes the second point: validate both normalization and mask.
    pred=wp.to_torch(r.sim.wp_states[-1].wp_x,requires_grad=False)
    expected=0.5*10000.*((pred[0]-r.sim.gt_object_points[1,0])**2).sum()
    np.testing.assert_allclose(loss,expected.item(),rtol=1e-5)
    finite=[];h=0.002
    for i in range(len(z)):
        delta=torch.zeros_like(z);delta[i]=h
        r.reset(z+delta);plus,_=r.frame(1,False,False)
        r.reset(z-delta);minus,_=r.frame(1,False,False)
        finite.append((plus-minus)/(2*h))
    np.testing.assert_allclose(grad.cpu(),finite,rtol=0.03,atol=2e-5)
    q=LogStiffnessPosterior(z,0.05,0.5)
    eps=torch.tensor([0.2,-0.3],device='cuda:0')
    sample=q.sample(eps)
    r.reset(sample);sample_loss,g=r.frame(1,True,False)
    sample.backward(g)
    # Finite difference in rho checks the complete Warp -> Torch bridge.
    with torch.no_grad():
        original=q.rho[0].item();h_rho=0.02
        q.rho[0]=original+h_rho;r.reset(q.sample(eps));plus,_=r.frame(1,False,False)
        q.rho[0]=original-h_rho;r.reset(q.sample(eps));minus,_=r.frame(1,False,False)
        q.rho[0]=original
    np.testing.assert_allclose(q.rho.grad[0].item(),(plus-minus)/(2*h_rho),rtol=0.05,atol=2e-5)
    r.reset(z);eager_loss,eager_g=r.frame(1,True,False)
    r.reset(z);r.capture()
    for _ in range(2):
        r.reset(z);graph_loss,graph_g=r.frame(1,True,True)
        np.testing.assert_allclose(graph_loss,eager_loss,rtol=1e-5)
        torch.testing.assert_close(graph_g,eager_g,rtol=1e-4,atol=1e-6)
    # Changing frame/mask must update graph inputs, with no stale gradient accumulation.
    graph_losses,graph_grad=r.trajectory(z,3,True)
    torch.testing.assert_close(r.z_buffer,z)
    eager=tiny();eager_losses,eager_grad=eager.trajectory(z,3,True)
    np.testing.assert_allclose(graph_losses,eager_losses,rtol=1e-4)
    torch.testing.assert_close(graph_grad,eager_grad,rtol=1e-4,atol=1e-6)
    opt=torch.optim.Adam(q.parameters())
    with tempfile.TemporaryDirectory() as d:
        p=Path(d)/'checkpoint.pt';checkpoint(p,q,opt,1,'vi',{})
        restored=torch.load(p,weights_only=False,map_location='cuda:0')
        torch.testing.assert_close(restored['posterior']['mu'],q.mu)
        assert restored['iteration']==1 and 'cuda_rng' in restored
    return dict(status='passed',loss=loss,autodiff=grad.cpu().tolist(),finite_difference=finite,
                checks=['mask_and_nll','finite_difference_z','finite_difference_rho',
                        'eager_graph','reset','gradient_clear','constant_sample','checkpoint'])


def main():
    p=argparse.ArgumentParser();p.add_argument('--sloth',action='store_true');p.add_argument('--output',type=Path)
    args=p.parse_args();result=check()
    if args.sloth:
        plan,_=read_plan(ROOT/'experiments/E-0001.yaml')
        r,_=build_simulator(plan);r.capture();start=time.monotonic()
        z=r.z_buffer.clone();loss,g=r.trajectory(z,4,True)
        assert torch.isfinite(g).all() and np.isfinite(loss).all()
        result['sloth']=dict(frames=3,loss=loss.tolist(),gradient_norm=g.norm().item(),seconds=time.monotonic()-start,
                             particles=r.sim.num_object_points,springs=r.sim.n_springs,object_springs=r.sim.object_springs)
    text=json.dumps(result,indent=2)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text+'\n')
if __name__=='__main__':main()
