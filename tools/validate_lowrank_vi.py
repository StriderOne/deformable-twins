from pathlib import Path
from types import SimpleNamespace
import json,shutil,sys
import numpy as np
import torch
from lab.calibration.phystwin_vi import read_plan,train,evaluate,write_json
from lab.calibration.phystwin_adapter import ROOT,build_simulator,sha256
from lab.calibration.posterior import make_posterior
sys.path.insert(0,str(ROOT/'tests'))
from smoke_vi_gradients import check,tiny
config=Path('experiments/E-0011.yaml');p,protocol=read_plan(config)
out=Path('/home/homa/.local/share/deformable-twins/artifacts/E-0011-validation');out.mkdir()
checks=check()
r=tiny();settings=dict(p['posterior'],rank=2,lower=1.);q=make_posterior(r.z_buffer.clone(),settings)
eps=torch.tensor([.2,-.3],device='cuda');eta=torch.tensor([.4,-.2],device='cuda');z=q.sample(eps,eta);r.reset(z);_,g=r.frame(1,True,False);z.backward(g)
ad=q.factor.grad[0,0].item();orig=q.factor[0,0].item();h=.002
with torch.no_grad():
 q.factor[0,0]=orig+h;r.reset(q.sample(eps,eta));a,_=r.frame(1,False,False)
 q.factor[0,0]=orig-h;r.reset(q.sample(eps,eta));b,_=r.frame(1,False,False)
np.testing.assert_allclose(ad,(a-b)/(2*h),rtol=.05,atol=2e-5)
r,_=build_simulator(p);r.capture();s=r.sim;folder=out/'smoke';folder.mkdir()
np.savez(folder/'topology.npz',edges=s.wp_springs.numpy(),rest=s.wp_rest_lengths.numpy(),mass=s.wp_masses.numpy(),object_springs=np.array(s.object_springs))
smoke=dict(p);smoke['optimizer']=dict(p['optimizer'],max_iterations=3);smoke['tensorboard']=dict(p['tensorboard'],image_every=3)
result=train(r,smoke,protocol,'vi',folder);assert result['status']=='completed'
# Rebuild to restore CMA initialization for evaluation.
r,_=build_simulator(p);r.capture();ev=out/'evaluation-smoke';ev.mkdir();shutil.copyfile(folder/'topology.npz',ev/'topology.npz');p['budget']['posterior_samples']=2
res=evaluate(r,p,protocol,SimpleNamespace(map_checkpoint=None,vi_checkpoint=folder/'checkpoint.pt'),ev);assert res['status']=='completed';write_json(ev/'validation.json',res)
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
acc=EventAccumulator(str(folder/'tensorboard'));acc.Reload();assert 'mesh/std_relative_to_prior' in acc.Tags()['images'];assert len(acc.Scalars('train/objective'))==3
write_json(out/'validation.json',dict(status='passed',config_sha256=sha256(config),source_hashes={str(f.relative_to(ROOT)):sha256(f) for f in (ROOT/'src/lab/calibration').glob('*.py')},smoke=result,gradient_checks=checks,factor_gradient={'autodiff':ad,'finite_difference':(a-b)/(2*h)},tensorboard_tags=acc.Tags()))
