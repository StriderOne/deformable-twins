"""Opt-in synthetic CUDA capacity test; no simulator or observed data."""
import json,time,torch
from pathlib import Path
import yaml
from lab.calibration.residual_flow import SpringGraph,GraphResidualPosterior
n=114479
i=torch.arange(n)
graph=SpringGraph(torch.stack((i,i+1),1), torch.cat([torch.stack((i,(i+d)%n),1) for d in range(3,11)]))
settings=yaml.safe_load((Path(__file__).resolve().parents[1]/'experiments/E-0014.yaml').read_text())['posterior']
torch.manual_seed(43)
q=GraphResidualPosterior(torch.full((n,),8.,device='cuda'),settings,graph)
optimizer=torch.optim.Adam(q.parameters(),lr=.001)
torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();start=time.monotonic()
z,penalty=q.sample_and_penalty(q.draw_noise())
(z.square().sum()+penalty).backward()
assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in q.parameters())
optimizer.step();torch.cuda.synchronize()
print(json.dumps(dict(test='synthetic graph only; no physics or real data',springs=n,seconds=time.monotonic()-start,peak_allocated_mib=torch.cuda.max_memory_allocated()/2**20,finite=True)))
