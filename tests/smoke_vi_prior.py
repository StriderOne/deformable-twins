"""Explicit stability gate for prior-initialized VI; never resample failures."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np
import torch

from lab.calibration.phystwin_adapter import ROOT, COMMIT, build_simulator, sha256
from lab.calibration.phystwin_vi import read_plan, write_json
from lab.calibration.posterior import LogStiffnessPosterior


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    plan,protocol=read_plan(args.config)
    result=dict(kind='stability_validation_not_training',experiment=plan['id'],protocol=plan['protocol'],
                repository_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                source_commit=COMMIT,config_sha256=sha256(args.config),samples=[],status='running')
    start=time.monotonic()
    try:
        r,_=build_simulator(plan);r.capture()
        q=LogStiffnessPosterior(r.z_buffer.clone(),plan['posterior']['initial_std'],plan['posterior']['prior_std'])
        torch.manual_seed(plan['preflight']['seed'])
        assert torch.allclose(q.std,q.prior_std)
        result['initial_kl']=q.kl().item()
        for i in range(plan['preflight']['samples']):
            sample=q.sample()
            try:
                losses,g=r.trajectory(sample,plan['preflight']['end_frame_exclusive'],True)
            except BaseException:
                torch.save({'sample':sample.detach().cpu(),'index':i},args.output/'failed-sample.pt')
                raise
            assert np.isfinite(losses).all() and torch.isfinite(g).all()
            result['samples'].append(dict(index=i,nll=float(losses.sum()),gradient_norm=g.norm().item(),
                saturated_fraction=(sample.detach()>=math.log(r.sim.spring_Y_max)).float().mean().item()))
        result['status']='passed'
    except BaseException as exc:
        result.update(status='failed',failure_reason=f'{type(exc).__name__}: {exc}')
        (args.output/'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        result['seconds']=time.monotonic()-start
        write_json(args.output/'validation.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
