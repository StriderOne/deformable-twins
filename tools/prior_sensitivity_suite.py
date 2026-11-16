"""Preflight, detached sequential training/evaluation, and paired analysis for E-0005."""
import argparse
import copy
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch
import yaml

from lab.calibration.phystwin_adapter import ROOT, build_simulator, sha256
from lab.calibration.phystwin_vi import read_plan, train, write_json

CONFIG = ROOT/'experiments/E-0005.yaml'


def preflight(output):
    output.mkdir(parents=True, exist_ok=False)
    base=yaml.safe_load(CONFIG.read_text())
    result={'status':'running','kind':'preflight_not_scientific_run', 'config_sha256':sha256(CONFIG),
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'script_sha256':sha256(Path(__file__)), 'variants':{}}
    try:
        plan,_=read_plan(CONFIG,'reference')
        rollout,_=build_simulator(plan)
        rollout.capture()
        s=rollout.sim
        initial_center=rollout.z_buffer.clone()
        first=None
        nll=None
        for variant in base['variants']:
            plan,protocol=read_plan(CONFIG,variant)
            plan['optimizer']['max_iterations']=base['preflight']['iterations']
            plan['budget']['vi_seconds']=base['preflight']['budget_seconds']
            folder=output/variant;folder.mkdir()
            np.savez(folder/'topology.npz',edges=s.wp_springs.numpy(),rest=s.wp_rest_lengths.numpy(),
                     mass=s.wp_masses.numpy(),object_springs=np.array(s.object_springs))
            # trajectory writes its sample into the shared stiffness buffer.
            rollout.z_buffer.copy_(initial_center)
            r=train(rollout,plan,protocol,'vi',folder)
            assert r['status']=='completed',r
            state=torch.load(folder/'checkpoint-000000.pt',map_location='cpu',weights_only=False)['posterior']
            if first is None:first=state
            for key in ('mu','rho'):torch.testing.assert_close(state[key],first[key],rtol=0,atol=0)
            row=json.loads((folder/'history.jsonl').read_text().splitlines()[0])
            if nll is None:nll=row['nll_without_constant']
            assert np.isclose(row['nll_without_constant'],nll,rtol=1e-6)
            result['variants'][variant]=r
            print('PREFLIGHT',variant,r['status'],flush=True)
        result['status']='passed'
    except BaseException:
        result.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        write_json(output/'validation.json',result)


def summarize(base,output):
    states={}
    provenance={}
    evaluations={}
    for v,ids in base['suite_runs'].items():
        m=json.loads((ROOT/'runs'/ids['train']/'manifest.json').read_text())
        folder=Path(m['artifact_directory'])
        states[v]={i:torch.load(folder/f'checkpoint-{i:06}.pt',map_location='cpu',weights_only=False)
                   for i in base['checkpoint_iterations']}
        provenance[v]={'train':ids['train'],'evaluate':ids['evaluate'],'commit':m['repository_commit'],
                       'checkpoint_sha256':sha256(folder/'checkpoint.pt')}
        em=json.loads((ROOT/'runs'/ids['evaluate']/'manifest.json').read_text())
        evaluations[v]=em['results']
    topology={s[base['optimizer']['max_iterations']]['metadata']['topology_sha256'] for s in states.values()}
    assert len(topology)==1
    n=states['reference'][0]['metadata']['object_springs']
    def summary(x):
        return dict(mean=float(x.mean()),p05=float(np.quantile(x,.05)),median=float(np.median(x)),p95=float(np.quantile(x,.95)))
    rows=[]
    for i in base['checkpoint_iterations']:
        p={v:s[i]['posterior'] for v,s in states.items()}
        mu={v:x['mu'].double().numpy() for v,x in p.items()}
        sd={v:torch.nn.functional.softplus(x['rho'].double()).numpy() for v,x in p.items()}
        for group,sl in [('object',slice(None,n)),('controller',slice(n,None))]:
            center=(mu['center_high']-mu['center_low'])/np.log(base['variants']['center_high']['prior_center_factor']/base['variants']['center_low']['prior_center_factor'])
            width=np.log(sd['width_wide']/sd['width_narrow'])/np.log(base['variants']['width_wide']['prior_std']/base['variants']['width_narrow']['prior_std'])
            row={'iteration':i,'group':group,'center_response':summary(center[sl]),'width_response':summary(width[sl]),'versus_reference':{}}
            for v in p:
                dm=mu[v][sl]-mu['reference'][sl];ds=sd[v][sl]-sd['reference'][sl]
                row['versus_reference'][v]={'mu_abs':summary(abs(dm)),'std_abs':summary(abs(ds)), 'marginal_w2':summary(np.sqrt(dm**2+ds**2))}
            rows.append(row)
    write_json(output/'comparison.json',{'interpretation':'prior_sensitivity_not_percentage_information','inputs':provenance,'paired_checkpoints':rows,'evaluations':evaluations})


def run(output):
    base=yaml.safe_load(CONFIG.read_text())
    validation=json.loads((output/'validation.json').read_text())
    assert validation['status']=='passed' and validation['config_sha256']==sha256(CONFIG)
    artifact_root=Path(base['execution']['artifact_root']).expanduser()
    for ids in base['suite_runs'].values():
        for rid in ids.values():
            if (ROOT/'runs'/rid).exists() or (artifact_root/rid).exists():raise FileExistsError(rid)
    status={'status':'running','started_at_unix':time.time(),'completed':[]}
    try:
        for variant,ids in base['suite_runs'].items():
            for mode,key in [('vi','train'),('evaluate','evaluate')]:
                status['active']={'variant':variant,'mode':mode,'run':ids[key]}
                write_json(output/'suite-status.json',status)
                cmd=[sys.executable,'-m','lab.calibration.phystwin_vi','--config',str(CONFIG),'--variant',variant,'--mode',mode,'--run-dir',str(ROOT/'runs'/ids[key])]
                if mode=='evaluate':cmd+=['--vi-checkpoint',str(artifact_root/ids['train']/'checkpoint.pt')]
                subprocess.run(cmd,cwd=ROOT,check=True)
                m=json.loads((ROOT/'runs'/ids[key]/'manifest.json').read_text())
                if m['status']!='completed':raise RuntimeError(f"{ids[key]}: {m['status']}")
                status['completed'].append(ids[key])
        summarize(base,output)
        status['status']='completed'
    except BaseException:
        status.update(status='failed',error=traceback.format_exc())
        raise
    finally:
        status['ended_at_unix']=time.time()
        write_json(output/'suite-status.json',status)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['preflight','run'])
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    (preflight if args.mode=='preflight' else run)(args.output)
