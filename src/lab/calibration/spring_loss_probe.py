"""Conditional, single-spring forward sensitivity; no optimizer or KL."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import traceback

import numpy as np
import torch

from .phystwin_adapter import build_simulator, sha256
from .phystwin_vi import read_plan, prepare_run, write_json
from .posterior import make_posterior


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    plan, protocol = read_plan(args.config)
    cfg = plan['probe']
    args.mode = 'evaluate'
    args.vi_checkpoint = Path(cfg['checkpoint'])
    run_dir, artifacts, manifest = prepare_run(args, plan, protocol)
    start = time.monotonic()
    try:
        rollout, _ = build_simulator(plan)
        s = rollout.sim
        np.savez(artifacts/'topology.npz', edges=s.wp_springs.numpy(), rest=s.wp_rest_lengths.numpy(),
                 mass=s.wp_masses.numpy(), object_springs=np.array(s.object_springs))
        state = torch.load(args.vi_checkpoint, map_location=rollout.z_buffer.device, weights_only=False)
        assert state['metadata']['topology_sha256'] == sha256(artifacts/'topology.npz')
        for key in ('data', 'likelihood', 'posterior'):
            assert state['metadata']['config'][key] == plan[key], key
        q = make_posterior(rollout.z_buffer.clone(), plan['posterior'])
        q.load_state_dict(state['posterior'])
        with torch.no_grad():
            ratio = q.std_ratio()[:s.object_springs].cpu().numpy()
            selected = {'near_one': int(np.argmin(abs(ratio-1))), 'minimum': int(np.argmin(ratio))}
            baseline = q.point_estimate().detach().clone()
            mu = q.mu.detach().cpu().numpy()
            std = q.std.detach().cpu().numpy()
        write_json(artifacts/'selection.json', {k: {'spring_id': i, 'std_ratio': float(ratio[i]),
                    'median_Y': float(baseline[i].exp())} for k,i in selected.items()})
        rollout.capture()
        rng = np.random.default_rng(cfg['seed'])
        eps = rng.standard_normal(cfg['posterior_samples'])
        lower, upper = plan['posterior']['lower'], plan['posterior']['upper']
        jobs = []
        for j in range(cfg['posterior_samples']):
            for label, i in selected.items():
                y = lower + (upper-lower)/(1+np.exp(-(mu[i]+std[i]*eps[j])))
                jobs.append(('posterior', label, i, float(y), j))
        for j,y in enumerate(np.linspace(lower, upper, cfg['grid_points'])):
            for label,i in selected.items():
                jobs.append(('grid', label, i, float(y), j))
        # Baseline repeats spread across the execution, including both endpoints.
        for at in reversed(np.linspace(0,len(jobs),cfg['baseline_repeats']).astype(int)):
            jobs.insert(at, ('baseline','baseline',-1,0.0,-1))
        end = protocol['splits']['calibration_targets'][1]
        coords = 3*int(np.sum(s.valid_counts[:end-1]))
        frames, rows = [], []
        deadline = time.monotonic()+cfg['budget_seconds']
        with (artifacts/'samples.jsonl').open('x') as stream:
            for idx,(kind,label,i,y,j) in enumerate(jobs):
                z = baseline.clone()
                if i >= 0:
                    z[i] = np.log(y)
                assert int(torch.count_nonzero(z != baseline)) <= (i >= 0)
                losses,_ = rollout.trajectory(z,end,False,deadline)
                loss = float(losses.sum())
                row = dict(kind=kind,label=label,spring_id=i,Y=y,sample_index=j,
                           nll=loss,rmse_m=float(np.sqrt(2*plan['likelihood']['sigma_obs_m']**2*loss/coords)))
                stream.write(json.dumps(row)+'\n');stream.flush()
                rows.append(row);frames.append(losses)
                if idx % 20 == 0:
                    print(f'{idx+1}/{len(jobs)}: {kind} {label} NLL={loss:.8f}',flush=True)
        np.save(artifacts/'frame-nll.npy',np.asarray(frames))
        base = np.array([r['nll'] for r in rows if r['kind']=='baseline'])
        summary = {'baseline_mean':float(base.mean()),'baseline_std':float(base.std()),
                   'baseline_range':float(np.ptp(base)), 'selected':selected,'slices':{}}
        for label in selected:
            summary['slices'][label] = {kind: {'min':float(min(vals)), 'max':float(max(vals)),
                'range':float(np.ptp(vals)), 'std':float(np.std(vals))}
                for kind in ('posterior','grid')
                for vals in [[r['nll'] for r in rows if r['kind']==kind and r['label']==label]]}
        write_json(artifacts/'summary.json',summary)
        manifest.update(status='completed',result=summary,source_run=cfg['source_run'],
                        resolved_physics=s.resolved_config)
    except BaseException as exc:
        manifest.update(status='failed',failure_reason=f'{type(exc).__name__}: {exc}')
        (artifacts/'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        manifest.update(ended_at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-start,
                        artifacts={str(p):sha256(p) for p in artifacts.iterdir() if p.is_file()})
        write_json(run_dir/'manifest.json',manifest)


if __name__ == '__main__':
    main()
