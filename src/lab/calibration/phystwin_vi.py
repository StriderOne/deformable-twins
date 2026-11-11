"""Offline PhysTwin VI/MAP pilot. See experiments/E-0001.yaml and P-0001."""
import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import logging
import math
from pathlib import Path
import random
import subprocess
import tarfile
import threading
import time
import traceback

import jsonschema
import numpy as np
import torch
import yaml

from .phystwin_adapter import ROOT, SOURCE, COMMIT, build_simulator, sha256
from .posterior import LogStiffnessPosterior


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_plan(path: Path):
    plan = yaml.safe_load(path.read_text())
    jsonschema.validate(plan, json.loads((ROOT/'schemas/vi-experiment.schema.json').read_text()))
    if plan['source_commit'] != COMMIT:
        raise ValueError('Plan does not match adapter commit')
    for key in ('sigma_obs_m',):
        if not math.isfinite(plan['likelihood'][key]) or plan['likelihood'][key] <= 0:
            raise ValueError('Expected positive finite observation noise')
    for key in ('initial_std', 'prior_std'):
        if not math.isfinite(plan['posterior'][key]) or plan['posterior'][key] <= 0:
            raise ValueError('Expected positive finite posterior/prior scales')
    if plan['optimizer']['samples_per_iteration'] != 1 or plan['execution']['hardware_execution']:
        raise ValueError('Only one-sample offline pilot supported')
    protocol_path = ROOT / f"registry/protocols/{plan['protocol']['id']}.yaml"
    protocol = yaml.safe_load(protocol_path.read_text())
    registry_schema=json.loads((ROOT/'schemas/vi-registry.schema.json').read_text())
    jsonschema.validate(protocol,registry_schema)
    benchmark=yaml.safe_load((ROOT/f"registry/benchmarks/{plan['benchmark']['id']}.yaml").read_text())
    jsonschema.validate(benchmark,registry_schema)
    if benchmark['version'] != plan['benchmark']['version']:
        raise ValueError('Benchmark version mismatch')
    for baseline in plan['baselines']:
        spec=yaml.safe_load((ROOT/f'registry/baselines/{baseline}.yaml').read_text())
        jsonschema.validate(spec,registry_schema)
    if protocol['version'] != plan['protocol']['version']:
        raise ValueError('Protocol version mismatch')
    split = json.loads(Path(plan['data']['split']).expanduser().read_text())
    train, continuation = protocol['splits']['calibration_targets'], protocol['splits']['continuation_targets']
    if train != [1, split['train'][1]] or continuation != split['test'] or split['train'][0] != 0:
        raise ValueError('Dataset and protocol splits disagree')
    if train[1] != continuation[0] or continuation[1] != split['frame_len']:
        raise ValueError('Expected contiguous training/continuation and complete frames')
    manifest=json.loads((ROOT/plan['data']['manifest']).read_text())
    for key in ('observations','split','optimal'):
        if str(Path(plan['data'][key]).expanduser().resolve()) not in manifest['sources']:
            raise ValueError(f'Unregistered dataset input: {key}')
    for p,entry in manifest['sources'].items():
        if sha256(Path(p)) != entry['sha256']:
            raise ValueError(f'Dataset provenance mismatch: {p}')
    return plan, protocol


def metrics(losses, counts, sigma: float, split: int) -> dict:
    result = {}
    for name, section in [('calibration', slice(0, split - 1)), ('continuation', slice(split - 1, None))]:
        loss = float(np.sum(losses[section]))
        coordinates = 3 * int(np.sum(counts[section]))
        result[name] = dict(nll_without_constant=loss, valid_coordinates=coordinates,
                            rmse_m=math.sqrt(2 * sigma**2 * loss / coordinates) if coordinates else None)
    return result


def posterior_summary(q, n_object: int, threshold: float) -> dict:
    ratio = (q.std / q.prior_std).detach()
    result = {}
    for name, r in [('object', ratio[:n_object]), ('controller', ratio[n_object:])]:
        result[name] = dict(count=r.numel(), mean_ratio=float(r.mean()),
                            median_ratio=float(r.median()),
                            fraction_above_threshold=float((r >= threshold).float().mean())) if r.numel() else {'count': 0}
    return result


def checkpoint(path: Path, q, optimizer, iteration: int, mode: str, meta: dict):
    payload = dict(posterior=q.state_dict(), optimizer=optimizer.state_dict(), iteration=iteration,
                   mode=mode, torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                   numpy_rng=np.random.get_state(), python_rng=random.getstate(), metadata=meta)
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def restore_checkpoint(path, q, optimizer, mode, meta):
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['mode'] != mode or state['metadata']['source_commit'] != meta['source_commit']:
        raise ValueError('Resume mode/source mismatch')
    if state['metadata']['topology_sha256'] != meta['topology_sha256']:
        raise ValueError('Resume topology mismatch')
    old, new = state['metadata']['config'], meta['config']
    for key in ('data', 'likelihood', 'posterior', 'execution'):
        if old[key] != new[key]:
            raise ValueError(f'Resume configuration mismatch: {key}')
    for key in old['optimizer']:
        if key != 'max_iterations' and old['optimizer'][key] != new['optimizer'][key]:
            raise ValueError(f'Resume optimizer mismatch: {key}')
    if not 0 <= state['iteration'] < new['optimizer']['max_iterations']:
        raise ValueError('Resume iteration must precede target')
    q.load_state_dict(state['posterior'])
    optimizer.load_state_dict(state['optimizer'])
    torch.set_rng_state(state['torch_rng'])
    torch.cuda.set_rng_state_all(state['cuda_rng'])
    np.random.set_state(state['numpy_rng'])
    random.setstate(state['python_rng'])
    return state['iteration']


class MemoryMonitor:
    """Sample process memory, including Warp allocations invisible to Torch."""
    def __init__(self):
        import os
        self.pid = os.getpid()
        self.peak = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.poll, daemon=True)

    def poll(self):
        while not self.stop.is_set():
            try:
                out = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,used_gpu_memory',
                                               '--format=csv,noheader,nounits'], text=True)
                for line in out.splitlines():
                    pid, mem = line.split(',')
                    if int(pid) == self.pid:
                        self.peak = max(self.peak, int(mem))
            except (subprocess.SubprocessError, ValueError):
                pass  # Memory sampling is diagnostic, not part of scientific computation.
            self.stop.wait(1)


def prepare_run(args, plan, protocol):
    run_dir = args.run_dir.resolve()
    if not run_dir.name.startswith('R-'):
        raise ValueError('run-dir must use an R-NNNN ID')
    run_dir.mkdir(parents=True, exist_ok=False)
    artifacts = Path(plan['execution']['artifact_root']).expanduser() / run_dir.name
    artifacts.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(artifacts / 'run.log')])
    tracked = subprocess.check_output(['git', 'ls-files', '--cached', '--others', '--exclude-standard'], cwd=ROOT, text=True)
    snapshot = artifacts / 'source.tar.gz'
    source_hashes = {}
    with tarfile.open(snapshot, 'w:gz') as tar:
        for rel in sorted(set(tracked.splitlines())):
            path = ROOT / rel
            if path.is_file() and not rel.startswith('runs/'):
                source_hashes[rel] = sha256(path)
                tar.add(path, arcname=rel)
    inputs = {str(Path(p).expanduser().resolve()): sha256(Path(p).expanduser())
              for key, p in plan['data'].items() if key != 'manifest'}
    for option in ('map_checkpoint', 'vi_checkpoint', 'resume_checkpoint'):
        p = getattr(args, option, None)
        if p:
            inputs[str(p.resolve())] = sha256(p)
    manifest = dict(id=run_dir.name, format_version=1, experiment=plan['id'],
                    benchmark=plan['benchmark'], protocol=plan['protocol'],
                    baseline=plan.get('baseline_by_mode', {'map': 'BL-0002', 'vi': 'BL-0003', 'evaluate': 'BL-0001'})[args.mode],
                    mode=args.mode, status='running', started_at=datetime.now(timezone.utc).isoformat(),
                    repository_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    source_commit=COMMIT, source_hashes=source_hashes, source_snapshot=str(snapshot),
                    working_diff_sha256=hashlib.sha256(subprocess.check_output(['git','diff','HEAD'],cwd=ROOT)).hexdigest(),
                    config_sha256=sha256(args.config), config=plan, protocol_spec=protocol,
                    dataset_manifest_sha256=sha256(ROOT / plan['data']['manifest']), inputs=inputs,
                    versions={n: importlib.metadata.version(n) for n in ['torch','warp-lang','numpy','open3d','PyYAML']},
                    seed=plan['optimizer']['seed'], hardware=torch.cuda.get_device_name(0),
                    gpu_inventory=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
                    artifact_directory=str(artifacts), failure_reason=None,
                    limitations=['truncated_gradients','data_dependent_prior','single_session','assumed_noise','no_independent_test'])
    write_json(run_dir / 'manifest.json', manifest)
    write_json(artifacts / 'resolved-plan.json', {'experiment':plan,'protocol':protocol})
    return run_dir, artifacts, manifest


def train(rollout, plan, protocol, mode: str, artifacts: Path, resume_checkpoint=None):
    s = rollout.sim
    opt = plan['optimizer']
    seed = opt['seed']
    torch.manual_seed(seed)
    q = LogStiffnessPosterior(rollout.z_buffer.clone(), plan['posterior']['initial_std'], plan['posterior']['prior_std'])
    parameters = [q.mu, q.rho] if mode == 'vi' else [q.mu]
    optimizer = torch.optim.Adam(parameters, lr=opt['lr'], betas=tuple(opt['betas']))
    end = protocol['splits']['calibration_targets'][1]
    deadline = time.monotonic() + plan['budget'][f'{mode}_seconds']
    start = time.monotonic()
    iteration = 0
    meta = dict(config=plan, source_commit=COMMIT, object_springs=s.object_springs,
                topology_sha256=sha256(artifacts / 'topology.npz'))
    if resume_checkpoint:
        iteration = restore_checkpoint(resume_checkpoint, q, optimizer, mode, meta)
        meta['resume_checkpoint'] = str(resume_checkpoint.resolve())
        meta['resume_checkpoint_sha256'] = sha256(resume_checkpoint)
    initial_iteration = iteration
    checkpoint(artifacts / 'checkpoint.pt', q, optimizer, iteration, mode, meta)
    history = []
    status = 'completed'
    for iteration in range(initial_iteration + 1, opt['max_iterations'] + 1):
        if time.monotonic() >= deadline:
            status = 'budget_limited'
            break
        optimizer.zero_grad(set_to_none=True)
        z = q.sample() if mode == 'vi' else q.mu
        try:
            losses, grad = rollout.trajectory(z, end, True, deadline)
        except TimeoutError:
            status = 'budget_limited'
            break
        except Exception:
            torch.save({'sample':z.detach().cpu(), 'posterior':q.state_dict()}, artifacts / 'failed-sample.pt')
            raise
        penalty = q.kl() if mode == 'vi' else q.map_penalty()
        torch.autograd.backward((z, penalty), (grad, torch.ones_like(penalty)))
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters):
            raise FloatingPointError('Non-finite posterior gradient')
        # Record the objective at the pre-update sample; std summaries are post-update.
        penalty_value = float(penalty.detach())
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in parameters):
            raise FloatingPointError('Non-finite posterior parameter after update')
        threshold = protocol['primary_metric']['std_ratio_threshold']
        row = dict(iteration=iteration, seconds=time.monotonic()-start,
                   nll_without_constant=float(losses.sum()), penalty=penalty_value,
                   objective=float(losses.sum())+penalty_value,
                   rmse_m=math.sqrt(2 * plan['likelihood']['sigma_obs_m']**2 * losses.sum()
                                   / (3 * s.valid_counts[:end-1].sum())),
                   saturated_fraction=float((z.detach() >= math.log(s.spring_Y_max)).float().mean()),
                   std=posterior_summary(q, s.object_springs, threshold) if mode == 'vi' else None)
        history.append(row)
        with (artifacts / 'history.jsonl').open('a') as f:
            f.write(json.dumps(row, allow_nan=False)+'\n')
        checkpoint(artifacts / 'checkpoint.pt', q, optimizer, iteration, mode, meta)
        logging.info('%s iteration=%d nll=%.5g rmse=%.5g elapsed=%.1fs', mode, iteration,
                     row['nll_without_constant'], row['rmse_m'], row['seconds'])
    completed = len(history)
    return dict(status=status, completed_iterations=completed, initial_iteration=initial_iteration,
                total_iterations=initial_iteration + completed,
                training_seconds=time.monotonic()-start, last_iteration=history[-1] if history else None,
                convergence='not_established', posterior=posterior_summary(q,s.object_springs,protocol['primary_metric']['std_ratio_threshold']) if mode=='vi' else None)


def evaluate(rollout, plan, protocol, args, artifacts):
    start = time.monotonic()
    deadline = start + plan['budget']['evaluation_seconds']
    end = protocol['splits']['continuation_targets'][1]
    split = protocol['splits']['calibration_targets'][1]
    center = rollout.z_buffer.clone()
    candidates = [('CMA_initialization', center)]
    q_vi = None
    for mode, path in [('MAP',args.map_checkpoint),('VI_mu',args.vi_checkpoint)]:
        if path:
            state = torch.load(path, map_location='cuda:0', weights_only=False)
            q = LogStiffnessPosterior(center, plan['posterior']['initial_std'], plan['posterior']['prior_std'])
            q.load_state_dict(state['posterior'])
            if q.mu.shape != center.shape or state['metadata']['topology_sha256'] != sha256(artifacts/'topology.npz'):
                raise ValueError('Checkpoint topology mismatch')
            candidates.append((mode,q.mu.detach().clone()))
            if mode=='VI_mu': q_vi=q
    if q_vi is not None:
        torch.manual_seed(plan['optimizer']['seed']+1)
        for i in range(plan['budget']['posterior_samples']):
            candidates.append((f'VI_sample_{i:02}',q_vi.sample().detach()))
        ratios = (q_vi.std/q_vi.prior_std).detach().cpu().numpy()
        np.save(artifacts/'std_ratio.npy',ratios)
    rows = []
    for name,z in candidates:
        try:
            losses,_=rollout.trajectory(z,end,False,deadline)
        except TimeoutError:
            break
        except Exception:
            torch.save({'name':name,'sample':z.cpu()},artifacts/'failed-sample.pt')
            raise
        row=dict(name=name, **metrics(losses,rollout.sim.valid_counts,plan['likelihood']['sigma_obs_m'],split),
                 saturated_fraction=float((z>=math.log(rollout.sim.spring_Y_max)).float().mean()))
        rows.append(row)
        np.save(artifacts/f'{name}-frame-nll.npy',losses)
        write_json(artifacts/'evaluation.json',{'results':rows})
        logging.info('evaluation %s %s',name,row)
    plot_results(args, artifacts, rollout.sim.object_springs)
    return dict(status='completed' if len(rows)==len(candidates) else 'budget_limited',
                evaluation_seconds=time.monotonic()-start, completed_rollouts=len(rows),
                requested_rollouts=len(candidates),results=rows)


def plot_results(args, artifacts: Path, object_springs: int):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if (artifacts/'std_ratio.npy').exists():
        ratios=np.load(artifacts/'std_ratio.npy')
        fig,ax=plt.subplots()
        for name,r in [('Object',ratios[:object_springs]),('Controller',ratios[object_springs:])]:
            ax.hist(r,bins=40,alpha=0.6,label=name)
        ax.set(xlabel='Posterior std / prior std (log stiffness)',ylabel='Spring count')
        ax.legend(); fig.tight_layout(); fig.savefig(artifacts/'std-ratio.png'); plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for name,path in [('MAP',args.map_checkpoint),('VI',args.vi_checkpoint)]:
        if not path: continue
        history_path=path.parent/'history.jsonl'
        history=[json.loads(line) for line in history_path.read_text().splitlines()]
        axes[0].plot([r['iteration'] for r in history],[r['rmse_m'] for r in history],label=name)
        if name=='VI':
            for group in ('object','controller'):
                axes[1].plot([r['iteration'] for r in history],[r['std'][group]['mean_ratio'] for r in history],label=group)
    axes[0].set(xlabel='Iteration',ylabel='Training sampled RMSE [m]')
    axes[1].set(xlabel='Iteration',ylabel='Mean posterior / prior std')
    for ax in axes:
        if ax.lines: ax.legend()
    fig.tight_layout();fig.savefig(artifacts/'training.png');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--mode',choices=['map','vi','evaluate'],default='vi')
    parser.add_argument('--map-checkpoint',type=Path)
    parser.add_argument('--vi-checkpoint',type=Path)
    parser.add_argument('--resume-checkpoint', type=Path)
    args=parser.parse_args()
    if args.resume_checkpoint and args.mode == 'evaluate':
        parser.error('--resume-checkpoint requires training mode')
    plan,protocol=read_plan(args.config)
    run_dir,artifacts,manifest=prepare_run(args,plan,protocol)
    monitor=MemoryMonitor(); monitor.thread.start()
    start=time.monotonic()
    try:
        seed=plan['optimizer']['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
        rollout,data=build_simulator(plan)
        if len(data['object_points']) != protocol['splits']['continuation_targets'][1]:
            raise ValueError('Unexpected dataset frame count')
        s=rollout.sim
        np.savez(artifacts/'topology.npz',edges=s.wp_springs.numpy(),rest=s.wp_rest_lengths.numpy(),
                 mass=s.wp_masses.numpy(),object_springs=np.array(s.object_springs))
        manifest['resolved_physics']=s.resolved_config
        manifest['counts']=dict(particles=s.num_object_points,springs=s.n_springs,object_springs=s.object_springs)
        if plan['execution']['use_graph']: rollout.capture()
        result=(evaluate(rollout,plan,protocol,args,artifacts) if args.mode=='evaluate'
                else train(rollout,plan,protocol,args.mode,artifacts,args.resume_checkpoint))
        manifest.update(result)
    except BaseException as exc:
        manifest.update(status='failed',failure_reason=f'{type(exc).__name__}: {exc}')
        (artifacts/'traceback.txt').write_text(traceback.format_exc())
        raise
    finally:
        monitor.stop.set();monitor.thread.join()
        manifest.update(ended_at=datetime.now(timezone.utc).isoformat(),wall_seconds=time.monotonic()-start,
                        peak_process_gpu_memory_mib_sampled=monitor.peak,
                        artifacts={str(p):sha256(p) for p in artifacts.iterdir() if p.is_file()})
        write_json(run_dir/'manifest.json',manifest)


if __name__=='__main__':
    main()
