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
from .posterior import LogStiffnessPosterior, UniformStiffnessPosterior, make_posterior
from .residual_flow import GraphResidualPosterior, SpringGraph, spatial_edges


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def read_plan(path: Path, variant: str | None = None):
    plan = yaml.safe_load(path.read_text())
    if plan.get('variants'):
        if variant not in plan['variants']:
            raise ValueError('Select a configured prior variant')
        plan['posterior'].update(plan['variants'][variant])
        plan['selected_variant'] = variant
    elif variant is not None:
        raise ValueError('Plan has no variants')
    jsonschema.validate(plan, json.loads((ROOT/'schemas/vi-experiment.schema.json').read_text()))
    if plan['source_commit'] != COMMIT:
        raise ValueError('Plan does not match adapter commit')
    for key in ('sigma_obs_m',):
        if not math.isfinite(plan['likelihood'][key]) or plan['likelihood'][key] <= 0:
            raise ValueError('Expected positive finite observation noise')
    for key in ('initial_std', 'prior_std') if plan['posterior']['family'] == 'diagonal_gaussian_log_stiffness' else ('initial_std',):
        if not math.isfinite(plan['posterior'][key]) or plan['posterior'][key] <= 0:
            raise ValueError('Expected positive finite posterior/prior scales')
    if plan['posterior']['family'] in ('logistic_normal_stiffness', 'low_rank_logistic_normal_stiffness', 'graph_residual_flow'):
        if not 0 < plan['posterior']['lower'] < plan['posterior']['upper'] <= 1e5:
            raise ValueError('Uniform bounds must lie within positive PhysTwin stiffness support')
    if 'initial_median_Y' in plan['posterior']:
        value = plan['posterior']['initial_median_Y']
        if not plan['posterior']['lower'] < value < plan['posterior']['upper']:
            raise ValueError('Initial physical median must be strictly inside prior bounds')
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
    with torch.no_grad():
        ratio = q.std_ratio().detach()
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
    if state['metadata'].get('graph_sha256') != meta.get('graph_sha256'):
        raise ValueError('Resume graph mismatch')
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
    if plan['id'] == 'E-0014':
        status = subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all','--ignore-submodules=none'], cwd=ROOT, text=True)
        if status:
            raise RuntimeError('Scientific run requires a clean worktree:\n'+status)
        for note in ROOT.rglob('*.md'):
            if '.git' not in note.parts and any(line.lstrip().startswith('- [ ] Human input required') for line in note.read_text(errors='replace').splitlines()):
                raise RuntimeError(f'Outstanding researcher input: {note}')
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
    if plan['posterior'].get('prior_family') == 'uniform_stiffness':
        manifest['limitations'].remove('data_dependent_prior')
        manifest['limitations'].extend(['CMA_initialized_posterior','bounded_variational_family','quadrature_KL'])
    if plan['posterior']['family'] == 'graph_residual_flow':
        manifest['limitations'].remove('quadrature_KL')
        manifest['limitations'].extend(['stochastic_logdet','framewise_parameter_changes','initial_graph_choice','mode_coverage_unestablished'])
        manifest['versions']['scipy'] = importlib.metadata.version('scipy')
    if plan['id'] == 'E-0014':
        manifest['clean_worktree_verified'] = True
    if plan['id'] == 'E-0014' and args.mode == 'evaluate' and getattr(args, 'vi_checkpoint', None):
        state = torch.load(args.vi_checkpoint, map_location='cpu', weights_only=False)
        manifest['baseline'] = state['metadata']['config']['baseline_by_mode']['vi']
    if plan.get('comet', {}).get('enabled'):
        manifest['versions']['comet_ml'] = importlib.metadata.version('comet_ml')
    if 'initial_median_Y' in plan['posterior']:
        manifest['limitations'].remove('CMA_initialized_posterior')
        manifest['limitations'].append('CMA_initialized_topology_and_other_physics')
    write_json(run_dir / 'manifest.json', manifest)
    write_json(artifacts / 'resolved-plan.json', {'experiment':plan,'protocol':protocol})
    return run_dir, artifacts, manifest


def build_posterior(center, settings, plan, artifacts):
    graph = None
    if settings['family'] == 'graph_residual_flow':
        center = torch.full_like(center, math.log(settings['initial_median_Y']))
        import pickle
        with Path(plan['data']['observations']).open('rb') as f:
            data = pickle.load(f)
        points = np.concatenate([data['object_points'][0], data['surface_points'], data['interior_points'], data['controller_points'][0]])
        topology = np.load(artifacts/'topology.npz')
        endpoints = topology['edges'].astype(np.int64)
        cache = artifacts/'posterior-graph.npz'
        if cache.exists():
            saved = np.load(cache)
            if not np.array_equal(saved['endpoints'], endpoints):
                raise ValueError('Graph topology mismatch')
            neighbors = saved['neighbors']
        else:
            neighbors = spatial_edges(points, endpoints, settings['spatial_neighbors'])
            np.savez(cache, endpoints=endpoints, neighbors=neighbors)
        graph = SpringGraph(endpoints, neighbors)
        degree = np.bincount(neighbors.ravel(), minlength=len(endpoints))
        write_json(artifacts/'graph-summary.json', dict(sha256=sha256(cache),
                   springs=len(endpoints), spatial_pairs=len(neighbors),
                   degree_max=int(degree.max()), degree_mean=float(degree.mean())))
    return make_posterior(center, settings, graph)


def train(rollout, plan, protocol, mode: str, artifacts: Path, resume_checkpoint=None):
    s = rollout.sim
    opt = plan['optimizer']
    seed = opt['seed']
    torch.manual_seed(seed)
    center = rollout.z_buffer.clone()
    if 'initial_median_Y' in plan['posterior']:
        center.fill_(math.log(plan['posterior']['initial_median_Y']))
    q = build_posterior(center, plan['posterior'], plan, artifacts)
    if mode != 'vi' and isinstance(q, (UniformStiffnessPosterior, GraphResidualPosterior)):
        raise ValueError('Uniform pilot supports VI only')
    parameters = list(q.parameters()) if mode == 'vi' else [q.mu]
    optimizer = torch.optim.Adam(parameters, lr=opt['lr'], betas=tuple(opt['betas']))
    end = protocol['splits']['calibration_targets'][1]
    deadline = time.monotonic() + plan['budget'][f'{mode}_seconds']
    start = time.monotonic()
    iteration = 0
    meta = dict(config=plan, source_commit=COMMIT, object_springs=s.object_springs,
                topology_sha256=sha256(artifacts / 'topology.npz'))
    if isinstance(q, GraphResidualPosterior):
        meta['graph_sha256'] = sha256(artifacts/'posterior-graph.npz')
    if resume_checkpoint:
        iteration = restore_checkpoint(resume_checkpoint, q, optimizer, mode, meta)
        meta['resume_checkpoint'] = str(resume_checkpoint.resolve())
        meta['resume_checkpoint_sha256'] = sha256(resume_checkpoint)
    initial_iteration = iteration
    checkpoint(artifacts / 'checkpoint.pt', q, optimizer, iteration, mode, meta)
    if initial_iteration in plan.get('checkpoint_iterations', []):
        checkpoint(artifacts / f'checkpoint-{initial_iteration:06d}.pt', q, optimizer, initial_iteration, mode, meta)
    monitor = None
    if plan.get('tensorboard', {}).get('enabled'):
        if plan.get('comet', {}).get('enabled'):
            from lab.calibration.vi_multistart_monitor import MultistartMonitor as Monitor
        else:
            from lab.calibration.vi_monitor import Monitor
        monitor = Monitor(plan, artifacts, q, s.object_springs)
    evaluation = plan.get('periodic_evaluation')
    if evaluation:
        from lab.calibration.vi_multistart_monitor import evaluate_current
        result = evaluate_current(rollout, q, plan, protocol, artifacts, initial_iteration, deadline)
        monitor.evaluation(result, initial_iteration)
    try:
        history = []
        status = 'completed'
        for iteration in range(initial_iteration + 1, opt['max_iterations'] + 1):
            if time.monotonic() >= deadline:
                status = 'budget_limited'
                break
            interval = opt.get('update_every_frames', end - 1)
            if interval < end - 1:
                from .vi_frame_updates import train_pass
                try:
                    losses, penalty_value, z = train_pass(
                        rollout, q, optimizer, end, interval,
                        float('inf') if isinstance(q, GraphResidualPosterior) else deadline)
                except BaseException:
                    if isinstance(q, GraphResidualPosterior):
                        torch.save({'posterior':q.state_dict(), 'log_stiffness':rollout.z_buffer.detach().cpu(),
                                    'partial_iteration':iteration}, artifacts/'failed-partial-pass.pt')
                    raise
            else:
                optimizer.zero_grad(set_to_none=True)
                if isinstance(q, GraphResidualPosterior):
                    z, flow_penalty = q.sample_and_penalty(q.draw_noise())
                else:
                    z = q.sample() if mode == 'vi' else q.mu
                try:
                    losses, grad = rollout.trajectory(z, end, True, deadline)
                except TimeoutError:
                    status = 'budget_limited'
                    break
                except Exception:
                    torch.save({'sample':z.detach().cpu(), 'posterior':q.state_dict()}, artifacts / 'failed-sample.pt')
                    raise
                penalty = (flow_penalty if isinstance(q, GraphResidualPosterior)
                           else q.kl() if mode == 'vi' else q.map_penalty())
                torch.autograd.backward((z, penalty), (grad, torch.ones_like(penalty)))
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in parameters):
                    raise FloatingPointError('Non-finite posterior gradient')
                # Record the objective at the pre-update sample; std summaries are post-update.
                penalty_value = float(penalty.detach())
                optimizer.step()
                if not all(torch.isfinite(p).all() for p in parameters):
                    raise FloatingPointError('Non-finite posterior parameter after update')
            threshold = protocol['primary_metric']['std_ratio_threshold']
            row = dict(iteration=iteration, optimizer_steps=iteration * math.ceil((end-1)/interval),
                       seconds=time.monotonic()-start,
                       nll_without_constant=float(losses.sum()), penalty=penalty_value,
                       objective=float(losses.sum())+penalty_value,
                       rmse_m=math.sqrt(2 * plan['likelihood']['sigma_obs_m']**2 * losses.sum()
                                       / (3 * s.valid_counts[:end-1].sum())),
                       saturated_fraction=float((z.detach() >= math.log(s.spring_Y_max)).float().mean()),
                       std=posterior_summary(q, s.object_springs, threshold) if mode == 'vi' else None)
            if isinstance(q, UniformStiffnessPosterior):
                with torch.no_grad():
                    fraction = (z.detach().exp()-q.lower)/(q.upper-q.lower)
                    margin = plan['diagnostics']['boundary_margin_fraction']
                    row['near_uniform_bounds_fraction'] = float(((fraction < margin) | (fraction > 1-margin)).float().mean())
                    row['max_logit_std'] = float(q.std.max())
            if monitor:
                monitor.update(row, q)
            if evaluation and iteration % evaluation['every'] == 0:
                result = evaluate_current(rollout, q, plan, protocol, artifacts, iteration, deadline)
                monitor.evaluation(result, iteration)
            history.append(row)
            with (artifacts / 'history.jsonl').open('a') as f:
                f.write(json.dumps(row, allow_nan=False)+'\n')
            if iteration % plan.get('checkpoint_every', 1) == 0 or iteration == opt['max_iterations']:
                checkpoint(artifacts / 'checkpoint.pt', q, optimizer, iteration, mode, meta)
            if iteration in plan.get('checkpoint_iterations', []):
                checkpoint(artifacts / f'checkpoint-{iteration:06d}.pt', q, optimizer, iteration, mode, meta)
            logging.info('%s iteration=%d nll=%.5g rmse=%.5g elapsed=%.1fs', mode, iteration,
                         row['nll_without_constant'], row['rmse_m'], row['seconds'])
    finally:
        if monitor:
            monitor.close()
    completed = len(history)
    checkpoint(artifacts / 'checkpoint.pt', q, optimizer, initial_iteration+completed, mode, meta)
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
            q = build_posterior(center, state['metadata']['config']['posterior'], plan, artifacts)
            if isinstance(q, GraphResidualPosterior) and state['metadata'].get('graph_sha256') != sha256(artifacts/'posterior-graph.npz'):
                raise ValueError('Evaluation graph mismatch')
            q.load_state_dict(state['posterior'])
            if q.mu.shape != center.shape or state['metadata']['topology_sha256'] != sha256(artifacts/'topology.npz'):
                raise ValueError('Checkpoint topology mismatch')
            label = 'VI_median' if mode == 'VI_mu' and isinstance(q, UniformStiffnessPosterior) else mode
            if not isinstance(q, GraphResidualPosterior):
                candidates.append((label,q.point_estimate().detach().clone()))
            if mode=='VI_mu': q_vi=q
    if q_vi is not None:
        torch.manual_seed(plan.get('periodic_evaluation', {}).get('seed', plan['optimizer']['seed']+1))
        for i in range(plan['budget']['posterior_samples']):
            candidates.append((f'VI_sample_{i:02}',q_vi.sample().detach()))
        with torch.no_grad():
            ratios = q_vi.std_ratio().cpu().numpy()
        np.save(artifacts/'std_ratio.npy',ratios)
        if isinstance(q_vi, GraphResidualPosterior):
            samples = q_vi.diagnostic_samples()
            np.savez(artifacts/'physical-posterior.npz', mean_Y=samples.mean(0).cpu().numpy(),
                     std_Y=samples.std(0).cpu().numpy(), median_Y=samples.median(0).values.cpu().numpy(),
                     q025_Y=torch.quantile(samples,.025,dim=0).cpu().numpy(),
                     q975_Y=torch.quantile(samples,.975,dim=0).cpu().numpy(), std_ratio=ratios,
                     estimator=np.array('32 reproducible joint samples'))
        if isinstance(q_vi, UniformStiffnessPosterior):
            with torch.no_grad():
                mean, std = q_vi.physical_moments()
                np.savez(artifacts/'physical-posterior.npz', mean_Y=mean.cpu().numpy(), std_Y=std.cpu().numpy(),
                         median_Y=q_vi.point_estimate().exp().cpu().numpy(),
                         q025_Y=q_vi.log_stiffness(q_vi.mu-1.95996398454*q_vi.std).exp().cpu().numpy(),
                         q975_Y=q_vi.log_stiffness(q_vi.mu+1.95996398454*q_vi.std).exp().cpu().numpy(),
                         std_ratio=ratios)
    rows = []
    flow_predictions = {}
    collect_pairs = plan.get('comparison', {}).get('type') == 'exploratory_joint_sample_diversity'
    if collect_pairs:
        np.savez(artifacts/'joint-samples.npz', **{name:z.cpu().numpy() for name,z in candidates if name.startswith('VI_sample')})
    for name,z in candidates:
        try:
            if collect_pairs:
                rollout.reset(z)
                losses, positions = [], []
                for frame in range(1,end):
                    if time.monotonic() >= deadline:
                        raise TimeoutError('Evaluation budget exhausted')
                    loss,_ = rollout.frame(frame,False)
                    losses.append(loss)
                    positions.append(rollout.sim.wp_states[-1].wp_x.numpy().copy())
                losses = np.asarray(losses)
                if name.startswith('VI_sample'):
                    flow_predictions[name] = np.stack(positions)
                    np.save(artifacts/f'{name}-positions.npy', flow_predictions[name])
            else:
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
    if collect_pairs:
        from .flow_analysis import compare_samples
        compare_samples(candidates, flow_predictions, rows, rollout.sim.object_springs, split, artifacts)
    plot_results(args, artifacts, rollout.sim.object_springs,
                 'Y' if isinstance(q_vi, (UniformStiffnessPosterior, GraphResidualPosterior)) else 'log stiffness')
    return dict(status='completed' if len(rows)==len(candidates) else 'budget_limited',
                evaluation_seconds=time.monotonic()-start, completed_rollouts=len(rows),
                requested_rollouts=len(candidates),results=rows)


def plot_results(args, artifacts: Path, object_springs: int, coordinates: str = "log stiffness"):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    if (artifacts/'std_ratio.npy').exists():
        ratios=np.load(artifacts/'std_ratio.npy')
        fig,ax=plt.subplots()
        for name,r in [('Object',ratios[:object_springs]),('Controller',ratios[object_springs:])]:
            ax.hist(r,bins=40,alpha=0.6,label=name)
        ax.set(xlabel=f'Posterior std / prior std ({coordinates})',ylabel='Spring count')
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
    parser.add_argument('--variant')
    args=parser.parse_args()
    if args.resume_checkpoint and args.mode == 'evaluate':
        parser.error('--resume-checkpoint requires training mode')
    plan,protocol=read_plan(args.config, args.variant)
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
