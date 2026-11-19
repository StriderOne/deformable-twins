"""Explicit Comet logging and paired physical-posterior views."""
import json
import subprocess
from pathlib import Path
import numpy as np
import torch
from .vi_monitor import Monitor


class MultistartMonitor(Monitor):
    def __init__(self, plan, artifacts, q, object_springs):
        import comet_ml
        self.artifacts = artifacts
        config = comet_ml.ExperimentConfig(
            name=artifacts.name, log_code=False, log_graph=False, parse_args=False,
            log_git_metadata=False, log_git_patch=False, log_env_details=False,
            log_env_gpu=False, log_env_cpu=False, log_env_host=False,
            log_env_network=False, log_env_disk=False, auto_output_logging=False,
            auto_param_logging=False, auto_metric_logging=False, auto_log_co2=False,
            offline_directory=str(artifacts/'comet-offline'))
        self.comet = comet_ml.start(project_name=plan['comet']['project'], experiment_config=config)
        if self.comet is None:
            raise RuntimeError('Comet did not create an experiment')
        self.comet.log_parameters({
            'experiment': plan['id'], 'run': artifacts.name,
            'initial_median_Y': plan['posterior']['initial_median_Y'],
            'seed': plan['optimizer']['seed'], 'passes': plan['optimizer']['max_iterations'],
            'update_every_frames': plan['optimizer']['update_every_frames'],
            'repository_commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=Path(__file__).resolve().parents[3],text=True).strip()})
        (artifacts/'comet.json').write_text(json.dumps({'url':self.comet.url,'key':self.comet.get_key()},indent=2)+'\n')
        super().__init__(plan, artifacts, q, object_springs)

    def update(self, row, q):
        super().update(row, q)
        metrics = {f'train/{k}':v for k,v in row.items() if isinstance(v,(int,float)) and k!='iteration'}
        for group, stats in row['std'].items():
            metrics.update({f'std/{group}/{k}':v for k,v in stats.items()})
        self.comet.log_metrics(metrics, step=row['iteration'])

    def evaluation(self, result, step):
        values = {}
        for name, splits in result.items():
            for split, metrics in splits.items():
                for key,value in metrics.items():
                    values[f'eval/{name}/{split}/{key}'] = value
        for key,value in values.items():
            self.writer.add_scalar(key,value,step)
        self.writer.flush()
        self.comet.log_metrics(values,step=step)

    @torch.no_grad()
    def draw(self, q, step):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import TwoSlopeNorm, LogNorm
        mean,std = q.physical_moments()
        mean = mean.cpu().numpy()
        ratio = q.std_ratio().cpu().numpy()
        np.savez(self.artifacts/f'posterior-{step:06d}.npz',mean_Y=mean,std_Y=std.cpu().numpy(),std_ratio=ratio)
        fig,axes=plt.subplots(3,2,figsize=(12,13),layout='constrained')
        order=np.argsort(abs(ratio[:self.n]-1))
        for i,pair in enumerate([(0,1),(0,2),(1,2)]):
            for j,(values,cmap,norm) in enumerate([
                (ratio,'coolwarm_r',TwoSlopeNorm(vmin=0,vcenter=1,vmax=1.75)),
                (mean,'viridis',LogNorm(vmin=100,vmax=100000))]):
                ax=axes[i,j]
                lines=LineCollection(self.segments[order][:,:,pair],cmap=cmap,norm=norm,linewidths=.3,rasterized=True)
                lines.set_array(values[:self.n][order]);ax.add_collection(lines)
                ax.autoscale_view();ax.set_aspect('equal')
                ax.set_title(('Posterior std / prior std' if j==0 else 'Posterior mean Y')+f' | view {pair}')
                fig.colorbar(lines,ax=ax)
        fig.suptitle(f'{self.artifacts.name} | pass {step} | object springs')
        image=self.artifacts/f'springs-{step:06d}.png'
        fig.savefig(image,dpi=120)
        self.writer.add_figure('mesh/std_and_mean',fig,step,close=True)
        self.writer.add_histogram('std/object_ratio',ratio[:self.n],step)
        self.writer.add_histogram('mean/object_Y',mean[:self.n],step)
        self.writer.flush()
        self.comet.log_image(str(image),name=f'{self.artifacts.name}_pass_{step:04d}_std_and_mean',step=step)

    def close(self):
        super().close()
        self.comet.end()


@torch.no_grad()
def evaluate_current(rollout, q, plan, protocol, artifacts, step, deadline):
    """Evaluate complete fixed-parameter trajectories without consuming training RNG."""
    from .phystwin_vi import metrics
    cfg=plan['periodic_evaluation']
    end=protocol['splits']['continuation_targets'][1]
    split=protocol['splits']['calibration_targets'][1]
    rows=[]
    with torch.random.fork_rng(devices=[q.mu.device.index] if q.mu.is_cuda else []):
        torch.manual_seed(cfg['seed'])
        candidates=[q.point_estimate()]+[q.sample() for _ in range(cfg['posterior_samples'])]
        for z in candidates:
            losses,_=rollout.trajectory(z,end,False,deadline)
            rows.append(metrics(losses,rollout.sim.valid_counts,plan['likelihood']['sigma_obs_m'],split))
    result={'median':rows[0]}
    if len(rows)>1:
        result['sample_mean']={split:{key:float(np.mean([row[split][key] for row in rows[1:]]))
                                     for key in ('nll_without_constant','rmse_m')}
                               for split in ('calibration','continuation')}
    with (artifacts/'periodic-evaluation.jsonl').open('a') as f:
        f.write(json.dumps({'iteration':step,'result':result,'samples':rows},allow_nan=False)+'\n')
    return result
