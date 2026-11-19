"""Run E-0013 sequentially and retain every technical outcome."""
import json
import subprocess
import sys
from pathlib import Path
import numpy as np
import yaml
from lab.calibration.phystwin_adapter import ROOT, sha256
from lab.calibration.phystwin_vi import read_plan, write_json
from run_update_comparison import clean_commit


def compare(completed, output):
    means, ratios, ids = [], [], []
    topology = None
    for plan, artifacts in completed:
        current = np.load(artifacts/'topology.npz')
        if topology is None:
            topology = {key:current[key].copy() for key in current.files}
        else:
            for key, value in topology.items():
                np.testing.assert_array_equal(value,current[key])
        with np.load(artifacts/'physical-posterior.npz') as data:
            means.append(data['mean_Y']);ratios.append(data['std_ratio'])
        ids.append(plan['run_ids']['train'])
    means=np.asarray(means);ratios=np.asarray(ratios)
    np.savez(output/'all-solutions.npz',run_ids=ids,mean_Y=means,std_ratio=ratios,edges=topology['edges'],object_springs=topology['object_springs'])
    n=len(ids)
    mean_distance=np.zeros((n,n));ratio_distance=np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            mean_distance[i,j]=np.abs(means[i]-means[j]).mean()
            ratio_distance[i,j]=np.abs(ratios[i]-ratios[j]).mean()
    write_json(output/'pairwise.json',dict(run_ids=ids,mean_Y_MAE=mean_distance.tolist(),std_ratio_MAE=ratio_distance.tolist(),interpretation='Descriptive distances; seed and initialization jointly varied'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(13,6),layout='constrained')
    for ax,values,title in zip(axes,[mean_distance,ratio_distance],['Mean stiffness MAE','Std / prior std MAE']):
        image=ax.imshow(values);fig.colorbar(image,ax=ax);ax.set_title(title)
        ax.set_xticks(range(n),ids,rotation=90);ax.set_yticks(range(n),ids)
    fig.savefig(output/'pairwise.png',dpi=150);plt.close(fig)


def main():
    suite=yaml.safe_load((ROOT/'experiments/E-0013.yaml').read_text())
    plans=[(ROOT/path,read_plan(ROOT/path)[0]) for path in suite['suite_plans']]
    root=Path(plans[0][1]['execution']['artifact_root']).expanduser()
    output=root/'E-0013-suite';output.mkdir(exist_ok=False)
    status={'experiment':'E-0013','status':'running','runs':[]}
    completed=[]
    try:
        for config,plan in plans:
            record={'train':plan['run_ids']['train'],'seed':plan['optimizer']['seed'],'initial_median_Y':plan['posterior']['initial_median_Y'],'status':'running'}
            status['runs'].append(record);write_json(output/'status.json',status)
            for mode,key in [('vi','train'),('evaluate','evaluate')]:
                # Every scientific run is guarded independently, including evaluations.
                commit=clean_commit()
                rid=plan['run_ids'][key];run_dir=output/'manifests'/rid
                command=[sys.executable,'-m','lab.calibration.phystwin_vi','--config',str(config),'--mode',mode,'--run-dir',str(run_dir)]
                if mode=='evaluate':command+=['--vi-checkpoint',str(root/plan['run_ids']['train']/'checkpoint.pt')]
                with (output/f'{rid}.log').open('w') as log:
                    result=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
                manifest_path=run_dir/'manifest.json'
                manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
                record[key]={'id':rid,'returncode':result.returncode,'manifest':str(manifest_path),'status':manifest.get('status','failed_before_manifest')}
                if result.returncode or manifest.get('status')!='completed':
                    record['status']='failed';record['failure_reason']=manifest.get('failure_reason','See process log');break
                if manifest['repository_commit']!=commit:raise RuntimeError('Worktree commit changed during run')
                record['status']='completed' if mode=='evaluate' else 'evaluating'
                write_json(output/'status.json',status)
            if record['status']=='completed':completed.append((plan,root/plan['run_ids']['evaluate']))
            write_json(output/'status.json',status)
        if completed:
            compare(completed,output)
            import comet_ml
            last_plan,_ = completed[-1]
            connection=json.loads((root/last_plan['run_ids']['train']/'comet.json').read_text())
            config=comet_ml.ExperimentConfig(log_code=False,log_graph=False,parse_args=False,
                log_git_metadata=False,log_git_patch=False,log_env_details=False,
                log_env_gpu=False,log_env_cpu=False,log_env_host=False,log_env_network=False,
                log_env_disk=False,auto_output_logging=False,auto_param_logging=False,
                auto_metric_logging=False,auto_log_co2=False)
            experiment=comet_ml.start(experiment_key=connection['key'],experiment_config=config)
            experiment.log_image(str(output/'pairwise.png'),name='multistart_pairwise_comparison')
            experiment.log_asset(str(output/'pairwise.json'))
            experiment.log_asset(str(output/'all-solutions.npz'))
            experiment.end()
        status['status']='completed' if len(completed)==len(plans) else 'completed_with_failures'
    except BaseException as exc:
        status.update(status='interrupted_or_failed',failure_reason=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write_json(output/'status.json',status)


if __name__=='__main__':
    main()
