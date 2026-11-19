"""Execute the approved E-0012 pilot from a clean worktree."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
from lab.calibration.phystwin_adapter import ROOT, sha256
from lab.calibration.phystwin_vi import read_plan, write_json


def clean_commit():
    status = subprocess.check_output(['git','status','--porcelain=v1','--untracked-files=all','--ignore-submodules=none'],cwd=ROOT,text=True)
    if status:
        raise RuntimeError('Scientific run requires clean worktree:\n'+status)
    return subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()


def main():
    plans = [ROOT/'experiments/E-0012/update-1.yaml',ROOT/'experiments/E-0012/update-7.yaml',ROOT/'experiments/E-0012.yaml']
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path)
    parser.add_argument('--output-name', default='E-0012-comparison-attempt2')
    args = parser.parse_args()
    if args.config:
        plans = [args.config.resolve()]
    resolved = [read_plan(p)[0] for p in plans]
    root = Path(resolved[0]['execution']['artifact_root']).expanduser()
    reference = root/'R-0030/std_ratio.npy'
    reference_topology = root/'R-0030/topology.npz'
    reference_ratios = np.load(reference)
    output = root/args.output_name
    output.mkdir(exist_ok=False)
    rows = []
    for config, plan in zip(plans, resolved):
        ids = plan['run_ids']
        for mode, key in [('vi','train'),('evaluate','evaluate')]:
            rid = ids[key]
            commit = clean_commit()
            run_dir = output/'manifests'/rid
            cmd = [sys.executable,'-m','lab.calibration.phystwin_vi','--config',str(config),'--run-dir',str(run_dir),'--mode',mode]
            if mode == 'vi' and plan.get('continuation'):
                resume = Path(plan['continuation']['checkpoint'])
                if sha256(resume) != plan['continuation']['checkpoint_sha256']:
                    raise ValueError('Resume checkpoint hash mismatch')
                cmd += ['--resume-checkpoint', str(resume)]
            if mode == 'evaluate':
                cmd += ['--vi-checkpoint',str(root/ids['train']/'checkpoint.pt')]
            with (output/f'{rid}.log').open('w') as log:
                subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            manifest = json.loads((run_dir/'manifest.json').read_text())
            if manifest['repository_commit'] != commit or manifest['status'] != 'completed':
                raise RuntimeError(f'Incomplete run or commit changed: {rid}')
        target = root/ids['evaluate']
        with np.load(reference_topology) as a, np.load(target/'topology.npz') as b:
            for key in a.files:
                np.testing.assert_array_equal(a[key],b[key])
            n_object = int(a['object_springs'])
        ratios = np.load(target/'std_ratio.npy')
        np.testing.assert_equal(ratios.shape,reference_ratios.shape)
        delta = ratios-reference_ratios
        np.savez(output/f"{ids['train']}-paired-ratios.npz",reference=reference_ratios,candidate=ratios,difference=delta)
        groups = {}
        for name, values in [('all',delta),('object',delta[:n_object]),('controller',delta[n_object:])]:
            absolute = np.abs(values)
            groups[name] = dict(count=len(values),mean_absolute_difference=float(absolute.mean()),median_absolute_difference=float(np.median(absolute)),p95_absolute_difference=float(np.quantile(absolute,.95)),maximum_absolute_difference=float(absolute.max()))
        train = json.loads((output/'manifests'/ids['train']/'manifest.json').read_text())
        rows.append(dict(interval=plan['optimizer']['update_every_frames'],train=ids['train'],evaluate=ids['evaluate'],training_seconds=train['training_seconds'],ratio_differences=groups,results=manifest['results']))
        write_json(output/'comparison.json',dict(reference_train='R-0029',reference_evaluate='R-0030',reference_ratio_sha256=sha256(reference),results=rows))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1,len(rows),figsize=(5*len(rows),4),sharex=True,sharey=True, squeeze=False)
    axes = axes.ravel()
    for ax,row in zip(axes,rows):
        data = np.load(output/f"{row['train']}-paired-ratios.npz")
        ax.hexbin(data['reference'],data['candidate'],gridsize=60,mincnt=1,bins='log')
        ax.plot([0,1.8],[0,1.8],color='red',linewidth=1)
        ax.set(title=f"Update every {row['interval']} frames",xlabel='E-0011 posterior / prior std')
    axes[0].set_ylabel('Candidate posterior / prior std')
    fig.tight_layout()
    fig.savefig(output/'paired-ratios.png',dpi=160)


if __name__ == '__main__':
    main()
