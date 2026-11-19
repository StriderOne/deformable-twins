"""Plot immutable output of lab.calibration.spring_loss_probe."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('artifacts',type=Path);p.add_argument('output',type=Path)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
rows=[json.loads(line) for line in (a.artifacts/'samples.jsonl').read_text().splitlines()]
selection=json.loads((a.artifacts/'selection.json').read_text())
config=json.loads((a.artifacts/'resolved-plan.json').read_text())['experiment']['probe']
for label in selection:
    for kind, key in [('posterior','posterior_samples'),('grid','grid_points')]:
        assert sum(r['kind']==kind and r['label']==label for r in rows)==config[key], 'Incomplete probe'
assert all(np.isfinite(r['nll']) for r in rows), 'Nonfinite loss'
base=np.array([r['nll'] for r in rows if r['kind']=='baseline']);ref=base.mean()
colors={'near_one':'#377eb8','minimum':'#e34a33'}
fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
for col,label in enumerate(colors):
    color=colors[label];sel=selection[label]
    samples=[r for r in rows if r['label']==label and r['kind']=='posterior']
    grid=[r for r in rows if r['label']==label and r['kind']=='grid']
    losses=np.array([r['nll'] for r in samples])-ref
    ax=axes[0,col];ax.hist(losses,bins=16,color=color,alpha=.8,edgecolor='white',density=True)
    ax.set_title(f"Spring {sel['spring_id']} | std / prior std = {sel['std_ratio']:.3f}")
    ax.set_ylabel('Empirical density');ax.set_xlabel('Change in tracking NLL from fixed baseline')
    ax.axvline(0,color='black',ls='--',lw=1)
    ax.axvspan(base.min()-ref,base.max()-ref,color='gray',alpha=.25)
    ax=axes[1,col]
    ax.plot([r['Y']/1000 for r in grid],[r['nll']-ref for r in grid],color=color,lw=1.5,label='Common stiffness grid')
    ax.scatter([r['Y']/1000 for r in samples],losses,color=color,s=18,alpha=.65,label='Posterior samples')
    ax.axvline(sel['median_Y']/1000,color=color,ls=':',label='Posterior median')
    ax.axhline(0,color='black',ls='--',lw=1)
    ax.axhspan(base.min()-ref,base.max()-ref,color='gray',alpha=.3,label='Repeated baseline range')
    ax.set_xlabel('Stiffness Y / 1000');ax.set_ylabel('Change in tracking NLL');ax.legend(fontsize=8)
    for ax in axes[:,col]:ax.grid(alpha=.15)
fig.suptitle(f'R-0024 | Individual spring perturbations, others fixed at posterior medians\nBaseline NLL = {ref:.3f}; full training rollout, no KL',fontsize=13)
fig.savefig(a.output/'loss-distributions.png',dpi=170)
# Overlay uses shared axes so sensitivity magnitudes can be compared directly.
fig,ax=plt.subplots(figsize=(10,5),layout='constrained')
for label,color in colors.items():
    grid=[r for r in rows if r['label']==label and r['kind']=='grid']
    ax.plot([r['Y']/1000 for r in grid],[r['nll']-ref for r in grid],'.-',color=color,label=f"Spring {selection[label]['spring_id']}, ratio {selection[label]['std_ratio']:.3f}")
ax.axhspan(base.min()-ref,base.max()-ref,color='gray',alpha=.4,label='Baseline numerical range')
ax.axhline(0,color='black',ls='--',lw=1);ax.set(xlabel='Stiffness Y / 1000',ylabel='Change in tracking NLL',title='Same stiffness range, same frozen parameters | R-0024')
ax.legend();ax.grid(alpha=.2);fig.savefig(a.output/'shared-grid.png',dpi=170)
