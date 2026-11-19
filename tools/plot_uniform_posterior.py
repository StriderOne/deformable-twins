"""Freeze a running checkpoint and plot physical marginals for every spring."""
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
from datetime import datetime, timezone

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import expit
import torch

ROOT=Path(__file__).resolve().parents[1]


def main():
    run='R-0020'
    manifest=json.loads((ROOT/'runs'/run/'manifest.json').read_text())
    source=Path(manifest['artifact_directory'])/'checkpoint.pt'
    raw=source.read_bytes()  # Atomic replacement by training: one complete version.
    state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=False)
    assert state['metadata']['config']['posterior']['family']=='logistic_normal_stiffness'
    step=state['iteration']
    out=source.parent.parent/f'{run}-distributions-i{step:06d}'
    out.mkdir(exist_ok=False)
    (out/'checkpoint.pt').write_bytes(raw)
    p=state['posterior'];mu=p['mu'].double().numpy()
    sd=torch.nn.functional.softplus(p['rho'].double()).numpy()
    lo=float(p['lower']);hi=float(p['upper']);width=hi-lo;prior_sd=width/math.sqrt(12)
    n=state['metadata']['object_springs'];ids=np.arange(len(mu))
    nodes=p['nodes'].double().numpy();weights=p['weights'].double().numpy()
    mean=np.empty(len(mu));std=mean.copy()
    for start in range(0,len(mu),2048):
        sl=slice(start,start+2048)
        f=expit(mu[sl][None,:]+sd[sl][None,:]*nodes)
        m=(f*weights).sum(0)
        mean[sl]=lo+width*m
        std[sl]=width*np.sqrt(((f-m)**2*weights).sum(0))
    median=lo+width*expit(mu)
    low=lo+width*expit(mu-1.95996398454*sd)
    high=lo+width*expit(mu+1.95996398454*sd)
    assert np.isfinite(std).all() and np.all((low>=lo)&(high<=hi))
    with (out/'all-springs.csv').open('w') as f:
        w=csv.writer(f);w.writerow(['spring_id','group','mean_Y','median_Y','std_Y','prior_std_Y','std_ratio','q025_Y','q975_Y','mu_logit','std_logit'])
        for i in ids:w.writerow([i,'object' if i<n else 'controller',mean[i],median[i],std[i],prior_sd,std[i]/prior_sd,low[i],high[i],mu[i],sd[i]])
    colors=['#2267a8','#dc7b24'];groups=[('Object',slice(None,n)),('Controller',slice(n,None))]
    fig,ax=plt.subplots(2,2,figsize=(15,8),gridspec_kw={'width_ratios':[3,1]},layout='constrained')
    for (name,sl),color in zip(groups,colors):
        ax[0,0].scatter(ids[sl],median[sl]/1000,s=1,alpha=.45,color=color,label=name,rasterized=True)
        ax[1,0].scatter(ids[sl],std[sl]/prior_sd,s=1,alpha=.45,color=color,rasterized=True)
        ax[0,1].hist(median[sl]/1000,bins=60,histtype='step',density=True,color=color,label=name)
        ax[1,1].hist(std[sl]/prior_sd,bins=60,histtype='step',density=True,color=color)
    ax[0,0].axhline((lo+hi)/2000,color='black',ls='--',lw=1,label='Prior median')
    ax[1,0].axhline(1,color='black',ls='--',lw=1,label='Prior width')
    ax[0,0].set(ylabel='Posterior median Y / 1000',xlabel='Original spring ID',ylim=(0,hi/1000));ax[0,0].legend(markerscale=5)
    ax[1,0].set(ylabel='Posterior std(Y) / prior std(Y)',xlabel='Original spring ID');ax[1,0].legend()
    ax[0,1].set(xlabel='Posterior median Y / 1000',ylabel='Density across springs');ax[0,1].legend()
    ax[1,1].set(xlabel='Posterior std / prior std',ylabel='Density across springs')
    fig.suptitle(f'{run} | iteration {step} | all {len(mu):,} springs | physical Y\nUniform prior: [{lo:g}, {hi:g}] — intermediate checkpoint',fontsize=14)
    fig.savefig(out/'all-springs.png',dpi=160);plt.close(fig)
    grid=np.linspace(lo+width*1e-5,hi-width*1e-5,3000)
    t=(grid-lo)/width;u=np.log(t)-np.log1p(-t)
    fig,ax=plt.subplots(2,3,figsize=(15,8),layout='constrained');selected=[]
    for row,(name,sl) in enumerate(groups):
        ix=ids[sl];order=ix[np.argsort(std[sl],kind='stable')]
        choices=[order[0],order[len(order)//2],order[-1]]
        for col,(i,label) in enumerate(zip(choices,['Smallest std','Median-ranked std','Largest std'])):
            density=np.exp(-.5*((u-mu[i])/sd[i])**2)/(sd[i]*math.sqrt(2*math.pi)*width*t*(1-t))
            a=ax[row,col];a.plot(grid/1000,density*1000,color=colors[row],label='Posterior')
            a.axhline(1000/width,color='black',ls='--',label='Uniform prior')
            a.axvspan(low[i]/1000,high[i]/1000,color=colors[row],alpha=.12,label='95% posterior interval')
            a.set(title=f'{name} #{i}: {label}\nstd/prior std = {std[i]/prior_sd:.3f}',xlabel='Stiffness Y / 1000',ylabel='Probability density per 1000 Y',xlim=(0,hi/1000))
            a.legend(fontsize=8);selected.append(int(i))
    fig.suptitle(f'{run} | iteration {step} | individual spring marginals\nSelection by std rank within each group; not randomly selected',fontsize=14)
    fig.savefig(out/'spring-marginals.png',dpi=160);plt.close(fig)
    sha=lambda b:hashlib.sha256(b).hexdigest()
    result={'run':run,'iteration':step,'kind':'intermediate_visualization_not_final_finding',
            'created_at':datetime.now(timezone.utc).isoformat(),'training_commit':manifest['repository_commit'],
            'analysis_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
            'script_sha256':sha(Path(__file__).read_bytes()),'checkpoint_sha256':sha(raw),
            'prior_bounds':[lo,hi],'prior_std_Y':prior_sd,'selected_ids':selected,'groups':{}}
    for name,sl in groups:
        result['groups'][name]={'count':len(ids[sl]),'mean_std_Y':float(std[sl].mean()),'median_std_ratio':float(np.median(std[sl]/prior_sd)),'mean_std_ratio':float((std[sl]/prior_sd).mean())}
    result['artifacts']={str(p):sha(p.read_bytes()) for p in out.iterdir()}
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    record=ROOT/'runs'/f'{run}-distributions-i{step:06d}.json'
    with record.open('x') as f:json.dump(result,f,indent=2)
    print(out)
    print(json.dumps(result['groups']))


if __name__=='__main__':main()
