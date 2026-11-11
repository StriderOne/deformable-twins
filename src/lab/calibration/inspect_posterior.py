"""Export every spring marginal, with an interactive selector; no new simulation."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
from statistics import NormalDist

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

from .phystwin_adapter import ROOT, sha256


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--run-manifest',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    manifest=json.loads(args.run_manifest.read_text())
    expected=manifest['artifacts'][str(args.checkpoint.resolve())]
    if sha256(args.checkpoint)!=expected:
        raise ValueError('Checkpoint hash differs from training manifest')
    args.output.mkdir(parents=True,exist_ok=False)
    state=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    q=state['posterior']
    mu=q['mu'].numpy().astype(float)
    sigma=torch.nn.functional.softplus(q['rho']).numpy().astype(float)
    prior_mu=q['prior_mu'].numpy().astype(float)
    prior_std=q['prior_std'].numpy().astype(float)
    n=int(state['metadata']['object_springs'])
    count=len(mu);ids=np.arange(count)
    assert 0<n<count and np.isfinite(mu).all() and np.isfinite(sigma).all() and (sigma>0).all()
    z95=NormalDist().inv_cdf(0.95)
    lower=np.exp(mu-z95*sigma);upper=np.exp(mu+z95*sigma);median=np.exp(mu)
    with (args.output/'all-springs.csv').open('w') as f:
        w=csv.writer(f)
        w.writerow(['spring_id','group','mu_log_Y','std_log_Y','prior_std_log_Y','std_ratio',
                    'median_Y_N','q05_Y_N','q95_Y_N'])
        for i in ids:
            w.writerow([int(i),'object' if i<n else 'controller',mu[i],sigma[i],prior_std[i],
                        sigma[i]/prior_std[i],median[i],lower[i],upper[i]])
    groups=[('Object',slice(0,n),'#2364aa'),('Controller',slice(n,None),'#e97632')]
    low=min(sigma.min(),prior_std.min());high=max(sigma.max(),prior_std.max())
    padding=max((high-low)*0.08,high*0.01)
    limits=(low-padding,high+padding)
    fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
    for col,(name,s,c) in enumerate(groups):
        axes[0,col].scatter(ids[s],sigma[s],s=2 if col==0 else 7,alpha=.6,color=c,rasterized=True)
        axes[0,col].axhline(prior_std[s][0],color='black',ls='--',lw=1,label='Prior std')
        axes[0,col].set(title=f'{name}: every spring ({len(ids[s]):,})',xlabel='Original spring ID',ylabel='Posterior std of log(Y)',ylim=limits)
        axes[0,col].legend()
        axes[1,col].hist(sigma[s],bins=60,color=c,alpha=.85)
        axes[1,col].axvline(prior_std[s][0],color='black',ls='--',lw=1)
        axes[1,col].set(xlabel='Posterior std of log(Y)',ylabel='Spring count',xlim=limits)
    fig.suptitle(f"{manifest['id']}: per-spring uncertainty, no averaging across springs")
    fig.savefig(args.output/'all-springs.png',dpi=160);plt.close(fig)

    # Every point is present. Clicking or entering an ID shows that exact marginal.
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,subplot_titles=['Posterior std of log(Y), every spring','Posterior mu of log(Y), every spring'])
    for name,s,c in groups:
        for row,values in [(1,sigma),(2,mu)]:
            fig.add_trace(go.Scattergl(x=ids[s].tolist(),y=values[s].tolist(),mode='markers',
                marker=dict(size=3,color=c),name=name,legendgroup=name,showlegend=row==1,
                hovertemplate='Spring %{x}<br>value %{y:.6f}<extra>'+name+'</extra>'),row=row,col=1)
    fig.update_layout(height=650,title=f"{manifest['id']} — click any spring, or enter its ID below")
    fig.update_xaxes(title_text='Original spring ID',row=2,col=1)
    scatter=fig.to_html(full_html=False,include_plotlyjs=True,div_id='springs')
    payload=json.dumps(dict(mu=mu.tolist(),std=sigma.tolist(),pm=prior_mu.tolist(),ps=prior_std.tolist(),n=n))
    script=r'''
<script>
const data=PAYLOAD;
const input=document.getElementById('spring-id');
function selectSpring(value){
 const i=Math.max(0,Math.min(data.mu.length-1,Math.round(Number(value)||0)));input.value=i;
 const m=data.mu[i],s=data.std[i],pm=data.pm[i],ps=data.ps[i];
 const lo=Math.min(m-4*s,pm-4*ps),hi=Math.max(m+4*s,pm+4*ps);
 const x=Array.from({length:401},(_,j)=>lo+(hi-lo)*j/400);
 const pdf=(mean,sd)=>x.map(z=>Math.exp(-0.5*((z-mean)/sd)**2)/(sd*Math.sqrt(2*Math.PI)));
 Plotly.react('density',[
  {x,y:pdf(pm,ps),name:'Prior',mode:'lines',line:{color:'#777',dash:'dash'}},
  {x,y:pdf(m,s),name:'Posterior',mode:'lines',line:{color:'#2364aa'}}
 ],{title:`Spring ${i} (${i<data.n?'object':'controller'}): exact marginal`,
 xaxis:{title:{text:'log(Y), before simulator clamp'}},yaxis:{title:{text:'Probability density'}},height:400});
 document.getElementById('values').textContent=`mu=${m.toFixed(6)}, std=${s.toFixed(6)}, std/prior=${(s/ps).toFixed(6)}; median Y=${Math.exp(m).toFixed(1)} N`;
}
input.addEventListener('input',()=>selectSpring(input.value));
document.getElementById('springs').on('plotly_click',e=>selectSpring(e.points[0].x));
selectSpring(0);
</script>'''.replace('PAYLOAD',payload)
    html='<!doctype html><html><head><meta charset="utf-8"><title>All spring posteriors</title></head><body style="font:16px sans-serif;max-width:1400px;margin:auto">'
    html+='<h1>Every spring has its own Gaussian in log stiffness</h1><p>These are approximate latent marginals, not calibrated physical confidence intervals. Correlations are not represented. The simulator clamps stiffness after exponentiation.</p>'
    html+=scatter+f'<label>Spring ID (0–{count-1}): <input id="spring-id" type="number" min="0" max="{count-1}" value="0"></label><p id="values"></p><div id="density"></div>'+script+'</body></html>'
    (args.output/'all-springs.html').write_text(html)
    summary={}
    for name,s,_ in groups:
        a=sigma[s]
        summary[name.lower()]=dict(count=len(a),std_quantiles=dict(zip(['min','p05','median','p95','max'],np.quantile(a,[0,.05,.5,.95,1]).tolist())))
    result=dict(kind='checkpoint_inspection_no_training',source_run=manifest['id'],
        training_commit=manifest['repository_commit'],
        analysis_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        checkpoint_sha256=expected,run_manifest_sha256=sha256(args.run_manifest),
        analysis_source_sha256=sha256(Path(__file__)),spring_count=count,summary=summary,
        artifacts={str(f):sha256(f) for f in args.output.iterdir() if f.is_file()})
    # Generated data cover all IDs without decimation; PNG display alone cannot resolve all points.
    with (args.output/'all-springs.csv').open() as f:
        rows=list(csv.DictReader(f));assert len(rows)==count
        assert int(rows[0]['spring_id'])==0 and int(rows[-1]['spring_id'])==count-1
    (args.output/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
