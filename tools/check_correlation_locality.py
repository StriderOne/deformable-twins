"""Exploratory uniform-reference locality audit; no selection on correlation."""
import json,hashlib,pickle
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
BASE=Path('/home/homa/.local/share/deformable-twins/artifacts')
OUT=BASE/'R-0029-correlation-locality'
PLAN=dict(run='R-0029',protocol='P-0005',seed=42,reference_count=512,far_samples=128,far_distance_quantile=.75,quadrature_order=32,selection='uniform object spring IDs without replacement, before measuring correlations',neighbor='shares at least one vertex; excludes self',metric='mean absolute physical Pearson correlation; neighbors minus far',interpretation='exploratory within one fitted posterior; no independent replication; selected best examples are not representative')

def main():
    OUT.mkdir(exist_ok=False);(OUT/'plan.json').write_text(json.dumps(PLAN,indent=2))
    m=json.loads((ROOT/'runs/R-0029/manifest.json').read_text());ev=json.loads((ROOT/'runs/R-0030/manifest.json').read_text());p=BASE/'R-0029/checkpoint.pt';top=BASE/'R-0030/topology.npz';data=Path(m['config']['data']['observations']);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(p)==m['artifacts'][str(p)] and sha(top)==ev['artifacts'][str(top)] and sha(data)==m['inputs'][str(data)]
    q=torch.load(p,map_location='cpu',weights_only=False)['posterior'];mu=q['mu'].double().numpy();f=q['factor'].double().numpy();d=torch.nn.functional.softplus(q['rho'].double()).numpy();sd=np.sqrt(d*d+(f*f).sum(1));a=f/sd[:,None]
    t=np.load(top);n=int(t['object_springs']);edges=t['edges'][:n].astype(int)
    with data.open('rb') as stream:r=pickle.load(stream)
    pts=np.concatenate([r['object_points'][0],r['surface_points'],r['interior_points']]);mid=pts[edges].mean(1)
    adjacency=[[] for _ in pts]
    for i,(v,w) in enumerate(edges):adjacency[v].append(i);adjacency[w].append(i)
    rng=np.random.default_rng(PLAN['seed']);refs=rng.choice(n,PLAN['reference_count'],replace=False)
    rows=[];pairs=[];group=[];owners=[]
    for idx,ref in enumerate(refs):
        near=np.setdiff1d(np.union1d(adjacency[edges[ref,0]],adjacency[edges[ref,1]]),[ref])
        dist=np.linalg.norm(mid-mid[ref],axis=1);cut=np.quantile(np.delete(dist,ref),.75);pool=np.flatnonzero(dist>=cut);pool=np.setdiff1d(pool,np.append(near,ref));far=rng.choice(pool,PLAN['far_samples'],replace=False)
        ids=np.concatenate([near,far]);pairs.extend(zip(np.full(len(ids),ref),ids));group.extend([0]*len(near)+[1]*len(far));owners.extend([idx]*len(ids));rows.append(dict(reference=int(ref),neighbors=len(near),far_cutoff_m=float(cut)))
    pairs=np.array(pairs);owners=np.array(owners);group=np.array(group)
    def compute(order):
        x,w=np.polynomial.hermite.hermgauss(order);x*=np.sqrt(2);w/=np.sqrt(np.pi)
        vals=expit(mu[:,None]+sd[:,None]*x);means=vals@w;stds=np.sqrt(((vals-means[:,None])**2)@w);result=np.empty(len(pairs))
        for start in range(0,len(pairs),256):
            sl=slice(start,start+256);i,j=pairs[sl].T;rho=np.sum(a[i]*a[j],axis=1);rho=rho[:,None,None]
            y=expit(mu[j,None,None]+sd[j,None,None]*(rho*x[None,:,None]+np.sqrt(1-rho*rho)*x[None,None,:]))
            cov=np.einsum('bij,bi,j->b',y-means[j,None,None],(vals[i]-means[i,None])*w,w)
            result[sl]=cov/(stds[i]*stds[j])
        return result
    corr=compute(32);check=compute(48);err=float(abs(corr-check).max());assert err<.002;corr=check
    for idx,row in enumerate(rows):
        near=np.abs(corr[(owners==idx)&(group==0)]);far=np.abs(corr[(owners==idx)&(group==1)])
        row.update(near_mean=float(near.mean()),far_mean=float(far.mean()),difference=float(near.mean()-far.mean()),near_max=float(near.max()),far_max=float(far.max()))
    diff=np.array([r['difference'] for r in rows]);near=np.array([r['near_mean'] for r in rows]);far=np.array([r['far_mean'] for r in rows]);best=sorted(rows,key=lambda r:r['difference'],reverse=True)[:5]
    record=dict(plan=PLAN,training_commit=m['repository_commit'],quadrature_max_error=err,reference_fraction_neighbor_stronger=float((diff>0).mean()),mean_neighbor=float(near.mean()),mean_far=float(far.mean()),median_difference=float(np.median(diff)),positive_reference_count=int((diff>0).sum()),best_selected_examples=best,inputs={str(k):sha(k) for k in [p,top,data]},script_sha256=sha(Path(__file__)))
    (OUT/'summary.json').write_text(json.dumps(record,indent=2));(OUT/'references.json').write_text(json.dumps(rows,indent=2));np.savez_compressed(OUT/'pairs.npz',pairs=pairs,physical_correlation=corr,group=group,reference_index=owners)
    fig,axs=plt.subplots(1,2,figsize=(11,4.5),layout='constrained');axs[0].scatter(far,near,s=10,alpha=.5);lim=max(near.max(),far.max());axs[0].plot([0,lim],[0,lim],'k--');axs[0].set(xlabel='Far: mean absolute correlation',ylabel='Neighbors: mean absolute correlation',title='512 uniformly selected reference springs')
    axs[1].hist(diff,bins=45);axs[1].axvline(0,color='black',ls='--');axs[1].set(xlabel='Neighbor mean − far mean |correlation|',ylabel='Reference count',title='Positive = stronger local dependence');fig.savefig(OUT/'locality.png',dpi=170)
    print(json.dumps(record,indent=2))
if __name__=='__main__':main()
