"""Analyze one row of the learned covariance without allocating a dense matrix."""
import hashlib
import json
import pickle
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm
import plotly.graph_objects as go

ROOT=Path(__file__).resolve().parents[1]
REF=109751

def physical_corr(mu,sd,rho,ref,order):
    nodes,weights=np.polynomial.hermite.hermgauss(order)
    nodes=nodes*np.sqrt(2);weights=weights/np.sqrt(np.pi)
    values=expit(mu[:,None]+sd[:,None]*nodes)
    mean=values@weights
    std=np.sqrt(((values-mean[:,None])**2)@weights)
    ref_centered=values[ref]-mean[ref]
    result=np.empty(len(mu))
    for start in range(0,len(mu),256):
        sl=slice(start,start+256);r=rho[sl,None,None]
        v=expit(mu[sl,None,None]+sd[sl,None,None]*(r*nodes[None,:,None]+np.sqrt(np.maximum(0,1-r*r))*nodes[None,None,:]))
        cov=np.einsum('bij,i,j->b',v-mean[sl,None,None],weights*ref_centered,weights)
        result[sl]=cov/(std[ref]*std[sl])
    result[ref]=1
    return result,std

def main():
    manifests={rid:json.loads((ROOT/'runs'/rid/'manifest.json').read_text()) for rid in ['R-0029','R-0030']}
    train=manifests['R-0029'];ev=manifests['R-0030']
    base=Path(ev['artifact_directory']);checkpoint=Path(train['artifact_directory'])/'checkpoint.pt'
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for p,m in [(checkpoint,train),(base/'topology.npz',ev)]:assert sha(p)==m['artifacts'][str(p)]
    state=torch.load(checkpoint,map_location='cpu',weights_only=False)['posterior']
    mu=state['mu'].double().numpy();factor=state['factor'].double().numpy();d=torch.nn.functional.softplus(state['rho'].double()).numpy()
    variance=d*d+(factor*factor).sum(1);sd=np.sqrt(variance)
    covariance=factor@factor[REF];covariance[REF]+=d[REF]**2
    latent=covariance/(sd*sd[REF]);np.testing.assert_allclose(latent[REF],1)
    # Dense small-block verification of the determinant-free covariance row.
    ids=np.array([0,1,REF,len(mu)-1]);dense=np.diag(d[ids]**2)+factor[ids]@factor[ids].T
    np.testing.assert_allclose(covariance[ids],dense[:,2])
    physical,fracstd=physical_corr(mu,sd,latent,REF,32)
    check,_=physical_corr(mu,sd,latent,REF,48)
    error=float(abs(check-physical).max());assert error<.002,error
    physical=check
    zero,_=physical_corr(mu[:2],sd[:2],np.array([1.,0.]),0,32);assert abs(zero[1])<1e-10
    topo=np.load(base/'topology.npz');n=int(topo['object_springs']);edges=topo['edges'][:n].astype(int)
    data=Path(train['config']['data']['observations']);assert sha(data)==train['inputs'][str(data)]
    with data.open('rb') as f:raw=pickle.load(f)
    points=np.concatenate([raw['object_points'][0],raw['surface_points'],raw['interior_points']])
    np.testing.assert_allclose(np.linalg.norm(points[edges[:,0]]-points[edges[:,1]],axis=1),topo['rest'][:n],rtol=1e-4,atol=1e-7)
    mid=points[edges].mean(1);distance=np.linalg.norm(mid-mid[REF],axis=1)
    neighbor=np.isin(edges,edges[REF]).any(1);neighbor[REF]=False
    other=np.arange(n)!=REF
    out=base.parent/f'R-0029-correlations-ref{REF}';out.mkdir(exist_ok=False)
    np.savez(out/'covariance-row.npz',latent_covariance=covariance,latent_correlation=latent,physical_correlation=physical)
    np.savetxt(out/'object-springs.csv',np.column_stack([np.arange(n),edges,distance,neighbor,latent[:n],physical[:n]]),delimiter=',',header='spring_id,vertex_a,vertex_b,distance_m,shares_vertex,latent_correlation,physical_correlation',comments='')
    limit=max(float(abs(physical[:n][other]).max()),1e-6)
    _,_,rot=np.linalg.svd(points-points.mean(0),full_matrices=False);xyz=(points-points.mean(0))@rot.T
    order=np.argsort(abs(physical[:n]));order=order[order!=REF]
    fig,axes=plt.subplots(1,3,figsize=(17,6),layout='constrained')
    for ax,pair in zip(axes,[(0,1),(0,2),(1,2)]):
        lines=LineCollection(xyz[edges[order]][:,:,pair],cmap='coolwarm',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit),linewidths=.35,rasterized=True)
        lines.set_array(physical[order]);ax.add_collection(lines)
        seg=xyz[edges[REF]];ax.plot(seg[:,pair[0]],seg[:,pair[1]],color='lime',lw=3,label=f'Reference {REF} (self = 1)')
        ax.autoscale_view();ax.set_aspect('equal');ax.legend(fontsize=8)
    fig.colorbar(lines,ax=axes,label='Physical stiffness Pearson correlation with reference')
    fig.suptitle('R-0029 | 10,000 updates | all object springs | grey ≈ 0, red positive, blue negative\nCorrelation under learned rank-32 posterior, not independent physical evidence')
    fig.savefig(out/'correlation-mesh.png',dpi=160);plt.close(fig)
    segments=np.full((n,3,3),np.nan);segments[:,:2]=points[edges];segments=segments.reshape(-1,3)
    colors=np.repeat(np.clip(physical[:n],-limit,limit),3)
    fig=go.Figure(go.Scatter3d(x=segments[:,0],y=segments[:,1],z=segments[:,2],mode='lines',line=dict(color=colors,colorscale='RdBu_r',cmin=-limit,cmax=limit,width=2,colorbar=dict(title='corr(Y,Yref)')),hoverinfo='skip',name='All object springs'))
    seg=points[edges[REF]];fig.add_trace(go.Scatter3d(x=seg[:,0],y=seg[:,1],z=seg[:,2],mode='lines+markers',line=dict(color='lime',width=10),name=f'Reference {REF}'))
    fig.update_layout(title='Physical stiffness correlations: R-0029, ref 109751 (green)',scene=dict(aspectmode='data'),height=850);fig.write_html(out/'correlation-mesh.html')
    fig,ax=plt.subplots(figsize=(9,5),layout='constrained');ax.scatter(distance[other]*100,abs(physical[:n][other]),s=1,alpha=.12)
    ax.scatter(distance[neighbor]*100,abs(physical[:n][neighbor]),s=8,color='orange',label='Shares a vertex with reference');ax.legend();ax.set(xlabel='Spring midpoint distance [cm], frame 0',ylabel='Absolute physical stiffness correlation',title='Locality diagnostic — reference self-correlation excluded');fig.savefig(out/'correlation-distance.png',dpi=160);plt.close(fig)
    stats=lambda mask:dict(count=int(mask.sum()),mean_abs=float(abs(physical[:n][mask]).mean()),median_abs=float(np.median(abs(physical[:n][mask]))),max_abs=float(abs(physical[:n][mask]).max()))
    far=other & (distance>=np.quantile(distance[other],.75))
    record=dict(run='R-0029',evaluation='R-0030',protocol='P-0005',reference=REF,reference_selection=('same centroid-nearest reference as previous mean-field visualization' if REF==109751 else 'exploratory reference supplied by CLI; may be selected after locality scan'),quadrature_max_difference=error,all_other=stats(other),shares_vertex=stats(neighbor),farthest_quartile=stats(far),training_commit=train['repository_commit'],inputs={str(p):sha(p) for p in [checkpoint,base/'topology.npz',data]},limitations='one preselected reference, one seed, rank-32 approximate posterior; geometric midpoint distance not graph distance; controllers excluded from mesh')
    (out/'summary.json').write_text(json.dumps(record,indent=2));print(json.dumps(record,indent=2))

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--reference',type=int,default=109751)
    REF=parser.parse_args().reference
    main()
