"""Color every object spring by its final physical posterior/prior std ratio."""
import hashlib
import json
import pickle
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
import numpy as np
import plotly.graph_objects as go
from plotly.colors import sample_colorscale

ROOT=Path(__file__).resolve().parents[1]


def main():
    train=json.loads((ROOT/'runs/R-0020/manifest.json').read_text())
    ev=json.loads((ROOT/'runs/R-0021/manifest.json').read_text())
    base=Path(ev['artifact_directory'])
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ['topology.npz','physical-posterior.npz']:
        assert sha(base/name)==ev['artifacts'][str(base/name)]
    source=Path(train['config']['data']['observations']).expanduser()
    assert sha(source)==train['inputs'][str(source)]
    with source.open('rb') as f:data=pickle.load(f)
    points=np.concatenate([data['object_points'][0],data['surface_points'],data['interior_points']]).astype(np.float32)
    topo=np.load(base/'topology.npz');post=np.load(base/'physical-posterior.npz')
    n=int(topo['object_springs']);edges=topo['edges'][:n].astype(int);ratio=post['std_ratio'][:n]
    assert len(ratio)==n and np.isfinite(ratio).all()
    np.testing.assert_allclose(np.linalg.norm(points[edges[:,0]]-points[edges[:,1]],axis=1),topo['rest'][:n],rtol=1e-4,atol=1e-7)
    out=base.parent/'R-0020-spring-std-change-colors';out.mkdir(exist_ok=False)
    low=float(ratio.min());high=float(ratio.max())
    extent=max(1-low,high-1);vmin=1-extent;vmax=1+extent
    norm=TwoSlopeNorm(vmin=vmin,vcenter=1.,vmax=vmax)
    palette=LinearSegmentedColormap.from_list('change',['#e52413','#eeeeee','#1479e8'])
    plotly_palette=[[0,'#e52413'],[.5,'#eeeeee'],[1,'#1479e8']]
    _,_,rotation=np.linalg.svd(points-points.mean(0),full_matrices=False)
    xyz=(points-points.mean(0))@rotation.T
    fig,axes=plt.subplots(1,3,figsize=(18,7),layout='constrained')
    for ax,(a,b) in zip(axes,[(0,1),(0,2),(1,2)]):
        ax.scatter(xyz[:,a],xyz[:,b],s=1,color='#b8b8b8',alpha=.2,rasterized=True)
        # Draw weakly changed springs first so they do not obscure stronger changes.
        depth=next(i for i in range(3) if i not in (a,b))
        order=np.argsort(np.abs(ratio-1),kind='stable')
        lines=LineCollection(xyz[edges[order]][:,:,[a,b]],cmap=palette,norm=norm,linewidths=.6,rasterized=True)
        lines.set_array(ratio[order]);ax.add_collection(lines)
        ax.set_aspect('equal');ax.autoscale_view();ax.set(xlabel=f'PC{a+1} [m]',ylabel=f'PC{b+1} [m]');ax.grid(alpha=.1)
    bar=fig.colorbar(lines,ax=axes,shrink=.7,pad=.02)
    bar.set_label('Posterior std(Y) / prior std(Y) — grey = 1')
    fig.suptitle(f'R-0020 | iteration 2000 | all {n:,} object springs\nGrey: width close to prior | brighter red: narrower | brighter blue: wider\nController springs excluded; equal deviations from 1 have equal color intensity',fontsize=14)
    fig.savefig(out/'spring-std-colors.png',dpi=180);plt.close(fig)
    segments=points[edges];mid=segments.mean(1)
    # Plotly supports scalar colors on each line vertex; both ends get the spring's value.
    coords=np.full((n,3,3),np.nan);coords[:,:2]=segments;coords=coords.reshape(-1,3)
    colors=np.repeat(ratio,3)
    fig=go.Figure([
        go.Scatter3d(x=coords[:,0],y=coords[:,1],z=coords[:,2],mode='lines',line=dict(color=colors,colorscale=plotly_palette,cmin=vmin,cmax=vmax,width=3,colorbar=dict(title='std / prior std')),name='All object springs',hoverinfo='skip'),
        go.Scatter3d(x=mid[:,0],y=mid[:,1],z=mid[:,2],mode='markers',marker=dict(size=2,color=ratio,colorscale=plotly_palette,cmin=vmin,cmax=vmax,showscale=False),customdata=np.column_stack([np.arange(n),edges,ratio]),hovertemplate='Spring %{customdata[0]:.0f}<br>Vertices %{customdata[1]:.0f}, %{customdata[2]:.0f}<br>std(Y)/prior std(Y): %{customdata[3]:.6f}<extra></extra>',name='Hover for exact ratio')])
    fig.update_layout(title=f'All {n} object springs | R-0020, iteration 2000 | std ratio [{low:.3f}, {high:.3f}]',height=900,scene=dict(aspectmode='data',xaxis_title='Original X [m]',yaxis_title='Original Y [m]',zaxis_title='Original Z [m]'))
    fig.write_html(out/'spring-std-colors.html',include_plotlyjs=True)
    result={'run':'R-0020','evaluation':'R-0021','iteration':2000,'springs':n,'controllers_excluded':True,'color':'grey at ratio 1, red below, blue above; symmetric linear intensity by abs(ratio-1)','limits':[vmin,vmax],'data_limits':[low,high],'geometry':'frame 0, PCA projection for PNG; original coordinates for HTML','training_commit':train['repository_commit'],'analysis_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'script_sha256':sha(Path(__file__)),'artifacts':{str(p):sha(p) for p in out.iterdir()}}
    with (ROOT/'runs/R-0020-spring-std-change-colors.json').open('x') as f:json.dump(result,f,indent=2)
    print(out);print(low,high)


if __name__=='__main__':main()
