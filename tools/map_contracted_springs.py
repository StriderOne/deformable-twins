"""Locate final object springs with std(Y) / prior std(Y) below protocol threshold."""
import csv
import hashlib
import json
import pickle
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np
import plotly.graph_objects as go

ROOT=Path(__file__).resolve().parents[1]


def main():
    train=json.loads((ROOT/'runs/R-0020/manifest.json').read_text())
    evaluation=json.loads((ROOT/'runs/R-0021/manifest.json').read_text())
    assert train['status']==evaluation['status']=='completed'
    artifact=Path(evaluation['artifact_directory'])
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    for name in ['topology.npz','physical-posterior.npz']:
        p=artifact/name;assert sha(p)==evaluation['artifacts'][str(p)]
    data_path=Path(train['config']['data']['observations']).expanduser()
    assert sha(data_path)==train['inputs'][str(data_path)]
    with data_path.open('rb') as f:data=pickle.load(f)
    points=np.concatenate((data['object_points'][0],data['surface_points'],data['interior_points'])).astype(np.float32)
    topology=np.load(artifact/'topology.npz');posterior=np.load(artifact/'physical-posterior.npz')
    n=int(topology['object_springs']);edges=topology['edges'][:n].astype(int)
    assert edges.min()>=0 and edges.max()<len(points)
    np.testing.assert_allclose(np.linalg.norm(points[edges[:,0]]-points[edges[:,1]],axis=1),topology['rest'][:n],rtol=1e-4,atol=1e-7)
    threshold=train['protocol_spec']['primary_metric']['std_ratio_threshold']
    ratio=posterior['std_ratio'][:n];selected=np.flatnonzero(ratio<threshold)
    expected=train['posterior']['object']['fraction_above_threshold']
    assert abs((1-len(selected)/n)-expected)<1e-6
    out=artifact.parent/'R-0020-contracted-springs';out.mkdir(exist_ok=False)
    with (out/'selected-springs.csv').open('w') as f:
        writer=csv.writer(f);writer.writerow(['spring_id','vertex_a','vertex_b','std_ratio','mean_Y','std_Y','rest_length_m'])
        for i in selected:writer.writerow([i,*edges[i],ratio[i],posterior['mean_Y'][i],posterior['std_Y'][i],topology['rest'][i]])
    # PCA rotation is used only for the static views, not for simulation or the 3D export.
    origin=points.mean(0);_,_,rotation=np.linalg.svd(points-origin,full_matrices=False)
    projected=(points-origin)@rotation.T
    fig,axes=plt.subplots(1,3,figsize=(17,7),layout='constrained')
    for ax,(a,b) in zip(axes,[(0,1),(0,2),(1,2)]):
        ax.scatter(projected[:,a],projected[:,b],s=2,color='#b0b4bb',alpha=.25,rasterized=True)
        segments=projected[edges[selected]][:,:,[a,b]]
        ax.add_collection(LineCollection(segments,colors='#db342a',linewidths=.65,alpha=.5,rasterized=True))
        ax.set_aspect('equal');ax.autoscale_view();ax.set(xlabel=f'PC{a+1} [m]',ylabel=f'PC{b+1} [m]')
        ax.grid(alpha=.15)
    fig.suptitle(f'R-0020 | final iteration 2000 | {len(selected):,} / {n:,} object springs ({100*len(selected)/n:.2f}%)\nRed: posterior std(Y) < {threshold:g} x prior std(Y); grey: initial object geometry (frame 0)',fontsize=14)
    fig.savefig(out/'selected-springs.png',dpi=170);plt.close(fig)
    segments=points[edges[selected]]
    xyz=np.full((len(segments),3,3),np.nan);xyz[:,:2,:]=segments;xyz=xyz.reshape(-1,3)
    mid=segments.mean(1)
    fig=go.Figure([
        go.Scatter3d(x=points[:,0],y=points[:,1],z=points[:,2],mode='markers',marker=dict(size=2,color='#b0b4bb',opacity=.14),name='Object geometry, frame 0',hoverinfo='skip'),
        go.Scatter3d(x=xyz[:,0],y=xyz[:,1],z=xyz[:,2],mode='lines',line=dict(color='#db342a',width=3),opacity=.65,name='Selected object springs',hoverinfo='skip'),
        go.Scatter3d(x=mid[:,0],y=mid[:,1],z=mid[:,2],mode='markers',marker=dict(size=2,color=ratio[selected],colorscale='YlOrRd_r',colorbar=dict(title='std / prior std')),customdata=np.column_stack((selected,edges[selected],ratio[selected],posterior['mean_Y'][selected],posterior['std_Y'][selected])),hovertemplate='Spring %{customdata[0]:.0f}<br>Vertices %{customdata[1]:.0f}, %{customdata[2]:.0f}<br>std ratio %{customdata[3]:.4f}<br>Mean Y %{customdata[4]:.1f}<br>Std Y %{customdata[5]:.1f}<extra></extra>',name='Hover for spring ID')])
    fig.update_layout(title=f'R-0020: {len(selected)} object springs with std ratio < {threshold}; iteration 2000',scene=dict(aspectmode='data',xaxis_title='Original X [m]',yaxis_title='Original Y [m]',zaxis_title='Original Z [m]'),height=850)
    fig.write_html(out/'selected-springs.html',include_plotlyjs=True)
    result={'run':'R-0020','evaluation_run':'R-0021','iteration':2000,'selected_count':len(selected),'object_count':n,'selected_fraction':len(selected)/n,'threshold':threshold,'controller_springs_excluded':True,'geometry':'frame 0 original dataset coordinates; static PCA projection only','selection':'std(Y)/prior_std(Y) strictly below threshold','rest_lengths_verified':True,'training_commit':train['repository_commit'],'analysis_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'script_sha256':sha(Path(__file__)),'inputs':{str(p):sha(p) for p in [data_path,artifact/'topology.npz',artifact/'physical-posterior.npz']},'artifacts':{str(p):sha(p) for p in out.iterdir()}}
    path=ROOT/'runs/R-0020-contracted-springs.json'
    with path.open('x') as f:json.dump(result,f,indent=2)
    print(out);print('selected',len(selected),'of',n)


if __name__=='__main__':main()
