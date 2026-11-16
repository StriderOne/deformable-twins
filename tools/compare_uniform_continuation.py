"""Compare completed same-topology uniform VI checkpoints and evaluation runs."""
import json
from pathlib import Path

import numpy as np
from lab.calibration.phystwin_adapter import ROOT, sha256
from lab.calibration.phystwin_vi import write_json


def change(before,after):
    d=after-before
    return {'mean_signed':float(d.mean()),'mean_absolute':float(abs(d).mean()),
            'median_absolute':float(np.median(abs(d))),'p95_absolute':float(np.quantile(abs(d),.95))}


def compare(plan,output):
    ids=[plan['continuation']['reference_evaluation'],plan['run_ids']['evaluate']]
    manifests=[json.loads((ROOT/'runs'/rid/'manifest.json').read_text()) for rid in ids]
    assert all(m['status']=='completed' for m in manifests)
    paths=[Path(m['artifact_directory']) for m in manifests]
    for m,p in zip(manifests,paths):
        for name in ['topology.npz','physical-posterior.npz']:
            assert sha256(p/name)==m['artifacts'][str(p/name)]
    topologies=[np.load(p/'topology.npz') for p in paths]
    assert all(np.array_equal(topologies[0][k],topologies[1][k]) for k in topologies[0].files)
    n=int(topologies[0]['object_springs'])
    posterior=[np.load(p/'physical-posterior.npz') for p in paths]
    groups={}
    for group,sl in [('object',slice(None,n)),('controller',slice(n,None))]:
        groups[group]={key:change(posterior[0][key][sl],posterior[1][key][sl]) for key in ['mean_Y','median_Y','std_Y','std_ratio']}
        groups[group]['fraction_std_decreased']=float((posterior[1]['std_Y'][sl]<posterior[0]['std_Y'][sl]).mean())
    results={m['id']:m['results'] for m in manifests}
    record={'experiment':plan['id'],'reference_evaluation':ids[0],'new_evaluation':ids[1],
            'evaluation_commits':{m['id']:m['repository_commit'] for m in manifests},
            'topology_equal':True,'groups':groups,'evaluations':results,
            'interpretation':'same-data continuation; convergence not established automatically; continuation frames not independent test'}
    write_json(output/'comparison.json',record)
    count=len(posterior[0]['mean_Y'])
    np.savetxt(output/'paired-springs.csv',np.column_stack([np.arange(count)]+[p[k] for p in posterior for k in ['mean_Y','std_Y','std_ratio']]),delimiter=',',header='spring_id,old_mean_Y,old_std_Y,old_std_ratio,new_mean_Y,new_std_Y,new_std_ratio',comments='')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,5),layout='constrained')
    for ax,key in zip(axes,['mean_Y','std_Y']):
        for label,sl in [('Object',slice(None,n)),('Controller',slice(n,None))]:
            ax.scatter(posterior[0][key][sl],posterior[1][key][sl],s=1,alpha=.3,label=label,rasterized=True)
        low=min(p[key].min() for p in posterior);high=max(p[key].max() for p in posterior)
        ax.plot([low,high],[low,high],'k--',lw=1)
        ax.set(xlabel=f'{ids[0]} {key}',ylabel=f'{ids[1]} {key}');ax.legend()
    fig.savefig(output/'paired-posterior.png',dpi=160);plt.close(fig)
