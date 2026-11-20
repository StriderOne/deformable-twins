"""Descriptive joint-sample comparisons; no mode count or acceptance threshold."""
import json
from itertools import combinations
import numpy as np


def compare_samples(candidates, predictions, rows, n_object, split, artifacts):
    samples = {name:z.detach().cpu().numpy() for name,z in candidates if name in predictions}
    errors = {row['name']:row for row in rows}
    pairs = []
    for a,b in combinations(samples,2):
        row = dict(a=a,b=b)
        for group,section in [('object',slice(None,n_object)),('controller',slice(n_object,None))]:
            difference = samples[a][section]-samples[b][section]
            row[group+'_log_stiffness_rms'] = float(np.sqrt(np.mean(difference**2))) if difference.size else None
        for segment,section in [('calibration',slice(None,split-1)),('continuation',slice(split-1,None))]:
            difference = predictions[a][section]-predictions[b][section]
            row[segment+'_prediction_rms_m'] = float(np.sqrt(np.mean(difference**2)))
            row[segment+'_worst_pair_rmse_m'] = max(errors[a][segment]['rmse_m'],errors[b][segment]['rmse_m'])
        pairs.append(row)
    (artifacts/'sample-pairs.json').write_text(json.dumps(dict(pairs=pairs,
        prediction_coordinates='all simulated particles, all xyz; observation errors use original visibility masks',
        interpretation='descriptive; separated modes and mechanical equivalence not established'),indent=2)+'\n')
    if not pairs:
        return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,segment in zip(axes,['calibration','continuation']):
        dots=ax.scatter([p['object_log_stiffness_rms'] for p in pairs],
                        [p[segment+'_prediction_rms_m'] for p in pairs],
                        c=[p[segment+'_worst_pair_rmse_m'] for p in pairs])
        ax.set(xlabel='Object log-stiffness RMS difference',ylabel='Prediction RMS difference [m]',title=segment)
        fig.colorbar(dots,ax=ax,label='Worst pair observation RMSE [m]')
    fig.tight_layout();fig.savefig(artifacts/'sample-pairs.png');plt.close(fig)
