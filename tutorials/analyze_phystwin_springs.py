"""Descriptive checkpoint inspection, not a critical-node or calibration benchmark.

Predefined summaries: +/-10% and +/-25% around initial Y, clamp saturation,
10/50/90 percentiles and mean incident object-edge Y per node (degree normalized).
No trajectories, interventions, held-out losses or physical outcomes are evaluated.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np

from phystwin_runtime import DEFAULT_EXPORT, load_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--graph', type=Path, default=DEFAULT_EXPORT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    output = args.output or Path.home() / '.cache/deformable-twins/spring-analysis' / stamp
    output.mkdir(parents=True, exist_ok=False)
    g, provenance = load_graph(args.graph)
    y0 = float(provenance['config']['init_spring_Y'])
    raw = np.exp(g['log_y'].astype(np.float64))
    active = raw > float(g['y_min'])
    y = np.where(active, np.clip(raw, g['y_min'], g['y_max']), 0.)
    k = y / g['rest']
    n = int(g['n_object'])
    inner = (g['edges'] < n).all(axis=1)
    if not (y0 > 0 and inner.any() and (~inner).any()):
        raise ValueError('Expected positive initial Y and both object/controller springs')
    groups = [('Внутри объекта', inner, '#377eb8'), ('Связи захватов', ~inner, '#e47b20')]
    summary = {}
    for label, mask, _ in groups:
        ratio = y[mask] / y0
        summary[label] = dict(count=int(mask.sum()),
            initial_Y=y0, Y_p10_p50_p90=np.quantile(y[mask], [.1, .5, .9]).tolist(),
            within_10pct=float(np.mean(np.abs(ratio-1) <= .10)),
            within_25pct=float(np.mean(np.abs(ratio-1) <= .25)),
            below_half_initial=float(np.mean(ratio < .5)),
            above_double_initial=float(np.mean(ratio > 2)),
            upper_saturated=float(np.mean(raw[mask] >= g['y_max'])),
            inactive=int(np.sum(~active[mask])),
            k_p10_p50_p90=np.quantile(k[mask], [.1, .5, .9]).tolist())
    endpoints = g['edges'][inner].reshape(-1)
    degree = np.bincount(endpoints, minlength=n)
    total = np.bincount(endpoints, weights=np.repeat(y[inner], 2), minlength=n)
    mean_y = np.divide(total, degree, out=np.full(n, np.nan), where=degree > 0)
    connected = degree > 0
    assert degree.sum() == 2 * inner.sum()
    np.testing.assert_allclose(np.dot(mean_y[connected], degree[connected]), 2*y[inner].sum())
    for stats in summary.values():
        assert 0 <= stats['within_10pct'] <= stats['within_25pct'] <= 1
    summary['nodes'] = dict(connected=int(connected.sum()), isolated=int((~connected).sum()),
                           mean_Y_ratio_p10_p50_p90=np.quantile(mean_y[connected]/y0,[.1,.5,.9]).tolist())
    fig = plt.figure(figsize=(15, 11), layout='constrained')
    ax = fig.add_subplot(221)
    positive = y[y > 0] / y0
    bins = np.geomspace(min(float(positive.min()), .01), max(float(positive.max()), 3), 55)
    for label, mask, color in groups:
        values = y[mask] / y0
        ax.hist(values[values>0], bins=bins, weights=np.full(np.sum(values>0), 100/mask.sum()),
                histtype='step', linewidth=2, color=color, label=f'{label}: {mask.sum():,}')
    ax.axvline(1, color='black', ls='--', label='Общее начальное Y₀')
    ax.axvline(float(g['y_max'])/y0, color='crimson', ls=':', label='Верхний предел solver')
    ax.set(xscale='log', xlabel='Рабочий параметр Y / начальное Y₀', ylabel='Доля связей в интервале, %',
           title='1. Неоднородность параметра пружин')
    ax.legend(fontsize=9)
    ax = fig.add_subplot(222)
    labels = ['В пределах ±10%', 'В пределах ±25%', 'На верхнем пределе']
    keys = ['within_10pct', 'within_25pct', 'upper_saturated']
    for j, (label, _, color) in enumerate(groups):
        values = [100*summary[label][key] for key in keys]
        bars = ax.bar(np.arange(3)+(j-.5)*.36, values, .36, color=color, label=label)
        ax.bar_label(bars, fmt='%.1f%%', padding=3)
    ax.set(xticks=np.arange(3), xticklabels=labels, ylim=(0,105), ylabel='Доля связей, %',
           title='2. Насколько параметры изменились относительно Y₀')
    ax.text(.02,.95,'±10% и ±25% — вложенные диапазоны',transform=ax.transAxes,fontsize=9,va='top')
    ax.legend(loc='upper right',bbox_to_anchor=(1,.88),fontsize=9)
    ax = fig.add_subplot(223)
    for label, mask, color in groups:
        values = np.sort(k[mask])
        ax.plot(values, 100*np.arange(1,len(values)+1)/len(values),color=color,label=label)
    ax.set(xscale='log', xlabel='Эквивалентная продольная жёсткость k = Y / L₀, Н/м',
           ylabel='Доля связей с k ≤ значения по оси X, %',title='3. Длина пружины тоже влияет на жёсткость')
    ax.legend(fontsize=9)
    ax = fig.add_subplot(224, projection='3d')
    q=g['q'][:n]
    ratio=mean_y[connected]/y0
    span=max(float(ratio.max()),1/float(ratio.min()),1.2)
    norm=LogNorm(vmin=1/span,vmax=span)
    dots=ax.scatter(*q[connected].T,c=ratio,cmap='coolwarm',norm=norm,s=5,alpha=.9)
    ax.scatter(*g['q'][n:].T,c='#222222',marker='x',s=25,label='Управляющие точки')
    ax.set(xlabel='X, м',ylabel='Y, м',zlabel='Z, м',title='4. Среднее Y / Y₀ по связям узла внутри объекта')
    ax.set_box_aspect(np.maximum(np.ptp(q,axis=0),.01))
    ax.invert_zaxis()
    ax.view_init(elev=30,azim=25)
    ax.legend(fontsize=8)
    colorbar=fig.colorbar(dots,ax=ax,shrink=.6,label='Среднее по соседним пружинам / Y₀',ticks=[1/span,1,span])
    colorbar.ax.set_yticklabels([f'{1/span:.2f}', '1.00', f'{span:.2f}'])
    inputs=provenance['inputs']
    scene=next((Path(p).parent.name for p in inputs if p.endswith('/final_data.pkl')),args.graph.name)
    checkpoint=next((Path(p).name for p in inputs if p.endswith('.pth')), 'checkpoint')
    fig.suptitle(f'PhysTwin · {scene} · {checkpoint}\n'
                 f'{len(y):,} связей · Y₀ = {y0:,.0f} · значения после clamp',fontsize=16)
    fig.savefig(output/'springs.png',dpi=170)
    fig.savefig(output/'springs.pdf')
    plt.close(fig)
    np.savetxt(output/'nodes.csv',np.column_stack([np.arange(n),q,degree,mean_y,mean_y/y0]),delimiter=',',
               header='node_id,x_m,y_m,z_m,object_degree,mean_object_Y,mean_Y_over_initial',comments='')
    manifest=dict(kind='descriptive_checkpoint_analysis',status='completed',created_utc=stamp,
                  graph=str(args.graph.resolve()),graph_sha256=provenance['graph_sha256'],
                  source_provenance=provenance,script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  versions={'numpy':np.__version__,'matplotlib':matplotlib.__version__},
                  protocol=__doc__,summary=summary,
                  limitations=['Single checkpoint, no uncertainty across seeds.',
                               'Weight heterogeneity and node color do not establish causal importance.',
                               'No held-out evaluation or ablation; no physical execution.',
                               'Units follow simulator conventions; Y is an axial force scale, not Young modulus.',
                               'Node coordinates and edge ordering use reconstructed graph; see source provenance.'])
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(output)
    print(json.dumps(summary,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
