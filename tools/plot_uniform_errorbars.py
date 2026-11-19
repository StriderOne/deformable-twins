"""Plot all physical posterior means with +/- one standard deviation."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=Path.home()/'.local/share/deformable-twins/artifacts/R-0020-distributions-i001050'


def main():
    rows=list(csv.DictReader((BASE/'all-springs.csv').open()))
    ids=np.array([int(x['spring_id']) for x in rows])
    mean=np.array([float(x['mean_Y']) for x in rows])
    std=np.array([float(x['std_Y']) for x in rows])
    is_object=np.array([x['group']=='object' for x in rows])
    assert np.array_equal(ids,np.arange(len(ids))) and np.isfinite(mean).all() and (std>0).all()
    out=BASE/'mean-std';out.mkdir(exist_ok=False)
    fig,axes=plt.subplots(2,1,figsize=(18,10),layout='constrained')
    for ax,mask,color,label in zip(axes,[is_object,~is_object],['#2267a8','#dc7b24'],['Object','Controller']):
        x=ids[mask];m=mean[mask]/1000;s=std[mask]/1000
        segments=np.stack((np.column_stack((x,m-s)),np.column_stack((x,m+s))),axis=1)
        ax.add_collection(LineCollection(segments,colors=color,linewidths=.45,alpha=.12 if label=='Object' else .45,rasterized=True))
        ax.scatter(x,m,s=2 if label=='Object' else 9,color=color,label='Posterior mean',zorder=3)
        # End caps are meaningful at controller scale; object caps would merge into a solid fill.
        ax.scatter(x,m-s,s=4,marker='_',color=color,alpha=.35)
        ax.scatter(x,m+s,s=4,marker='_',color=color,alpha=.35,label='Mean +/- 1 std endpoints')
        ax.axhline(50.05,color='black',ls='--',lw=1,label='Uniform prior mean')
        ax.set(xlim=(x.min()-1,x.max()+1),ylim=(min(0,float((m-s).min())-1),max(100,float((m+s).max())+1)),
               xlabel='Original spring ID',ylabel='Stiffness Y / 1000',title=f'{label}: every spring ({len(x):,}), mean +/- 1 std(Y)')
        ax.grid(alpha=.18);ax.legend(loc='upper right')
    fig.suptitle('R-0020 | iteration 1050 | physical posterior mean and standard deviation\nBars are mean +/- std, not a confidence interval; values are not clipped to prior bounds',fontsize=14)
    fig.savefig(out/'mean-std-all.png',dpi=170)
    plt.close(fig)
    meta={'run':'R-0020','iteration':1050,'source_csv_sha256':hashlib.sha256((BASE/'all-springs.csv').read_bytes()).hexdigest(),
          'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
          'analysis_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
          'plot':'mean-std-all.png','springs':len(ids),'definition':'physical E[Y] +/- sqrt(Var[Y]); all springs, untruncated bars',
          'outside_support_bars':int(((mean-std<100)|(mean+std>100000)).sum())}
    (out/'manifest.json').write_text(json.dumps(meta,indent=2)+'\n')
    print(out/'mean-std-all.png')


if __name__=='__main__':main()
