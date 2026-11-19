"""TensorBoard metrics and frame-zero object spring width maps."""
import pickle
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter


class Monitor:
    def __init__(self, plan, artifacts, q, object_springs):
        self.writer = SummaryWriter(str(artifacts/'tensorboard'), flush_secs=15)
        self.every = plan['tensorboard']['image_every']
        self.n = object_springs
        with open(plan['data']['observations'], 'rb') as f:
            data = pickle.load(f)
        points = np.concatenate([data['object_points'][0], data['surface_points'], data['interior_points']])
        topology = np.load(artifacts/'topology.npz')
        edges = topology['edges'][:self.n].astype(int)
        np.testing.assert_allclose(np.linalg.norm(points[edges[:,0]]-points[edges[:,1]],axis=1),topology['rest'][:self.n],rtol=1e-4,atol=1e-7)
        _, _, rotation = np.linalg.svd(points-points.mean(0), full_matrices=False)
        self.segments = ((points-points.mean(0))@rotation.T)[edges]
        self.draw(q, 0)

    def update(self, row, q):
        step = row['iteration']
        for key in ['objective','nll_without_constant','penalty','rmse_m','seconds','near_uniform_bounds_fraction']:
            if key in row:
                self.writer.add_scalar('train/'+key, row[key], step)
        for group, stats in row['std'].items():
            for key in ['mean_ratio','median_ratio','fraction_above_threshold']:
                self.writer.add_scalar('std/'+group+'/'+key, stats[key], step)
        if step % self.every == 0:
            self.draw(q, step)

    @torch.no_grad()
    def draw(self, q, step):
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.collections import LineCollection
        from matplotlib.colors import TwoSlopeNorm
        ratio = q.std_ratio().cpu().numpy()
        self.writer.add_histogram('std/object_ratio', ratio[:self.n], step)
        self.writer.add_histogram('std/controller_ratio', ratio[self.n:], step)
        fig, axes = plt.subplots(1,3,figsize=(15,5),layout='constrained')
        for ax, pair in zip(axes, [(0,1),(0,2),(1,2)]):
            order = np.argsort(abs(ratio[:self.n]-1))
            lines = LineCollection(self.segments[order][:,:,pair], cmap='coolwarm_r',norm=TwoSlopeNorm(vmin=0,vcenter=1,vmax=1.75),linewidths=.3,rasterized=True)
            lines.set_array(ratio[:self.n][order]);ax.add_collection(lines)
            ax.autoscale_view();ax.set_aspect('equal');ax.set_title(str(pair))
        fig.colorbar(lines,ax=axes,label='std(Y) / uniform prior std; red = narrower')
        fig.suptitle(f'All object springs | iteration {step} | frame 0 PCA views')
        self.writer.add_figure('mesh/std_relative_to_prior',fig,step,close=True)
        self.writer.flush()

    def close(self):
        self.writer.close()
