"""Graph residual posterior; stochastic Russian-roulette log determinant.

The graph is fixed. Every aggregation is an l2 contraction; linear layers use
an explicit induced-norm upper bound rather than an approximate power iteration.
"""
import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def spatial_edges(points, springs, k):
    from scipy.spatial import cKDTree
    points, springs = np.asarray(points), np.asarray(springs, dtype=np.int64)
    if points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all():
        raise ValueError('Expected finite 3D points')
    if springs.ndim != 2 or springs.shape[1] != 2 or np.any(springs < 0) or np.any(springs >= len(points)):
        raise ValueError('Invalid spring endpoints')
    if k < 1 or len(springs) < 2:
        raise ValueError('Expected positive neighbors and at least two springs')
    centers = points[springs].mean(1)
    tree = cKDTree(centers)
    pairs = set()
    for i in range(len(springs)):
        count = min(len(springs), max(16, 2*k+1))
        while True:
            distances, indices = tree.query(centers[i], k=count)
            candidates = sorted(zip(np.atleast_1d(distances), np.atleast_1d(indices)))
            valid = [int(j) for _, j in candidates if j != i and not np.isin(springs[j], springs[i]).any()]
            if len(valid) >= k or count == len(springs):
                break
            count = min(len(springs), count*2)
        for j in valid[:k]:
            pairs.add((min(i,j), max(i,j)))
    return np.array(sorted(pairs), dtype=np.int64).reshape(-1,2)


class SpringGraph(nn.Module):
    def __init__(self, endpoints, neighbors):
        super().__init__()
        ends = torch.as_tensor(endpoints, dtype=torch.long)
        near = torch.as_tensor(neighbors, dtype=torch.long).reshape(-1,2)
        self.n = len(ends)
        self.vertices = int(ends.max())+1
        self.register_buffer('ends', ends.flatten())
        self.register_buffer('spring_ids', torch.arange(self.n).repeat_interleave(2))
        degree = torch.bincount(self.ends, minlength=self.vertices).float()
        self.register_buffer('incidence_weight', (2*degree[self.ends]).rsqrt())
        directed = torch.cat((near, near.flip(1)), dim=0)
        self.register_buffer('source', directed[:,0])
        self.register_buffer('target', directed[:,1])
        degree = torch.bincount(self.source, minlength=self.n).float()
        self.register_buffer('near_weight', (degree[self.source]*degree[self.target]).rsqrt())

    def forward(self, x):
        # B B^T with normalized bipartite incidence B; norm <= 1.
        w = self.incidence_weight[:,None].to(x.dtype)
        vertices = x.new_zeros(self.vertices, x.shape[1]).index_add(0,self.ends,x[self.spring_ids]*w)
        physical = x.new_zeros(x.shape).index_add(0,self.spring_ids,vertices[self.ends]*w)
        spatial = x.new_zeros(x.shape).index_add(0,self.target,x[self.source]*self.near_weight[:,None].to(x.dtype))
        return (x+physical+spatial)/3


def bounded_linear(x, layer):
    w = layer.weight
    bound = (w.abs().sum(0).max()*w.abs().sum(1).max()).sqrt().clamp_min(1.)
    return F.linear(x, w/bound, layer.bias)


class ResidualBlock(nn.Module):
    def __init__(self, width, contraction):
        super().__init__()
        self.first = nn.Linear(1,width)
        self.last = nn.Linear(width,1)
        self.contraction = contraction
        nn.init.normal_(self.last.weight, std=.001)
        nn.init.zeros_(self.last.bias)

    def residual(self, u, graph):
        h = torch.tanh(bounded_linear(u[:,None],self.first))
        return self.contraction*bounded_linear(graph(h),self.last).squeeze(1)

    def forward(self, u, graph):
        return u+self.residual(u,graph)

    def logdet(self, u, graph, minimum, continuation):
        g = self.residual(u,graph)
        v = torch.empty_like(u).bernoulli_(.5).mul_(2).sub_(1)
        power = v
        result = u.new_zeros(())
        neumann = torch.zeros_like(u)
        terms = minimum
        while torch.rand((),device=u.device) < continuation:
            terms += 1
        for j in range(1,terms+1):
            survival = continuation**max(0,j-minimum)
            coefficient = (-1)**(j+1)/survival
            neumann = neumann + coefficient*power
            power = torch.autograd.grad(g,u,power,create_graph=False,retain_graph=True)[0]
            result = result + (coefficient/j)*(v*power).sum()
        # Cyclic trace identity: d tr(J^k)/k = tr(J^(k-1) dJ).
        # This unbiased gradient estimator stores one differentiable VJP, not
        # the entire random-length series. Its sample gradient need not equal
        # the derivative of the particular sampled logdet value.
        jtn = torch.autograd.grad(g,u,neumann.detach(),create_graph=True,retain_graph=True)[0]
        surrogate = (v*jtn).sum()
        return result.detach()+surrogate-surrogate.detach()



class GraphResidualPosterior(nn.Module):
    def __init__(self, center, settings, graph):
        super().__init__()
        self.settings = settings
        if not (0 < settings['lower'] < settings['upper'] <= 1e5):
            raise ValueError('Invalid stiffness support')
        if not (0 < settings['contraction'] < 1 and settings['contraction']**2 < settings['trace_continuation'] < 1):
            raise ValueError('Trace survival must exceed squared contraction for a finite-variance bound')
        if any(settings[key] < 1 for key in ('blocks','width','spatial_neighbors','trace_min_terms')):
            raise ValueError('Expected positive architecture sizes')
        self.graph = graph.to(center.device)
        self.lower, self.upper = settings['lower'], settings['upper']
        fraction = (center.detach().exp()-self.lower)/(self.upper-self.lower)
        if not ((fraction>0)&(fraction<1)).all():
            raise ValueError('Initial stiffness outside prior support')
        self.mu = nn.Parameter(torch.logit(fraction))
        self.rho = nn.Parameter(torch.full_like(center,settings['initial_std']).expm1().log())
        self.blocks = nn.ModuleList([ResidualBlock(settings['width'],settings['contraction']) for _ in range(settings['blocks'])]).to(center)

    def draw_noise(self):
        return torch.randn_like(self.mu)

    def transform(self, epsilon, density=False):
        scale = F.softplus(self.rho)
        u = self.mu+scale*epsilon
        logq = (-.5*(epsilon.square()+math.log(2*math.pi))-scale.log()).sum()
        for block in self.blocks:
            if density:
                logq = logq-block.logdet(u,self.graph,self.settings['trace_min_terms'],self.settings['trace_continuation'])
            u = block(u,self.graph)
        log_jac = math.log(self.upper-self.lower)+F.logsigmoid(u)+F.logsigmoid(-u)
        z = torch.logaddexp(u.new_tensor(math.log(self.lower)), math.log(self.upper-self.lower)+F.logsigmoid(u))
        return z, logq-log_jac.sum()

    def sample(self, epsilon=None):
        return self.transform(self.draw_noise() if epsilon is None else epsilon)[0]

    def sample_and_penalty(self, epsilon):
        z, logq = self.transform(epsilon,True)
        return z, logq+self.mu.numel()*math.log(self.upper-self.lower)

    @torch.no_grad()
    def diagnostic_samples(self, count=32):
        # Diagnostics must not perturb the training RNG, including during resume.
        devices = [self.mu.device.index] if self.mu.is_cuda else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(1729)
            return torch.stack([self.sample().exp() for _ in range(count)])

    def physical_moments(self):
        samples = self.diagnostic_samples()
        return samples.mean(0),samples.std(0)

    def std_ratio(self):
        return self.physical_moments()[1]/((self.upper-self.lower)/math.sqrt(12))
