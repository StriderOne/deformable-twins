"""Diagonal Gaussian in the simulator's log-stiffness coordinates."""
import math
import torch
from torch import nn
from torch.nn import functional as F


class LogStiffnessPosterior(nn.Module):
    def __init__(self, center: torch.Tensor, initial_std: float, prior_std: float, prior_center_factor: float = 1.0):
        super().__init__()
        if not 0 < initial_std or not 0 < prior_std:
            raise ValueError('Standard deviations must be positive')
        if not math.isfinite(prior_center_factor) or prior_center_factor <= 0:
            raise ValueError('Prior center factor must be positive and finite')
        self.mu = nn.Parameter(center.detach().clone())
        self.rho = nn.Parameter(torch.full_like(center, initial_std).expm1().log())
        self.register_buffer('prior_mu', center.detach().clone() + math.log(prior_center_factor))
        self.register_buffer('prior_std', torch.full_like(center, prior_std))

    @property
    def std(self) -> torch.Tensor:
        return F.softplus(self.rho)

    def sample(self, epsilon: torch.Tensor | None = None) -> torch.Tensor:
        if epsilon is None:
            epsilon = torch.randn_like(self.mu)
        return self.mu + self.std * epsilon

    def kl(self) -> torch.Tensor:
        return (torch.log(self.prior_std / self.std)
                + (self.std.square() + (self.mu - self.prior_mu).square())
                / (2 * self.prior_std.square()) - 0.5).sum()

    def map_penalty(self) -> torch.Tensor:
        return (0.5 * ((self.mu - self.prior_mu) / self.prior_std).square()).sum()


class UniformStiffnessPosterior(nn.Module):
    """Independent logistic-normal Y with a proper Uniform(lower, upper) prior.

    mu/std are in unconstrained logit coordinates; the simulator receives log(Y).
    KL and physical moments use deterministic Gauss-Hermite quadrature.
    """
    def __init__(self, center, initial_std, lower, upper, quadrature_points=64):
        super().__init__()
        import numpy as np
        if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower < upper):
            raise ValueError('Expected finite positive ordered stiffness bounds')
        if not math.isfinite(initial_std) or initial_std <= 0:
            raise ValueError('Expected positive finite logit scale')
        fraction = (center.detach().exp() - lower) / (upper - lower)
        if not ((fraction > 0) & (fraction < 1)).all():
            raise ValueError('Initial median must be strictly within uniform support')
        self.mu = nn.Parameter(torch.logit(fraction))
        self.rho = nn.Parameter(torch.full_like(center, initial_std).expm1().log())
        self.register_buffer('lower', torch.tensor(lower, device=center.device, dtype=center.dtype))
        self.register_buffer('upper', torch.tensor(upper, device=center.device, dtype=center.dtype))
        nodes, weights = np.polynomial.hermite.hermgauss(quadrature_points)
        self.register_buffer('nodes', torch.as_tensor(nodes * math.sqrt(2), device=center.device, dtype=center.dtype)[:, None])
        self.register_buffer('weights', torch.as_tensor(weights / math.sqrt(math.pi), device=center.device, dtype=center.dtype)[:, None])

    @property
    def std(self):
        return F.softplus(self.rho)

    def log_stiffness(self, u):
        return torch.logaddexp(self.lower.log(), (self.upper-self.lower).log() + F.logsigmoid(u))

    def sample(self, epsilon=None):
        if epsilon is None:
            epsilon = torch.randn_like(self.mu)
        return self.log_stiffness(self.mu + self.std * epsilon)

    def point_estimate(self):
        return self.log_stiffness(self.mu)  # Median Y, not mean Y or mean log(Y).

    def kl(self):
        u = self.mu[None, :] + self.std[None, :] * self.nodes
        entropy_jacobian = ((F.softplus(u) + F.softplus(-u)) * self.weights).sum(dim=0)
        return (entropy_jacobian - self.std.log() - .5 * math.log(2 * math.pi * math.e)).sum()

    def physical_moments(self):
        u = self.mu[None, :] + self.std[None, :] * self.nodes
        fraction = torch.sigmoid(u)
        mean = (fraction * self.weights).sum(dim=0)
        variance = ((fraction-mean[None, :]).square() * self.weights).sum(dim=0)
        width = self.upper-self.lower
        return self.lower+width*mean, width*variance.sqrt()

    def std_ratio(self):
        return self.physical_moments()[1] / ((self.upper-self.lower)/math.sqrt(12))


def make_posterior(center, settings):
    if settings['family'] == 'low_rank_logistic_normal_stiffness':
        return LowRankUniformPosterior(center, settings['initial_std'], settings['lower'], settings['upper'], settings['quadrature_points'], settings['rank'])
    if settings['family'] == 'logistic_normal_stiffness':
        return UniformStiffnessPosterior(center, settings['initial_std'],
                                         settings['lower'], settings['upper'], settings['quadrature_points'])
    return LogStiffnessPosterior(center, settings['initial_std'], settings['prior_std'],
                                settings.get('prior_center_factor', 1.0))


class LowRankUniformPosterior(UniformStiffnessPosterior):
    """Gaussian latent covariance diag(d²)+UUᵀ; bounded physical stiffness."""
    def __init__(self, center, initial_std, lower, upper, quadrature_points=128, rank=32):
        super().__init__(center, initial_std, lower, upper, quadrature_points)
        if not 0 < rank <= center.numel():
            raise ValueError('Invalid covariance rank')
        # Small nonzero factors allow covariance learning; preserve initial marginals.
        factor = torch.randn(center.numel(), rank, device=center.device, dtype=center.dtype)
        factor = factor / factor.norm(dim=1, keepdim=True) * (initial_std * .01)
        self.factor = nn.Parameter(factor)
        with torch.no_grad():
            self.rho.copy_((initial_std**2-factor.square().sum(1)).sqrt().expm1().log())

    @property
    def diagonal_std(self):
        return F.softplus(self.rho)

    @property
    def std(self):
        return (self.diagonal_std.square()+self.factor.square().sum(1)).sqrt()

    def sample(self, epsilon=None, factor_epsilon=None):
        if epsilon is None:
            epsilon = torch.randn_like(self.mu)
        if factor_epsilon is None:
            factor_epsilon = torch.randn(self.factor.shape[1], device=self.mu.device, dtype=self.mu.dtype)
        return self.log_stiffness(self.mu+self.diagonal_std*epsilon+self.factor@factor_epsilon)

    def kl(self):
        # Determinant lemma avoids constructing the N x N covariance.
        scaled = self.factor / self.diagonal_std[:, None]
        gram = torch.eye(scaled.shape[1], device=scaled.device, dtype=scaled.dtype)+scaled.T@scaled
        logdet = 2*self.diagonal_std.log().sum()+2*torch.linalg.cholesky(gram).diagonal().log().sum()
        u = self.mu[None, :]+self.std[None, :]*self.nodes
        jacobian = ((F.softplus(u)+F.softplus(-u))*self.weights).sum()
        return jacobian-.5*(self.mu.numel()*math.log(2*math.pi*math.e)+logdet)
