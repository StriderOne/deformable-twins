"""Diagonal Gaussian in the simulator's log-stiffness coordinates."""
import torch
from torch import nn
from torch.nn import functional as F


class LogStiffnessPosterior(nn.Module):
    def __init__(self, center: torch.Tensor, initial_std: float, prior_std: float):
        super().__init__()
        if not 0 < initial_std or not 0 < prior_std:
            raise ValueError('Standard deviations must be positive')
        self.mu = nn.Parameter(center.detach().clone())
        self.rho = nn.Parameter(torch.full_like(center, initial_std).expm1().log())
        self.register_buffer('prior_mu', center.detach().clone())
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
