"""Sequential VI updates with common latent noise within each trajectory."""
import time
import numpy as np
import torch
from .posterior import LowRankUniformPosterior
from .residual_flow import GraphResidualPosterior


def train_pass(rollout, q, optimizer, end, interval, deadline):
    if not isinstance(q, (LowRankUniformPosterior, GraphResidualPosterior)):
        raise ValueError('Frame updates require a low-rank or graph residual posterior')
    if not 1 <= interval <= end - 1:
        raise ValueError('Invalid update interval')
    epsilon = torch.randn_like(q.mu)
    factor_epsilon = (torch.randn(q.factor.shape[1], device=q.mu.device, dtype=q.mu.dtype)
                      if isinstance(q, LowRankUniformPosterior) else None)
    losses = []
    penalty_value = 0.0
    for first in range(1, end, interval):
        optimizer.zero_grad(set_to_none=True)
        if isinstance(q, GraphResidualPosterior):
            z, full_penalty = q.sample_and_penalty(epsilon)
        else:
            z, full_penalty = q.sample(epsilon, factor_epsilon), q.kl()
        if first == 1:
            rollout.reset(z)
        else:
            with torch.no_grad():
                rollout.z_buffer.copy_(z.detach())
        stop = min(first + interval, end)
        gradient = torch.zeros_like(z)
        for frame in range(first, stop):
            if time.monotonic() >= deadline:
                # Partial parameter updates must be recorded as a failed pass.
                raise TimeoutError('Budget exhausted after partial trajectory updates')
            loss, grad = rollout.frame(frame, True)
            losses.append(loss)
            gradient.add_(grad)
        penalty = full_penalty * ((stop - first) / (end - 1))
        torch.autograd.backward((z, penalty), (gradient, torch.ones_like(penalty)))
        if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in q.parameters()):
            raise FloatingPointError('Non-finite posterior gradient')
        penalty_value += float(penalty.detach())
        optimizer.step()
        if not all(torch.isfinite(p).all() for p in q.parameters()):
            raise FloatingPointError('Non-finite posterior parameter after update')
    return np.asarray(losses), penalty_value, z
