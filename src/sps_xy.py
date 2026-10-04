"""Data-free SPS on a product of circles; see docs/sps_experiment_a.md.

The MH state is a whole path. An independent proposal only needs its endpoint
and log weight retained for subsequent decisions. Kernels are PERIODIZED
Gaussians, not Gaussians evaluated at a single shortest angular difference.
"""
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

TAU = 2 * math.pi


def wrap(x):
    return torch.remainder(x + math.pi, TAU) - math.pi


def energy(theta, J=1.0):
    """Total energy, with each right/down bond counted once."""
    return -J * (torch.cos(theta - theta.roll(-1, -1))
                 + torch.cos(theta - theta.roll(-1, -2))).sum((-2, -1))


def wrapped_log_prob(value, mean, std):
    """Per-angle log density w.r.t. dtheta, summed over images -4..4.

    Restriction 0 < std <= 2 is enforced. After centering the residual in
    [-pi,pi), omitted Gaussian images are at least 9*pi away. At std=2
    their pointwise density is < 2e-44; the relative error is < 4e-43.
    This is a controlled numerical approximation to an infinite wrapped sum.
    Use float64 for densities and accumulated path weights, including training.
    """
    std = torch.as_tensor(std, device=value.device, dtype=torch.float64)
    if not bool(torch.isfinite(std).all()) or bool(((std <= 0) | (std > 2)).any()):
        raise ValueError('Wrapped kernel requires finite 0 < std <= 2')
    residual = wrap(value.double() - mean.double())
    images = torch.arange(-4, 5, device=value.device, dtype=torch.float64) * TAU
    z = (residual[..., None] + images) / std[..., None]
    return torch.logsumexp(-0.5 * z.square(), dim=-1) - std.log() - 0.5 * math.log(TAU)


@dataclass
class SPSConfig:
    L: int = 4
    beta: float = 1.12
    J: float = 1.0
    horizon: float = 1.0
    train_steps: int = 32
    hidden: int = 32
    drift_bound: float = 12.0
    sigma_min: float = 0.2
    sigma_max: float = 2.0
    updates: int = 5000
    batch_size: int = 64
    learning_rate: float = 0.001
    seed: int = 11
    checkpoint_every: int = 250

    def validate(self):
        numeric = list(asdict(self).values())
        if not all(math.isfinite(v) for v in numeric):
            raise ValueError('All configuration values must be finite')
        if self.L < 3 or self.beta < 0 or self.J <= 0 or not 0 < self.horizon <= 1:
            raise ValueError('Require L>=3, beta>=0, J>0, 0<horizon<=1')
        if not 0 < self.sigma_min < 1 < self.sigma_max <= 2:
            raise ValueError('Require 0<sigma_min<1<sigma_max<=2')
        for name in ('L', 'train_steps', 'hidden', 'updates', 'batch_size', 'checkpoint_every', 'seed'):
            v = getattr(self, name)
            if not isinstance(v, int) or v < (0 if name == 'seed' else 1):
                raise ValueError(f'Invalid integer {name}')
        if self.learning_rate <= 0 or self.drift_bound <= 0:
            raise ValueError('Learning rate and drift bound must be positive')


class AngularDrift(nn.Module):
    def __init__(self, hidden, bound):
        super().__init__()
        self.bound = bound
        self.net = nn.Sequential(
            nn.Conv2d(3, hidden, 3, padding=1, padding_mode='circular'), nn.SiLU(),
            nn.Conv2d(hidden, hidden, 3, padding=1, padding_mode='circular'), nn.SiLU(),
            nn.Conv2d(hidden, 1, 3, padding=1, padding_mode='circular'))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, theta, t):
        t_channel = torch.ones_like(theta) * t
        x = torch.stack((theta.cos(), theta.sin(), t_channel), dim=1)
        raw = self.net(x).squeeze(1)
        return self.bound * torch.tanh(raw / self.bound)


class XYPathSampler(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.forward_drift = AngularDrift(config.hidden, config.drift_bound)
        self.backward_drift = AngularDrift(config.hidden, config.drift_bound)
        self.noise_net = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))
        nn.init.zeros_(self.noise_net[-1].weight)
        p = (1-config.sigma_min)/(config.sigma_max-config.sigma_min)
        nn.init.constant_(self.noise_net[-1].bias, math.log(p/(1-p)))

    def sigma(self, t):
        param = next(self.parameters())
        x = torch.as_tensor([[t]], dtype=param.dtype, device=param.device)
        c = self.config
        return (c.sigma_min + (c.sigma_max-c.sigma_min)*self.noise_net(x).sigmoid()).squeeze()

    def sample(self, batch_size, steps, generator=None, keep_path=False):
        if batch_size < 1 or steps < 1:
            raise ValueError('Positive batch size and path steps required')
        c = self.config
        p = next(self.parameters())
        shape = (batch_size, c.L, c.L)
        theta = TAU * torch.rand(shape, device=p.device, dtype=p.dtype, generator=generator) - math.pi
        log_qf = torch.full((batch_size,), -c.L*c.L*math.log(TAU), device=p.device, dtype=torch.float64)
        log_qb = torch.zeros_like(log_qf)
        path = [theta] if keep_path else None
        h = c.horizon / steps
        for i in range(steps):
            t0, t1 = i/steps, (i+1)/steps
            std = self.sigma((t0+t1)/2) * math.sqrt(h)
            mean_f = theta + h*self.forward_drift(theta, t0)
            eps = torch.randn(shape, device=p.device, dtype=p.dtype, generator=generator)
            next_theta = wrap(mean_f + std*eps)
            mean_b = next_theta + h*self.backward_drift(next_theta, t1)
            log_qf = log_qf + wrapped_log_prob(next_theta, mean_f, std).sum((-2, -1))
            log_qb = log_qb + wrapped_log_prob(theta, mean_b, std).sum((-2, -1))
            theta = next_theta
            if keep_path:
                path.append(theta)
        total_energy = energy(theta.double(), c.J)
        log_weight = -c.beta*total_energy + log_qb - log_qf
        return dict(theta=theta, log_weight=log_weight, energy=total_energy,
                    log_qf=log_qf, log_qb=log_qb,
                    path=torch.stack(path, 1) if keep_path else None)


def independence_mh(theta, log_weight, log_uniform, burn):
    """Offline independent proposal bank, (chains, initial+attempts, L, L).

    Rejections keep both the current endpoint and its OLD path weight. Outputs
    include repeats. Initial proposal is not an accepted transition or a sample.
    """
    import numpy as np
    theta, log_weight, log_uniform = map(np.asarray, (theta, log_weight, log_uniform))
    chains, total = log_weight.shape
    if theta.shape[:2] != (chains, total) or log_uniform.shape != (chains, total-1):
        raise ValueError('Mismatched proposal/uniform shapes')
    if not 0 <= burn < total-1:
        raise ValueError('Burn-in must leave at least one transition')
    if not all(np.isfinite(x).all() for x in (theta, log_weight, log_uniform)) or (log_uniform > 0).any():
        raise ValueError('Nonfinite proposals or invalid log uniforms')
    current = theta[:, 0].copy()
    weight = log_weight[:, 0].copy()
    output, weights, accepted, probabilities = [], [], [], []
    for j in range(1, total):
        log_alpha = np.minimum(0, log_weight[:, j]-weight)
        take = log_uniform[:, j-1] < log_alpha
        current[take], weight[take] = theta[take, j], log_weight[take, j]
        if j > burn:
            output.append(current.copy())
            weights.append(weight.copy())
            accepted.append(take)
            probabilities.append(np.exp(log_alpha))
    return dict(theta=np.stack(output, 1), log_weight=np.stack(weights, 1),
                accepted=np.stack(accepted, 1), alpha=np.stack(probabilities, 1))
