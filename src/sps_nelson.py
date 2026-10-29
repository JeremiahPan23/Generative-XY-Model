"""Bounded three-temperature SPS comparison on an absolute generation clock.

The two arms share a forward/noise parameterization and training paths.  The
``free`` arm learns a backward drift; ``nelson`` derives it from a prescribed
Boltzmann score.  This algebraic relation is a hypothesis being tested, not a
guarantee that intermediate proposal marginals are already Boltzmann.

The backward drift here is the mean displacement in a positive-duration
reverse kernel, matching ``sps_xy``.  With that convention the relation is
u_forward + u_backward = sigma(t)**2 * grad(log p_t).
"""
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from sps_xy import AngularDrift, TAU, energy, wrap, wrapped_log_prob


TARGETS = (
    dict(tag='T092', T=0.92),
    dict(tag='Tcentral', T=1 / 1.12),
    dict(tag='T087', T=0.87),
)


@dataclass
class NelsonConfig:
    L: int = 4
    J: float = 1.0
    T_min: float = 0.87
    train_steps: int = 32
    hidden: int = 32
    drift_bound: float = 12.0
    sigma_min: float = 0.2
    sigma_max: float = 2.0
    updates: int = 15000
    batch_size: int = 64
    learning_rate: float = 0.001
    seed: int = 11
    checkpoint_every: int = 250

    def validate(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError('All configuration values must be finite')
        if self.L < 3 or self.J < 0 or self.T_min != TARGETS[-1]['T']:
            raise ValueError('Require L>=3, J>=0, and fixed T_min=0.87')
        if not 0 < self.sigma_min < 1 < self.sigma_max <= 2:
            raise ValueError('Require 0<sigma_min<1<sigma_max<=2')
        for name in ('L', 'train_steps', 'hidden', 'updates', 'batch_size',
                     'checkpoint_every', 'seed'):
            value = getattr(self, name)
            lower = 0 if name == 'seed' else (3 if name == 'train_steps' else 1)
            if isinstance(value, bool) or not isinstance(value, int) or value < lower:
                raise ValueError(f'Invalid integer {name}')
        if self.learning_rate <= 0 or self.drift_bound <= 0:
            raise ValueError('Learning rate and drift bound must be positive')


def grad_total_energy(theta, J=1.0):
    """Analytic gradient of the total periodic XY energy, with autograd intact.

    Every site interacts with its four neighbors.  The derivative is of total
    energy, not energy per bond/site: no lattice-size normalization appears.
    J=0 is supported for the uniform-target mathematical check.
    """
    return J * sum(
        torch.sin(theta - theta.roll(shift, dim))
        for dim in (-2, -1) for shift in (-1, 1)
    )


class MultiTemperaturePathSampler(nn.Module):
    def __init__(self, config, arm='free'):
        super().__init__()
        config.validate()
        if arm not in ('free', 'nelson'):
            raise ValueError('Arm must be free or nelson')
        self.config = config
        self.arm = arm
        # Both arms consume RNG identically for common modules.  Constructing
        # the optional backward network before noise would break paired seeds.
        self.forward_drift = AngularDrift(config.hidden, config.drift_bound)
        self.noise_net = nn.Sequential(nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, 1))
        nn.init.zeros_(self.noise_net[-1].weight)
        probability = (1 - config.sigma_min) / (config.sigma_max - config.sigma_min)
        nn.init.constant_(self.noise_net[-1].bias,
                          math.log(probability / (1 - probability)))
        self.backward_drift = (AngularDrift(config.hidden, config.drift_bound)
                               if arm == 'free' else None)

    def sigma(self, t):
        param = next(self.parameters())
        time = torch.as_tensor(t, dtype=param.dtype, device=param.device).reshape(1, 1)
        c = self.config
        return (c.sigma_min + (c.sigma_max - c.sigma_min)
                * self.noise_net(time).sigmoid()).squeeze()

    def drift_backward(self, theta, t):
        if self.arm == 'free':
            return self.backward_drift(theta, t)
        # Do not bound/clamp this expression: doing so changes the condition.
        beta = t / self.config.T_min
        return (-self.forward_drift(theta, t)
                - self.sigma(t).square() * beta * grad_total_energy(theta, self.config.J))

    def training_grid(self):
        """32 transitions by default, landing exactly on all three targets.

        The first 30 transitions cover [0, t(T=.92)] equally; the remaining
        two hit the central temperature and T=.87.  Both arms use this same
        nonuniform grid.  Small toy configurations retain the exact endpoints.
        """
        c = self.config
        first_stop = c.T_min / TARGETS[0]['T']
        early_steps = c.train_steps - 2
        return ([first_stop * i / early_steps for i in range(early_steps + 1)]
                + [c.T_min / TARGETS[1]['T'], 1.0])

    def _endpoint(self, theta, log_qf, log_qb, t_stop):
        total_energy = energy(theta.double(), self.config.J)
        beta = t_stop / self.config.T_min
        return dict(theta=theta, energy=total_energy,
                    log_weight=-beta * total_energy + log_qb - log_qf,
                    log_qf=log_qf, log_qb=log_qb,
                    t_stop=t_stop, beta=beta)

    def _sample_grid(self, batch_size, grid, generator, keep_path, capture=()):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError('Positive integer batch size required')
        c = self.config
        p = next(self.parameters())
        shape = (batch_size, c.L, c.L)
        theta = TAU * torch.rand(shape, device=p.device, dtype=p.dtype,
                                 generator=generator) - math.pi
        log_qf = torch.full((batch_size,), -c.L * c.L * math.log(TAU),
                            device=p.device, dtype=torch.float64)
        log_qb = torch.zeros_like(log_qf)
        path = [theta] if keep_path else None
        captured = []
        for index, (t0, t1) in enumerate(zip(grid[:-1], grid[1:]), 1):
            dt = t1 - t0
            std = self.sigma((t0 + t1) / 2) * math.sqrt(dt)
            mean_f = theta + dt * self.forward_drift(theta, t0)
            eps = torch.randn(shape, device=p.device, dtype=p.dtype, generator=generator)
            next_theta = wrap(mean_f + std * eps)
            mean_b = next_theta + dt * self.drift_backward(next_theta, t1)
            log_qf = log_qf + wrapped_log_prob(next_theta, mean_f, std).sum((-2, -1))
            log_qb = log_qb + wrapped_log_prob(theta, mean_b, std).sum((-2, -1))
            theta = next_theta
            if keep_path:
                path.append(theta)
            if index in capture:
                prefix = self._endpoint(theta, log_qf, log_qb, t1)
                prefix['path_index'] = index
                captured.append(prefix)
        endpoint = self._endpoint(theta, log_qf, log_qb, grid[-1])
        endpoint['grid'] = list(grid)
        endpoint['path'] = torch.stack(path, 1) if keep_path else None
        return endpoint, captured

    def sample_prefixes(self, batch_size, generator=None, keep_path=False):
        grid = self.training_grid()
        steps = self.config.train_steps
        endpoint, prefixes = self._sample_grid(batch_size, grid, generator, keep_path,
                                                capture=(steps - 2, steps - 1, steps))
        for prefix, target in zip(prefixes, TARGETS):
            prefix.update(target)
        return dict(prefixes=prefixes, grid=endpoint['grid'], path=endpoint['path'])

    def sample(self, batch_size, steps, t_stop=1.0, generator=None, keep_path=False):
        """Sample [0,t_stop] using the absolute clock, not rescaled network time.

        At every step the network sees its actual generation time; changing
        t_stop changes both terminal beta and the last network time.  Equal
        steps at distinct temperatures do not mean equal dt.
        """
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError('Positive integer path steps required')
        if not math.isfinite(t_stop) or not 0 < t_stop <= 1:
            raise ValueError('Require finite 0<t_stop<=1')
        grid = [t_stop * i / steps for i in range(steps + 1)]
        endpoint, _ = self._sample_grid(batch_size, grid, generator, keep_path)
        return endpoint


def multi_temperature_loss(batch, L):
    """Equal weight for the three prefix objectives; each uses its own beta."""
    prefixes = batch['prefixes']
    if len(prefixes) != len(TARGETS) or not isinstance(L, int) or L < 1:
        raise ValueError('Require three prefixes and a positive lattice length')
    return -torch.stack([prefix['log_weight'].mean() for prefix in prefixes]).mean() / L**2
