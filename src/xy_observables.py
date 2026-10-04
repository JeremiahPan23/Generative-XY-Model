"""Shared XY diagnostics. Energies are per bond; pairs are a count convention."""
import numpy as np


def calculate_vorticity(theta):
    theta = np.asarray(theta, dtype=np.float64)
    if theta.ndim < 2 or theta.shape[-1] != theta.shape[-2]:
        raise ValueError('Expected (..., L, L) square angle fields')
    if not np.isfinite(theta).all():
        raise ValueError('Non-finite angles')
    down = np.roll(theta, -1, axis=-2)
    right = np.roll(theta, -1, axis=-1)
    corner = np.roll(down, -1, axis=-1)
    wrap = lambda d: (d + np.pi) % (2 * np.pi) - np.pi
    return np.rint((wrap(down-theta) + wrap(corner-down)
                    + wrap(right-corner) + wrap(theta-right)) / (2*np.pi)).astype(int)


def energy_density(theta, J=1.0):
    theta = np.asarray(theta, dtype=np.float64)
    return -J * (np.cos(theta-np.roll(theta, -1, axis=-1))
                 + np.cos(theta-np.roll(theta, -1, axis=-2))).mean(axis=(-2, -1)) / 2


def vortex_pairs(theta):
    q = calculate_vorticity(theta)
    return ((q == 1).sum(axis=(-2, -1)) + (q == -1).sum(axis=(-2, -1))) / 2


def observables(theta):
    q = calculate_vorticity(theta)
    plus = (q == 1).sum(axis=(-2, -1))
    minus = (q == -1).sum(axis=(-2, -1))
    if np.any(plus != minus) or np.any(np.abs(q) > 1):
        raise ValueError('Unexpected periodic charge: check singular pi bonds or malformed input')
    return dict(n_plus=plus, n_minus=minus, pairs=(plus+minus)/2,
                energy=energy_density(theta))


def mean_summary(values, seed=0, block_size=1, draws=4000):
    """Percentile bootstrap of independent samples or non-overlapping chain blocks.

    Shape (chains, draws) preserves chain boundaries. Stationarity and blocks
    longer than residual correlations are assumptions, not guaranteed by this CI.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[None, :]
    if x.ndim != 2 or x.shape[1] % block_size or x.size < 2:
        raise ValueError('Each chain must contain an integer number of blocks')
    blocks = x.reshape(x.shape[0], -1, block_size).mean(axis=-1).ravel()
    if len(blocks) < 2:
        raise ValueError('At least two blocks required')
    rng = np.random.default_rng(seed)
    means = blocks[rng.integers(len(blocks), size=(draws, len(blocks)))].mean(axis=1)
    return dict(n=int(x.size), mean=float(x.mean()), sd=float(x.std(ddof=1)),
                sem=float(blocks.std(ddof=1)/np.sqrt(len(blocks))),
                ci95=np.quantile(means, [.025, .975]).tolist(),
                block_size=block_size, n_blocks=len(blocks))


def chain_checks(values):
    """Finite-run diagnostics for a measured observable, not a mixing proof."""
    x = np.asarray(values, dtype=float)
    centered = x - x.mean(axis=1, keepdims=True)
    denominator = (centered**2).sum(axis=1)
    lag1 = np.divide((centered[:, :-1]*centered[:, 1:]).sum(axis=1), denominator,
                     out=np.zeros(len(x)), where=denominator > 0)
    halves = np.concatenate(np.split(x, 2, axis=1), axis=0)
    n = halves.shape[1]
    within = halves.var(axis=1, ddof=1).mean()
    between = n * halves.mean(axis=1).var(ddof=1)
    rhat = np.sqrt(((n-1)/n*within + between/n)/within) if within else None
    return dict(lag1_by_chain=lag1.tolist(), split_rhat=float(rhat) if rhat else None,
                chain_means=x.mean(axis=1).tolist(),
                first_half_means=x[:, :x.shape[1]//2].mean(axis=1).tolist(),
                second_half_means=x[:, x.shape[1]//2:].mean(axis=1).tolist(),
                block_sem={str(b): mean_summary(x, block_size=b, draws=100)['sem']
                           for b in (1, 8, 16, 32)})
