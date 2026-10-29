"""Raw intermediate-frame evaluation; no training, MH, or density scoring.

The frozen forward Euler step is reused algebraically. An inserted capture
time is an actual integration node, never an interpolated/relabeled frame.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f'{name} must be a positive integer')
    return value


def make_grid(base_steps, capture_times, policy='insert'):
    """Uniform base mesh, with exact captures inserted or strictly required.

    Tolerance only resolves floating-point arithmetic at the SAME node; it
    never permits rounding a requested time to a nearby simulation step.
    """
    positive_int(base_steps, 'base_steps')
    times = [float(t) for t in capture_times]
    if (len(times) < 2 or not all(math.isfinite(t) and 0 <= t <= 1 for t in times)
            or times != sorted(set(times)) or times[0] != 0 or times[-1] != 1):
        raise ValueError('Capture times must be unique, increasing, and include 0 and 1')
    if policy not in ('insert', 'strict'):
        raise ValueError('Grid policy must be insert or strict')
    grid = [i / base_steps for i in range(base_steps + 1)]
    for t in times:
        near = round(t * base_steps)
        if abs(grid[near] - t) <= 1e-14:
            grid[near] = t  # preserve requested floating-point representation
        elif policy == 'strict':
            raise ValueError(f'Capture t={t} is not on the {base_steps}-step mesh')
        else:
            grid.append(t)
    grid = np.array(sorted(grid), dtype=np.float64)
    dt = np.diff(grid)
    if np.any(dt <= 1e-14):
        raise ValueError('Duplicate or numerically unresolved integration nodes')
    indices = np.searchsorted(grid, times)
    if not np.array_equal(grid[indices], times):
        raise ValueError('Capture nodes not represented exactly')
    return dict(grid=grid, dt=dt, capture_times=np.array(times),
                capture_indices=indices, base_steps=base_steps, policy=policy)


def frame_metadata(times, T_min=0.87):
    if not math.isfinite(T_min) or T_min <= 0:
        raise ValueError('Invalid T_min')
    return [dict(index=i, time=float(t), beta=float(t / T_min),
                 temperature=None if t == 0 else float(T_min / t),
                 temperature_limit='infinity' if t == 0 else None)
            for i, t in enumerate(times)]


@torch.inference_mode()
def collect_raw_frames(model, count, mesh, batch_size, seed, progress=None):
    """Return theta[path, frame, y, x] and preserve path identity across time.

    Batch size, dtype, device and RNG algorithm are provenance: changing the
    batch size need not reproduce identical paths. No backward drift is used.
    """
    positive_int(count, 'count')
    positive_int(batch_size, 'batch_size')
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**63:
        raise ValueError('Invalid path RNG seed')
    param = next(model.parameters())
    generator = torch.Generator(device=param.device).manual_seed(seed)
    grid, wanted = mesh['grid'], mesh['capture_indices']
    if (grid[0] != 0 or grid[-1] != 1 or np.any(np.diff(grid) <= 0)
            or not np.array_equal(grid[wanted], mesh['capture_times'])):
        raise ValueError('Malformed mesh')
    captures = {int(index): slot for slot, index in enumerate(wanted)}
    output = np.empty((count, len(wanted), model.config.L, model.config.L),
                      dtype=param.detach().cpu().numpy().dtype)
    model.eval()
    for first in range(0, count, batch_size):
        n = min(batch_size, count - first)
        shape = (n, model.config.L, model.config.L)
        theta = 2 * math.pi * torch.rand(shape, device=param.device,
                                        dtype=param.dtype, generator=generator) - math.pi
        output[first:first+n, captures[0]] = theta.cpu().numpy()
        for index, (t0, t1) in enumerate(zip(grid[:-1], grid[1:]), 1):
            t0, t1 = float(t0), float(t1)
            dt = t1 - t0
            std = model.sigma((t0 + t1) / 2) * math.sqrt(dt)
            mean = theta + dt * model.forward_drift(theta, t0)
            eps = torch.randn(shape, device=param.device, dtype=param.dtype, generator=generator)
            theta = torch.remainder(mean + std * eps + math.pi, 2 * math.pi) - math.pi
            if index in captures:
                output[first:first+n, captures[index]] = theta.cpu().numpy()
        if progress:
            progress(first + n, count)
    if not np.isfinite(output).all():
        raise ValueError('Non-finite raw frame')
    return output


def measure_frames(theta, J=1.0):
    """All outputs retain leading axes, including path/chain/draw and frame."""
    from xy_observables import calculate_vorticity, energy_density
    a = np.asarray(theta, dtype=np.float64)
    if (a.ndim < 3 or a.shape[-1] != a.shape[-2] or a.shape[-1] < 3
            or not np.isfinite(a).all() or not math.isfinite(J) or J < 0):
        raise ValueError('Expected finite (..., L, L), L>=3, and nonnegative J')
    V = a.shape[-1] ** 2
    q = calculate_vorticity(a)
    if np.any(q.sum(axis=(-2, -1)) != 0) or np.any(np.abs(q) > 1):
        raise ValueError('Unphysical periodic charge; check singular pi bonds')
    Mx, My = np.cos(a).sum((-2, -1)), np.sin(a).sum((-2, -1))
    M2 = Mx**2 + My**2
    absolute_charge = np.abs(q).sum((-2, -1))
    e_bond = energy_density(a, J)
    return dict(energy_total=2 * V * e_bond, energy_per_site=2 * e_bond,
                energy_per_bond=e_bond, vortex_charge=q,
                vortex_abs_count=absolute_charge,
                vortex_net_charge=q.sum((-2, -1)),
                vortex_pair_count=absolute_charge / 2,
                vorticity_density=absolute_charge / V,
                Mx=Mx, My=My, Mx2=Mx**2, My2=My**2,
                M2=M2, M_abs=np.sqrt(M2), m_abs=np.sqrt(M2) / V)


MOMENTS = ('energy_per_bond', 'vorticity_density', 'Mx', 'My', 'Mx2', 'My2',
           'M2', 'M_abs', 'm_abs')


def _estimates(means, beta, V):
    result = {name: means[..., i] for i, name in enumerate(MOMENTS)}
    # Finite-sample plug-in central moments, with denominator n (not n-1).
    # Do not force mean magnetization to zero: the learned CNN may be biased.
    result['chi_vector_trace'] = beta / V * (
        result['M2'] - result['Mx']**2 - result['My']**2)
    result['chi_component_average'] = result['chi_vector_trace'] / 2
    result['chi_radial'] = beta / V * (result['M2'] - result['M_abs']**2)
    result['chi_spin_unscaled'] = result['M2'] / V
    return result


def summarize_frames(observations, times, L, *, chains=1, block_size=1,
                     bootstrap_draws=512, seed=911, paired_frames=True, T_min=0.87):
    """IID path bootstrap, or chain-stratified nonoverlapping block bootstrap.

    observations have shape (chains*draws, frames). Chain rows are contiguous;
    no block crosses chain boundaries. Nonlinear susceptibilities are recomputed
    in every bootstrap sample, using shared row indices across SPS frames.
    """
    for name, value in [('chains', chains), ('block_size', block_size),
                        ('bootstrap_draws', bootstrap_draws)]:
        positive_int(value, name)
    x = np.stack([observations[key] for key in MOMENTS], axis=-1).astype(np.float64)
    if (x.ndim != 3 or x.shape[1] != len(times) or not np.isfinite(x).all()
            or x.shape[0] % (chains * block_size)):
        raise ValueError('Expected aligned finite (records, frames) with complete chain blocks')
    blocks_per_chain = x.shape[0] // (chains * block_size)
    if blocks_per_chain < 2 or bootstrap_draws < 2:
        raise ValueError('At least two blocks per chain and bootstrap replicates required')
    blocks = x.reshape(chains, blocks_per_chain, block_size, len(times), -1).mean(axis=2)
    beta = np.array(times, dtype=float) / T_min
    point = _estimates(x.mean(axis=0), beta, L**2)
    rng = np.random.default_rng(seed)
    replicate = np.empty((bootstrap_draws, len(times), len(MOMENTS)))
    # Draw one replicate at a time: bounded memory even for a larger review budget.
    for b in range(bootstrap_draws):
        if paired_frames:
            indices = rng.integers(blocks_per_chain, size=(chains, blocks_per_chain))
            replicate[b] = blocks[np.arange(chains)[:, None], indices].mean(axis=(0, 1))
        else:
            for t in range(len(times)):
                indices = rng.integers(blocks_per_chain, size=(chains, blocks_per_chain))
                replicate[b, t] = blocks[np.arange(chains)[:, None], indices, t].mean(axis=(0, 1))
    boot = _estimates(replicate, beta, L**2)
    rows = []
    for t, metadata in enumerate(frame_metadata(times, T_min)):
        values = {key: dict(value=float(value[t]),
                           bootstrap_sem=float(boot[key][:, t].std(ddof=1)),
                           ci95=np.quantile(boot[key][:, t], [.025, .975]).tolist())
                  for key, value in point.items()}
        presence = int(np.count_nonzero(observations['vortex_abs_count'][:, t]))
        rare = dict(configurations_with_vortices=presence, total_records=x.shape[0],
                    fraction=presence / x.shape[0],
                    iid_zero_count_upper95=(float(-np.expm1(math.log(.05)/x.shape[0]))
                                            if presence == 0 and chains == 1 and block_size == 1 else None),
                    note='Zero observed events and a zero-width bootstrap interval do not prove zero density. '
                         'Binomial upper bound applies only to independent raw paths, not Wolff chains.')
        rows.append(dict(**metadata, estimates=values, vortex_event_counts=rare))
    return dict(frames=rows, records_per_frame=x.shape[0], chains=chains,
                block_size=block_size, blocks_per_chain=blocks_per_chain,
                bootstrap_draws=bootstrap_draws, bootstrap_seed=seed,
                paired_frames=paired_frames, susceptibility_estimator='finite_n_plugin',
                warning='Pointwise finite-run intervals; no equilibrium/mixing claim. '
                        'Block adequacy must be reviewed for each Wolff observable.')


def histogram_frames(observations, L, J=1.0):
    """Common fixed bins for both arms and Wolff. Counts sum to records/frame."""
    nframes = observations['energy_per_bond'].shape[1]
    edges = dict(energy_per_bond=np.linspace(-J, J, 41) if J > 0
                 else np.linspace(-.5, .5, 41),
                 m_abs=np.linspace(0, 1 + 1e-12, 41),
                 vortex_abs_count=np.arange(L**2 + 2) - .5,
                 vorticity_density=(np.arange(L**2 + 2) - .5) / L**2)
    result = {}
    for key, bins in edges.items():
        values = observations[key]
        counts = np.stack([np.histogram(values[:, t], bins=bins)[0] for t in range(nframes)])
        if np.any(counts.sum(axis=1) != len(values)):
            raise ValueError(f'Histogram would silently drop samples: {key}')
        result[key] = dict(edges=bins.tolist(), counts=counts.tolist(),
                           probability_mass=(counts / len(values)).tolist(),
                           note='Vortex count / L^2 gives density; masses are not PDF heights.')
    return result


def verify_frozen_run(root, spec):
    """Check old protocol bytes, every source pin, checkpoint and metadata."""
    root = Path(root)
    directory = root / spec['run_directory']
    protocol_path = directory / 'protocol.json'
    checkpoint = directory / 'train/checkpoint.pt'
    metadata_path = directory / 'train/checkpoint_metadata.json'
    if digest(protocol_path) != spec['protocol_sha256']:
        raise ValueError('Frozen protocol byte hash mismatch')
    protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    for relative, expected in protocol['source_sha256'].items():
        if digest(root / relative) != expected:
            raise ValueError(f'Frozen source mismatch: {relative}')
    if (digest(checkpoint) != spec['checkpoint_sha256']
            or metadata['checkpoint_sha256'] != spec['checkpoint_sha256']
            or metadata['protocol_sha256'] != spec['protocol_sha256']
            or metadata['config'] != protocol['baseline_config']
            or metadata['sources'] != protocol['source_sha256']
            or metadata['arm'] != spec['arm'] or metadata['update'] != 15000
            or metadata['config']['seed'] != 11 or metadata['config']['L'] != 4
            or metadata['config']['T_min'] != .87
            or metadata['environment'] != protocol['baseline_environment']
            or metadata['train_path_seed'] != protocol['path_generator_seed']):
        raise ValueError('Frozen checkpoint metadata mismatch')
    return dict(checkpoint=str(checkpoint.resolve()), checkpoint_sha256=digest(checkpoint),
                metadata_sha256=digest(metadata_path), protocol_sha256=digest(protocol_path),
                source_sha256=protocol['source_sha256'], metadata=metadata)


def load_frozen_model(root, spec, device='cpu'):
    verified = verify_frozen_run(root, spec)
    from sps_nelson import NelsonConfig, MultiTemperaturePathSampler
    payload = torch.load(verified['checkpoint'], map_location='cpu', weights_only=False)
    metadata = verified['metadata']
    for key in ('format_version', 'config', 'arm', 'update', 'sources',
                'protocol_sha256', 'environment', 'train_path_seed'):
        if payload[key] != metadata[key]:
            raise ValueError(f'Checkpoint payload disagrees with metadata: {key}')
    model = MultiTemperaturePathSampler(NelsonConfig(**payload['config']), arm=payload['arm'])
    model.load_state_dict(payload['model'], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError('Non-finite model parameters')
    model.to(device).eval().requires_grad_(False)
    return model, verified


def collect_wolff(times, *, L, J, T_min, chains, draws, burn, interval, seed, reuse=None):
    """New finite reference budget; t=0 uses exact IID uniform samples.

    Preserve chain/draw axes and all repetitions. The Numba RNG is seeded
    separately from default_rng. Initial states alternate ordered/random.
    """
    from numba import njit
    from sampler_wolff import wolff_step_xy
    for name, value in [('chains', chains), ('draws', draws), ('burn', burn), ('interval', interval)]:
        positive_int(value, name)
    if (isinstance(seed, bool) or not isinstance(seed, int) or seed < 0
            or seed + 1009 * (len(times) - 1) + chains - 1 >= 2**32):
        raise ValueError('Wolff seeds must fit the Numba 32-bit RNG range')
    seed_numba = njit(lambda value: np.random.seed(value))
    wolff_step_xy(np.zeros((L, L)), 0.0, J)  # compile before timed production calls
    output = np.empty((chains, draws, len(times), L, L), dtype=np.float64)
    seeds = []
    reuse = {} if reuse is None else reuse
    for f, t in enumerate(times):
        if f in reuse:
            if reuse[f].shape != (chains, draws, L, L) or not np.isfinite(reuse[f]).all():
                raise ValueError('Reused Wolff array does not match the specified chain/draw budget')
            output[:, :, f] = reuse[f]
            seeds.append(None)  # old seeds/burn/interval retained in reference reuse provenance
            continue
        frame_seeds = []
        for c in range(chains):
            current_seed = seed + 1009 * f + c
            frame_seeds.append(current_seed)
            rng = np.random.default_rng(current_seed)
            if t == 0:
                output[c, :, f] = rng.uniform(-math.pi, math.pi, (draws, L, L))
                continue
            seed_numba(current_seed)
            theta = (np.zeros((L, L)) if c % 2 == 0 else rng.uniform(-math.pi, math.pi, (L, L)))
            for _ in range(burn):
                wolff_step_xy(theta, float(t / T_min), J)
            for d in range(draws):
                for _ in range(interval):
                    wolff_step_xy(theta, float(t / T_min), J)
                output[c, d, f] = theta
        seeds.append(frame_seeds)
    return output, seeds


def wolff_diagnostics(observations, chains, draws):
    """Observable diagnostics, including the moments entering susceptibility."""
    from sps_diagnostics import chain_summary
    result = {}
    for key in MOMENTS:
        a = observations[key].reshape(chains, draws, -1)
        result[key] = [chain_summary(a[:, :, t]) for t in range(a.shape[-1])]
    return result
