"""Score an exact replay of saved raw paths without changing frozen samplers.

The proposal is a complete prefix path, not only its endpoint.  Replaying the
same RNG stream is necessary because saved intermediate endpoints alone do not
determine the transition densities along that path.  Every saved endpoint must
match bit for bit before any result is returned.
"""
import math

import numpy as np
import torch

from sps_intermediate import make_grid, positive_int
from sps_xy import energy, independence_mh, wrapped_log_prob


def _seed(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < 2**63:
        raise ValueError(f'{name} must be an integer in [0, 2**63)')
    return value


def _validated_mesh(mesh):
    """Require the exact mesh produced by the existing capture-grid builder."""
    if not isinstance(mesh, dict):
        raise ValueError('Expected a mesh dictionary')
    required = ('base_steps', 'capture_times', 'policy', 'grid', 'dt', 'capture_indices')
    if any(key not in mesh for key in required):
        raise ValueError('Incomplete mesh')
    try:
        canonical = make_grid(mesh['base_steps'], mesh['capture_times'], mesh['policy'])
    except (TypeError, IndexError, OverflowError) as exc:
        raise ValueError('Malformed mesh') from exc
    for name in ('grid', 'dt', 'capture_times', 'capture_indices'):
        value = np.asarray(mesh[name])
        if value.shape != canonical[name].shape or not np.array_equal(value, canonical[name]):
            raise ValueError(f'Mesh {name} is not the exact approved capture mesh')
    if not np.issubdtype(np.asarray(mesh['capture_indices']).dtype, np.integer):
        raise ValueError('Capture indices must be integers')
    return canonical


@torch.inference_mode()
def collect_scored_replay(model, mesh, saved_theta, batch_size, seed, progress=None):
    """Replay and score the original forward stream, checking ALL saved frames.

    ``saved_theta`` has shape (paths, frames, L, L).  The forward arithmetic and
    RNG calls are identical to ``collect_raw_frames``; backward drift and density
    evaluations consume no random numbers.  Densities and prefix weights use
    float64, including the initial uniform density.  Any endpoint mismatch is a
    hard error, never a tolerance-based match.

    Returns a dictionary containing checked ``theta``, ``log_qf``, ``log_qb``,
    ``log_weight`` and ``energy_total``; the latter four have shape (paths,
    frames).  ``path_id``, ``capture_times``, ``capture_indices`` and ``grid``
    preserve the original row and time-node identities.  The caller must pin
    the model/checkpoint, execution environment and saved-frame provenance.
    """
    positive_int(batch_size, 'batch_size')
    _seed(seed, 'path seed')
    mesh = _validated_mesh(mesh)
    if not hasattr(model, 'config') or not callable(getattr(model.config, 'validate', None)):
        raise ValueError('A validated frozen model configuration is required')
    model.config.validate()
    parameters = list(model.parameters())
    if not parameters:
        raise ValueError('Model has no parameters defining replay dtype/device')
    param = parameters[0]
    if param.dtype not in (torch.float32, torch.float64):
        raise ValueError('Replay requires float32 or float64 model parameters')
    if any(p.device != param.device or p.dtype != param.dtype
           or not bool(torch.isfinite(p).all()) for p in parameters):
        raise ValueError('Model parameters must be finite and have one dtype/device')
    if not callable(getattr(model, 'drift_backward', None)):
        raise ValueError('Frozen model must provide drift_backward')
    saved = np.asarray(saved_theta)
    L = model.config.L
    frames = len(mesh['capture_times'])
    dtype = param.detach().cpu().numpy().dtype
    if (saved.ndim != 4 or saved.shape[0] < 1
            or saved.shape[1:] != (frames, L, L) or saved.dtype != dtype
            or not np.isfinite(saved).all()):
        raise ValueError('Saved frames must be finite (paths, frames, L, L) in the model dtype')
    count = saved.shape[0]
    grid, wanted = mesh['grid'], mesh['capture_indices']
    captures = {int(index): slot for slot, index in enumerate(wanted)}
    output = np.empty_like(saved)
    scores = {key: np.empty((count, frames), dtype=np.float64)
              for key in ('log_qf', 'log_qb', 'log_weight', 'energy_total')}
    generator = torch.Generator(device=param.device).manual_seed(seed)
    model.eval()

    def capture(first, n, slot, theta, log_qf, log_qb):
        actual = theta.cpu().numpy()
        expected = saved[first:first+n, slot]
        if not np.array_equal(actual, expected):
            mismatch = np.flatnonzero(np.any(actual != expected, axis=(-2, -1)))
            path = first + int(mismatch[0])
            raise ValueError(f'Bitwise replay mismatch at path {path}, frame {slot}; '
                             'no scores may be treated as a replay of the saved paths')
        total_energy = energy(theta.double(), model.config.J)
        beta = float(mesh['capture_times'][slot]) / model.config.T_min
        values = dict(log_qf=log_qf, log_qb=log_qb, energy_total=total_energy,
                      log_weight=-beta * total_energy + log_qb - log_qf)
        if not all(bool(torch.isfinite(value).all()) for value in values.values()):
            raise ValueError(f'Non-finite prefix score at frame {slot}')
        output[first:first+n, slot] = actual
        for key, value in values.items():
            scores[key][first:first+n, slot] = value.cpu().numpy()

    for first in range(0, count, batch_size):
        n = min(batch_size, count - first)
        shape = (n, L, L)
        # Keep these forward expressions and RNG calls equal to collect_raw_frames.
        theta = 2 * math.pi * torch.rand(shape, device=param.device,
                                        dtype=param.dtype, generator=generator) - math.pi
        log_qf = torch.full((n,), -L * L * math.log(2 * math.pi),
                            device=param.device, dtype=torch.float64)
        log_qb = torch.zeros_like(log_qf)
        capture(first, n, captures[0], theta, log_qf, log_qb)
        for index, (t0, t1) in enumerate(zip(grid[:-1], grid[1:]), 1):
            t0, t1 = float(t0), float(t1)
            dt = t1 - t0
            std = model.sigma((t0 + t1) / 2) * math.sqrt(dt)
            mean = theta + dt * model.forward_drift(theta, t0)
            eps = torch.randn(shape, device=param.device, dtype=param.dtype, generator=generator)
            next_theta = torch.remainder(mean + std * eps + math.pi, 2 * math.pi) - math.pi
            mean_backward = next_theta + dt * model.drift_backward(next_theta, t1)
            log_qf = log_qf + wrapped_log_prob(next_theta, mean, std).sum((-2, -1))
            log_qb = log_qb + wrapped_log_prob(theta, mean_backward, std).sum((-2, -1))
            theta = next_theta
            if index in captures:
                capture(first, n, captures[index], theta, log_qf, log_qb)
        if progress:
            progress(first + n, count)
    return dict(theta=output, **scores, path_id=np.arange(count, dtype=np.int64),
                capture_times=mesh['capture_times'].copy(),
                capture_indices=mesh['capture_indices'].copy(), grid=grid.copy())


def _validated_frame_metadata(frames_metadata, frames):
    if not isinstance(frames_metadata, (list, tuple)) or len(frames_metadata) != frames:
        raise ValueError('Frame metadata must align with the saved frame axis')
    times = []
    previous = -1.0
    inferred_T_min = None
    for index, metadata in enumerate(frames_metadata):
        if not isinstance(metadata, dict) or any(
                key not in metadata for key in ('index', 'time', 'beta', 'temperature')):
            raise ValueError('Incomplete frame metadata')
        if (isinstance(metadata['index'], bool) or not isinstance(metadata['index'], int)
                or metadata['index'] != index):
            raise ValueError('Frame indices must preserve their original order')
        t, beta, temperature = metadata['time'], metadata['beta'], metadata['temperature']
        if (isinstance(t, bool) or isinstance(beta, bool)
                or not isinstance(t, (int, float)) or not isinstance(beta, (int, float))
                or not math.isfinite(t) or not math.isfinite(beta)
                or not previous < t <= 1 or t < 0 or beta < 0):
            raise ValueError('Frame times and inverse temperatures must be finite and increasing')
        if t == 0:
            if index != 0 or beta != 0 or temperature is not None:
                raise ValueError('The zero-time frame must represent beta=0, T=infinity')
        else:
            if (beta <= 0 or isinstance(temperature, bool)
                    or not isinstance(temperature, (int, float))
                    or not math.isfinite(temperature) or temperature <= 0
                    or not math.isclose(beta * temperature, 1.0, rel_tol=1e-13, abs_tol=1e-15)):
                raise ValueError('Finite-temperature frame has inconsistent beta and T')
            T_min = t / beta
            if inferred_T_min is None:
                inferred_T_min = T_min
            elif not math.isclose(T_min, inferred_T_min, rel_tol=1e-13, abs_tol=1e-15):
                raise ValueError('Frames do not share one absolute temperature schedule')
        previous = t
        times.append(t)
    if frames < 2 or times[0] != 0 or times[-1] != 1:
        raise ValueError('Scored frame axis must include time zero and time one')


def evaluate_scored_paths(theta, log_weight, frames_metadata, *, chains=4,
                          candidates_per_chain=1024, burn=128, seed=710011):
    """Offline prefix-path IMH and stable importance ESS for each temperature.

    Consecutive path rows form complete chains.  Each chain uses one initial
    path plus ``candidates_per_chain - 1`` independent proposals.  The first
    ``burn`` transitions are discarded; rejection repeats remain in every saved
    output.  The displayed acceptance is actual retained transition acceptance,
    not mean alpha.  Importance ESS uses ALL candidates, including initial and
    burn-in paths, with denominator ``chains * candidates_per_chain``.

    The return value has ``summary`` (JSON-compatible per-frame rows), ``frames``
    (per-frame dictionaries with proposals, uniforms and all frozen-MH arrays),
    and ``config``.  Uniform seeds are ``seed + 1009 * frame_index``.  Different
    arms may use the same seed to retain paired MH uniforms.  Time zero is saved
    but marked ``include_in_temperature_plot=False``.
    """
    positive_int(chains, 'chains')
    positive_int(candidates_per_chain, 'candidates_per_chain')
    _seed(seed, 'MH seed')
    if (isinstance(burn, bool) or not isinstance(burn, int)
            or not 0 <= burn < candidates_per_chain - 1):
        raise ValueError('Burn-in must leave at least one retained transition')
    theta, log_weight = np.asarray(theta), np.asarray(log_weight)
    count = chains * candidates_per_chain
    if (theta.ndim != 4 or theta.shape[0] != count or theta.shape[-1] < 3
            or theta.shape[-1] != theta.shape[-2]
            or not np.issubdtype(theta.dtype, np.floating)
            or log_weight.shape != theta.shape[:2]
            or not np.issubdtype(log_weight.dtype, np.floating)
            or not np.isfinite(theta).all() or not np.isfinite(log_weight).all()):
        raise ValueError('Expected aligned finite floating theta[path, frame, L, L] and weights[path, frame]')
    frames = theta.shape[1]
    _validated_frame_metadata(frames_metadata, frames)
    if seed + 1009 * (frames - 1) >= 2**63:
        raise ValueError('Derived MH frame seed exceeds supported range')
    retained_draws = candidates_per_chain - 1 - burn
    retained_count = chains * retained_draws
    summaries, frame_arrays = [], []
    for index, metadata in enumerate(frames_metadata):
        proposal_theta = theta[:, index].reshape(chains, candidates_per_chain,
                                                  theta.shape[-2], theta.shape[-1])
        proposal_weight = log_weight[:, index].astype(np.float64).reshape(chains, candidates_per_chain)
        frame_seed = seed + 1009 * index
        rng = np.random.default_rng(frame_seed)
        # random() is in [0,1); log1p(-u) is finite and represents log U on (0,1].
        log_uniform = np.log1p(-rng.random((chains, candidates_per_chain - 1)))
        mh = independence_mh(proposal_theta, proposal_weight, log_uniform, burn)
        with np.errstate(over='ignore', under='ignore'):
            scaled = np.exp(proposal_weight.ravel() - proposal_weight.max())
        normalized = scaled / scaled.sum()
        importance_ess = float(1.0 / np.square(normalized).sum())
        accepted_count = int(mh['accepted'].sum())
        row = dict(index=index, time=float(metadata['time']), beta=float(metadata['beta']),
                   temperature=(None if metadata['temperature'] is None
                                else float(metadata['temperature'])),
                   include_in_temperature_plot=metadata['time'] > 0,
                   candidate_count=count, retained_count=retained_count,
                   retained_draws_per_chain=retained_draws,
                   attempted_transitions_per_chain=candidates_per_chain - 1,
                   burn_transitions_per_chain=burn,
                   accepted_count=accepted_count,
                   accepted_fraction=accepted_count / retained_count,
                   acceptance_by_chain=mh['accepted'].mean(axis=1).tolist(),
                   mean_retained_alpha=float(mh['alpha'].mean()),
                   importance_ess=importance_ess,
                   importance_ess_fraction=importance_ess / count,
                   largest_normalized_importance_weight=float(normalized.max()),
                   mh_uniform_seed=frame_seed)
        summaries.append(row)
        frame_arrays.append(dict(proposal_theta=proposal_theta.copy(),
                                 proposal_log_weight=proposal_weight.copy(),
                                 log_uniform=log_uniform, **mh))
    return dict(summary=summaries, frames=frame_arrays,
                config=dict(chains=chains, candidates_per_chain=candidates_per_chain,
                            burn=burn, seed=seed, frame_seed_stride=1009,
                            importance_ess_denominator='all candidate paths, including initialization and burn-in',
                            acceptance_denominator='retained attempted transitions only; initialization excluded',
                            retained_records_include_rejections=True))
