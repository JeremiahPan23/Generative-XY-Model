"""Observable-specific, finite-run diagnostics; never a proof of equilibration."""
import numpy as np
from xy_observables import energy_density, vortex_pairs


def measurements(theta, J=1.0):
    a = np.asarray(theta, dtype=np.float64)
    result = dict(energy=energy_density(a, J), pairs=vortex_pairs(a),
                  magnetization=np.abs(np.exp(1j*a).mean((-2, -1))))
    for r in range(1, a.shape[-1]//2+1):
        result[f'G_{r}'] = 0.5*(np.cos(a-np.roll(a, r, -1)).mean((-2, -1))
                                + np.cos(a-np.roll(a, r, -2)).mean((-2, -1)))
    return result


def chain_summary(values):
    """ESS using per-chain initial positive autocorrelation pairs, capped at n.

    Classical split R-hat is a warning signal, not rank-normalized R-hat.
    Constant chains have unavailable ESS (not perfect independent sampling).
    SEM additionally accounts conservatively for variation among chain means.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] < 4 or not np.isfinite(x).all():
        raise ValueError('Need finite (chains, draws>=4)')
    m, n = x.shape
    ess, variances = [], []
    for row in x:
        z = row-row.mean()
        var = z.var(ddof=1)
        variances.append(var)
        if var <= 1e-28:
            ess.append(0.0)
            continue
        f = np.fft.rfft(z, n=2*n)
        ac = np.fft.irfft(f*f.conj(), n=2*n)[:n]
        ac /= ac[0]
        pair_sum, previous = 0.0, float('inf')
        for j in range(0, n-1, 2):
            pair = min(float(ac[j]+ac[j+1]), previous)
            if pair <= 0:
                break
            pair_sum += pair
            previous = pair
        tau = max(1.0, -1+2*pair_sum)
        ess.append(n/tau)
    half = n//2
    split = np.concatenate((x[:, :half], x[:, -half:]), axis=0)
    W = split.var(axis=1, ddof=1).mean()
    B = half*split.mean(1).var(ddof=1)
    rhat = float(np.sqrt(((half-1)*W+B)/(half*W))) if W > 1e-28 else None
    valid = all(e > 0 for e in ess)
    sem = None
    if valid:
        within_sem = np.sqrt(sum(v/e for v, e in zip(variances, ess)))/m
        between_sem = x.mean(1).std(ddof=1)/np.sqrt(m) if m > 1 else 0
        sem = float(max(within_sem, between_sem))
    enough = valid and m >= 2 and n >= 256 and min(ess) >= 50 and rhat is not None and rhat < 1.1
    return dict(mean=float(x.mean()), sd=float(x.std(ddof=1)), sem=sem,
                ci95=[float(x.mean()-1.96*sem), float(x.mean()+1.96*sem)] if sem is not None else None,
                ess=float(sum(ess)) if valid else None, ess_by_chain=ess,
                split_rhat=rhat, chain_means=x.mean(1).tolist(),
                first_half_mean=float(x[:, :half].mean()), last_half_mean=float(x[:, -half:].mean()),
                screening_pass=bool(enough),
                note='Approximate stationarity-dependent ESS/CI; screening is not a mixing proof.')
