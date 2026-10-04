"""CPU-only audit of saved SPS results; does not generate or train samples."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sps_diagnostics import measurements, chain_summary


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def numerically_equal(a, b):
    """Allow float64 platform roundoff, but compare decisions/text exactly."""
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(numerically_equal(v, b[k]) for k, v in a.items())
    if isinstance(a, (list, float)):
        return bool(np.allclose(a, b, rtol=1e-12, atol=1e-14))
    return a == b


def energy_loop(theta, J):
    result = np.zeros(theta.shape[:-2], dtype=np.float64)
    a = theta.astype(np.float64)
    L = a.shape[-1]
    for i in range(L):
        for j in range(L):
            result -= J * (np.cos(a[..., i, j] - a[..., (i+1) % L, j])
                           + np.cos(a[..., i, j] - a[..., i, (j+1) % L]))
    return result / (2*L*L)


def longest_rejections(row):
    longest = current = 0
    for accepted in row:
        current = 0 if accepted else current + 1
        longest = max(longest, current)
    return longest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--evaluation', default='eval_25906')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    report = json.loads((args.run / args.evaluation / 'summary.json').read_text())
    ref = report['reference']
    checks = {f'source:{name}': digest(ROOT / name) == expected
              for name, expected in report['sources'].items()}
    checks['checkpoint_hash'] = digest(args.run / 'train/checkpoint.pt') == report['checkpoint_sha256']
    checks['reference_hash'] = digest(args.run / 'reference/samples.npz') == ref['samples_sha256']
    checks['wolff_source_hash'] = digest(ROOT / 'src/sampler_wolff.py') == ref['sampler_sha256']
    with np.load(args.run / 'reference/samples.npz') as archive:
        ref_obs = measurements(archive['theta'], report['config']['J'])
    checks['reference_summaries'] = all(numerically_equal(chain_summary(v), report['reference_summaries'][k])
                                      for k, v in ref_obs.items())
    details = []
    fig, axes = plt.subplots(len(report['runs']), 2, figsize=(12, 8), squeeze=False)
    for row, run in enumerate(report['runs']):
        n = run['steps']
        path = args.run / args.evaluation / f'steps_{n}.npz'
        checks[f'N{n}:archive_hash'] = digest(path) == run['samples_sha256']
        with np.load(path) as archive:
            a = {k: archive[k] for k in archive.files}
        # Independent scalar-chain replay, using recorded random numbers.
        chains, draws = a['accepted'].shape
        burn = report['settings']['burn']
        replay_ok = True
        for c in range(chains):
            current = 0
            for j in range(1, a['proposal_log_weight'].shape[1]):
                logalpha = min(0., a['proposal_log_weight'][c, j] - a['proposal_log_weight'][c, current])
                accepted = bool(a['log_uniform'][c, j-1] < logalpha)
                if accepted:
                    current = j
                if j > burn:
                    k = j-burn-1
                    replay_ok &= (accepted == a['accepted'][c, k]
                                  and a['chain_log_weight'][c, k] == a['proposal_log_weight'][c, current]
                                  and np.array_equal(a['chain_theta'][c, k], a['proposal_theta'][c, current])
                                  and np.isclose(a['alpha'][c, k], np.exp(logalpha), rtol=1e-13, atol=1e-15))
        checks[f'N{n}:independent_MH_replay'] = bool(replay_ok)
        obs = measurements(a['chain_theta'], report['config']['J'])
        checks[f'N{n}:stored_observables'] = all(np.allclose(v, a[k], rtol=0, atol=1e-14) for k, v in obs.items())
        checks[f'N{n}:energy_bond_loop'] = bool(np.allclose(energy_loop(a['chain_theta'], report['config']['J']), a['energy'], rtol=0, atol=1e-14))
        checks[f'N{n}:recomputed_summaries'] = all(numerically_equal(chain_summary(v), run['summaries'][k]) for k, v in obs.items())
        checks[f'N{n}:acceptance'] = float(a['accepted'].mean()) == run['accepted_fraction']
        raw_obs = measurements(a['proposal_theta'], report['config']['J'])
        w = np.exp(a['proposal_log_weight'].ravel() - a['proposal_log_weight'].max())
        ordered_w = np.sort(w)
        worst_streaks = []
        terminal_stays = []
        for c in range(chains):
            longest = current = 0
            end = None
            for k, accepted in enumerate(a['accepted'][c]):
                current = 0 if accepted else current + 1
                if current > longest:
                    longest, end = current, k
            if end is not None:
                worst_streaks.append(dict(chain=c+1, length=longest,
                    first_retained_draw=end-longest+2, last_retained_draw=end+1,
                    log_weight=float(a['chain_log_weight'][c, end]),
                    energy=float(a['energy'][c, end]), pairs=float(a['pairs'][c, end])))
            accepted_indices = np.flatnonzero(a['accepted'][c])
            last = int(accepted_indices[-1]) if len(accepted_indices) else -1
            terminal_stays.append(dict(chain=c+1,
                last_accepted_retained_draw=last+1 if last >= 0 else None,
                trailing_rejections=draws-last-1,
                energy=float(a['energy'][c, -1]), pairs=float(a['pairs'][c, -1]),
                log_weight=float(a['chain_log_weight'][c, -1])))
        d = dict(steps=n, acceptance=run['accepted_fraction'],
                 max_observable_roundoff={k: float(np.max(np.abs(v-a[k]))) for k, v in obs.items()},
                 sampling_seconds=run['sampling_seconds'],
                 longest_rejection_streak_by_chain=[longest_rejections(x) for x in a['accepted']],
                 worst_streaks=worst_streaks,
                 terminal_stays=terminal_stays,
                 importance_ess_fraction=float(w.sum()**2 / (len(w) * np.square(w).sum())),
                 largest_normalized_importance_weight=float(ordered_w[-1]/w.sum()),
                 top10_normalized_importance_weight=float(ordered_w[-10:].sum()/w.sum()),
                 vortex_nonzero_draws_by_chain=np.count_nonzero(a['pairs'], axis=1).tolist(),
                 accepted_vortex_proposals_by_chain=np.sum((a['pairs'] > 0) & a['accepted'], axis=1).tolist(),
                 raw_proposal_means={k: float(v.mean()) for k, v in raw_obs.items()},
                 corrected_means={k: float(v.mean()) for k, v in obs.items()},
                 failed_observable_checks=[k for k, v in run['physical_checks'].items() if not v['screening_pass']],
                 absolute_bias_in_combined_sem={k: abs(v['bias'])/v['combined_sem'] for k, v in run['physical_checks'].items()})
        details.append(d)
        block_size = 64 if draws <= 2048 else 512
        for c in range(chains):
            # Blocks visualize chain disagreement without removing repeated states.
            blocks = a['energy'][c, :draws//block_size*block_size].reshape(-1, block_size).mean(1)
            axes[row, 0].plot(np.arange(len(blocks))*block_size+block_size/2, blocks, label=f'Chain {c+1}', lw=1)
            axes[row, 1].plot(np.arange(draws)+1, np.cumsum(a['pairs'][c])/(np.arange(draws)+1), lw=1)
        axes[row, 0].axhline(ref['summaries']['energy']['mean'], color='black', ls='--', label='Wolff mean')
        axes[row, 1].axhline(ref['summaries']['pairs']['mean'], color='black', ls='--')
        axes[row, 0].set(ylabel=f'N={n}: energy / bond', ylim=(-1, -.3))
        axes[row, 1].set(ylabel=f'N={n}: cumulative pairs', ylim=(0, .15))
        for ax in axes[row]:
            ax.grid(alpha=.2)
    axes[0, 0].set_title(f'{block_size}-draw block means (repeated states retained)')
    axes[0, 1].set_title('Rare vortices: cumulative mean in each chain')
    axes[0, 0].legend(ncol=3, fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel('Retained MH draw')
    passed = sum(r['screening_pass'] for r in report['runs'])
    fig.suptitle(f'L={report["config"]["L"]}, seed={report["config"]["seed"]}: '
                 f'{passed}/{len(report["runs"])} evaluations pass the preset screening')
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out / 'chain_diagnostics.png', dpi=160)
    plt.close(fig)
    audit = dict(checks=checks, all_audit_checks_pass=all(checks.values()), runs=details,
                 limitations=['Replays MH using saved weights; intermediate paths were not saved, so full path densities cannot be independently recomputed.',
                              'Observable summaries reuse the original diagnostic estimator; bond energy and MH replay are independent calculations.',
                              'Rejection streaks are restricted to the retained window and may be censored at its boundaries.',
                              'A successful data audit is not a successful physical screening or proof of mixing.'])
    (args.out / 'audit.json').write_text(json.dumps(audit, indent=2), encoding='utf-8')
    print(json.dumps(audit, indent=2))
    if not audit['all_audit_checks_pass']:
        raise SystemExit('Audit discrepancy: inspect audit.json')


if __name__ == '__main__':
    main()
