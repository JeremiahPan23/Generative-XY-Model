"""Explain one recorded SPS failure using saved data only (CPU, no sampling).

Outputs are derived analysis, separate from immutable experiment records.
Log Q_b - log Q_f is inferred from the saved weight identity, not recomputed
from unrecorded intermediate paths. All draw numbers in outputs are one-based.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'results/sps_a/L4_seed22_replica_26634'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bond_energy(theta, J):
    """Independent explicit bond loop, returning energy per bond."""
    a = np.asarray(theta, dtype=np.float64)
    L = a.shape[-1]
    result = np.zeros(a.shape[:-2])
    for i in range(L):
        for j in range(L):
            result -= J * (np.cos(a[..., i, j] - a[..., (i+1) % L, j])
                           + np.cos(a[..., i, j] - a[..., i, (j+1) % L]))
    return result / (2*L*L)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/sps_sticking_mechanism')
    args = parser.parse_args()
    summary_path = RUN/'eval_long/summary.json'
    report = json.loads(summary_path.read_text())
    audit_path = RUN/'analysis/audit.json'
    audit = json.loads(audit_path.read_text())
    run = next(r for r in report['runs'] if r['steps'] == 32)
    details = next(r for r in audit['runs'] if r['steps'] == 32)
    stay = max(details['worst_streaks'], key=lambda r: r['length'])
    c = stay['chain']-1
    entry = stay['first_retained_draw']-1  # accepted draw immediately before stay
    burn = report['settings']['burn']
    config = report['config']
    npz_path = RUN/'eval_long/steps_32.npz'
    checks = {
        'prior_audit_passed': audit['all_audit_checks_pass'],
        'proposal_archive_hash_matches_summary': digest(npz_path) == run['samples_sha256'],
        **{f'source_hash:{name}': digest(ROOT/name) == expected
           for name, expected in report['sources'].items()},
    }
    with np.load(npz_path, allow_pickle=False) as archive:
        a = {key: archive[key] for key in archive.files}
    if entry < 1:
        raise ValueError('Selected case must enter the stuck state in retained records')
    end = stay['last_retained_draw']
    held_w = float(a['chain_log_weight'][c, entry-1])
    held_e = float(bond_energy(a['chain_theta'][c, entry-1], config['J']))
    sl = slice(entry, end)  # retained zero-based rejection records
    raw_sl = slice(burn+entry+1, burn+end+1)  # matching candidate indices
    proposal_w = a['proposal_log_weight'][c, raw_sl]
    proposal_e = bond_energy(a['proposal_theta'][c, raw_sl], config['J'])
    delta = proposal_w-held_w
    alphas = np.exp(np.minimum(0., delta))
    uniforms = np.exp(a['log_uniform'][c, burn+entry:burn+end])
    checks.update({
        'entry_was_accepted': bool(a['accepted'][c, entry-1]),
        'entry_endpoint_matches_proposal': np.array_equal(
            a['chain_theta'][c, entry-1], a['proposal_theta'][c, burn+entry]),
        'entry_weight_matches_proposal': bool(held_w == a['proposal_log_weight'][c, burn+entry]),
        'record_count_matches_stay': len(proposal_w) == stay['length'],
        'all_rejection_flags': bool(np.all(~a['accepted'][c, sl])),
        'endpoint_exactly_unchanged': bool(np.all(
            a['chain_theta'][c, sl] == a['chain_theta'][c, entry-1])),
        'old_path_weight_exactly_unchanged': bool(np.all(a['chain_log_weight'][c, sl] == held_w)),
        'probabilities_match_saved_alpha': bool(np.allclose(alphas, a['alpha'][c, sl], rtol=1e-13, atol=1e-15)),
        'every_recorded_random_test_rejects': bool(np.all(
            a['log_uniform'][c, burn+entry:burn+end] >= np.minimum(0., delta))),
        'energy_matches_independent_bond_loop': bool(np.allclose(
            a['energy'][c, sl], held_e, rtol=0., atol=1e-14)),
    })
    # Independent scalar MH replay for the entire selected chain, including burn.
    current = 0
    replay_ok = True
    for j in range(1, a['proposal_log_weight'].shape[1]):
        logalpha = min(0., a['proposal_log_weight'][c, j]-a['proposal_log_weight'][c, current])
        accepted = a['log_uniform'][c, j-1] < logalpha
        if accepted:
            current = j
        if j > burn:
            k = j-burn-1
            replay_ok &= (accepted == a['accepted'][c, k]
                          and a['chain_log_weight'][c, k] == a['proposal_log_weight'][c, current]
                          and np.array_equal(a['chain_theta'][c, k], a['proposal_theta'][c, current]))
    checks['independent_selected_chain_MH_replay'] = bool(replay_ok)
    if not all(checks.values()):
        raise ValueError(f'Failed recorded-data checks: {checks}')

    factor = config['beta']*2*config['L']**2
    current_terms = dict(log_weight=held_w, energy_per_bond=held_e,
                         minus_beta_H=-factor*held_e,
                         inferred_log_Qb_minus_log_Qf=held_w+factor*held_e)
    examples = []
    for label, k in [('first_rejected', 0), ('lowest_energy', int(np.argmin(proposal_e))),
                     ('largest_weight', int(np.argmax(proposal_w)))]:
        examples.append(dict(label=label, retained_draw=entry+k+1,
            energy_per_bond=float(proposal_e[k]), log_weight=float(proposal_w[k]),
            minus_beta_H=float(-factor*proposal_e[k]),
            inferred_log_Qb_minus_log_Qf=float(proposal_w[k]+factor*proposal_e[k]),
            delta_log_weight=float(delta[k]), acceptance_probability=float(alphas[k]),
            recorded_uniform=float(uniforms[k]), accepted=False))
    # Estimate the fixed state's replacement probability from OTHER chains only.
    other = np.delete(a['proposal_log_weight'], c, axis=0)
    next_alpha = np.exp(np.minimum(0., other-held_w))
    probability = float(next_alpha.mean())
    xs = np.linspace(np.quantile(other, .05), held_w, 100)
    ys = np.asarray([np.exp(np.minimum(0., other-x)).mean() for x in xs])
    output = dict(
        config=config, steps=32, chain=c+1, entry_retained_draw=entry,
        first_rejection_retained_draw=entry+1, last_rejection_retained_draw=end,
        recorded_consecutive_rejections=len(proposal_w),
        terminal_stay_censored=True, retained_draws=a['accepted'].shape[1],
        retained_fraction_in_rejection_streak=len(proposal_w)/a['accepted'].shape[1],
        whole_group_acceptance=run['accepted_fraction'],
        current_state={**current_terms, 'vortex_pairs': stay['pairs']},
        first_rejected_candidate=examples[0], rejected_examples=examples,
        candidates_with_lower_energy=int(np.sum(proposal_e < held_e)),
        candidates_with_at_least_current_weight=int(np.sum(proposal_w >= held_w)),
        proposal_log_weight_quantiles=dict(zip(['min', 'q05', 'median', 'q95', 'max'],
            np.quantile(proposal_w, [0., .05, .5, .95, 1.]).tolist())),
        other_chain_proposal_count=other.size,
        estimated_next_acceptance=probability,
        estimated_next_acceptance_by_other_chain=next_alpha.mean(axis=1).tolist(),
        estimated_mean_attempts_until_acceptance=1/probability,
        estimated_probability_at_least_observed_rejections=float(np.exp(len(proposal_w)*np.log1p(-probability))),
        checks=checks, all_checks_pass=True,
        source_files={str(p.relative_to(ROOT)).replace('\\', '/'): digest(p)
                      for p in (npz_path, summary_path, audit_path, Path(__file__).resolve())},
        limitations=[
            'Case selected after observing the worst retained stay; not a preselected statistical test.',
            'Run ended during the stay: 12406 is the observed rejection count, not a completed waiting time.',
            'Log Qb - log Qf is algebraically inferred; intermediate paths were not saved, so their densities are not independently recomputed.',
            'Held-out-chain estimates use a finite proposal bank; rare unseen weights may affect probability and waiting-time estimates.',
            'Observed MH mechanism does not establish why training produced this tail, or prove all chains equilibrated.',
            'Zero vortices refers only to the saved endpoint, not to its unrecorded path.',
        ])
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'mechanism.json').write_text(json.dumps(output, indent=2)+'\n', encoding='utf-8')

    # A single scientific figure, English labels for later report reuse.
    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    draws = np.arange(1, a['accepted'].shape[1]+1)
    ax = axes[0, 0]
    ax.plot(draws, a['energy'][c], color='#2475b5', lw=.6)
    ax.axhline(report['reference']['summaries']['energy']['mean'], color='black', ls='--', lw=1, label='Wolff reference mean')
    ax.axvspan(entry+1, end, color='#f1c8c7', alpha=.55)
    ax.text(.53, .84, f'{len(proposal_w):,} rejections\nSame endpoint retained\n0 endpoint vortices',
            transform=ax.transAxes, ha='center', va='top')
    ax.set(title='A | The recorded chain stops changing', xlabel='Retained MH draw', ylabel='Energy per bond')
    ax.legend(loc='lower right', fontsize=9)

    ax = axes[0, 1]
    retained_proposals = a['proposal_log_weight'][c, burn+1:]
    ax.plot(draws, retained_proposals, color='#93bfe0', lw=.4, alpha=.85, label='New candidate (every attempt)')
    ax.plot(draws, a['chain_log_weight'][c], color='#a42d36', lw=1.1, label='Currently retained path')
    ax.axvspan(entry+1, end, color='#f1c8c7', alpha=.25)
    ax.set(title='B | Candidates continue; their weights are lower', xlabel='Retained MH draw', ylabel='Path log weight w')
    ax.legend(loc='lower right', fontsize=9)

    ax = axes[1, 0]
    candidate = examples[0]
    energy_terms = [current_terms['minus_beta_H'], candidate['minus_beta_H']]
    ratio_terms = [current_terms['inferred_log_Qb_minus_log_Qf'], candidate['inferred_log_Qb_minus_log_Qf']]
    ax.bar([0, 1], energy_terms, width=.6, color='#e7b453', label=r'Energy term $-\beta H$')
    ax.bar([0, 1], ratio_terms, bottom=energy_terms, width=.6, color='#548eaf', label='Inferred path probability term')
    for i, (e, r) in enumerate(zip(energy_terms, ratio_terms)):
        ax.text(i, e/2, f'{e:.2f}', ha='center', va='center')
        ax.text(i, e+r/2, f'{r:.2f}', ha='center', va='center', color='white')
        ax.text(i, e+r+1, f'w = {e+r:.2f}', ha='center')
    ax.set(title='C | Lower energy does not guarantee acceptance', ylabel='Contributions to path log weight',
           xticks=[0, 1], xticklabels=[f'Held state\ne = {held_e:.3f}', f'First rejected candidate\ne = {candidate["energy_per_bond"]:.3f}'], ylim=(0, 68))
    ax.legend(loc='upper right', fontsize=8)

    ax = axes[1, 1]
    ax.plot(xs, ys*100, color='#2475b5', lw=2)
    ax.scatter([held_w], [probability*100], color='#a42d36', s=45, zorder=5)
    ax.set_yscale('log')
    ax.set_yticks([100, 10, 1, .1, .01])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:g}'))
    ax.annotate(f'Actual held state\nEstimated next acceptance: {probability*100:.4f}%\nMean wait: about {1/probability:,.0f} attempts',
        xy=(held_w, probability*100), xytext=(.04, .11), textcoords='axes fraction',
        arrowprops=dict(arrowstyle='->', color='#a42d36'), fontsize=9)
    ax.set(title='D | This particular state is hard to replace', xlabel='Hypothetical current path log weight w',
           ylabel='Estimated next acceptance (%)')
    for panel in axes.ravel():
        panel.grid(alpha=.15)
        panel.set_axisbelow(True)
    fig.suptitle('Why one XY SPS chain sticks | L=4, beta=1.12, seed=22, N=32, chain 5', fontsize=14)
    fig.text(.5, .019, 'A-B: recorded data. C: decomposition inferred from saved weights and endpoint energies.\n'
             'D: finite candidate-bank estimate using the other seven chains; not a mixing proof. The stay continues at run end.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .95), h_pad=2)
    fig.savefig(args.out/'mechanism_explained.png', dpi=180)
    fig.savefig(args.out/'mechanism_explained.pdf')
    plt.close(fig)
    print(f'All {len(checks)} recorded-data checks passed; wrote CPU-only analysis to {args.out}')
    print(f'First rejected candidate: energy={candidate["energy_per_bond"]:.6f}, '
          f'alpha={candidate["acceptance_probability"]:.9g}, uniform={candidate["recorded_uniform"]:.6f}')


if __name__ == '__main__':
    main()
