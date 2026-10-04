"""Compare the three preselected SPS training seeds using saved CPU data only."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RUNS = [(11, 'L4_seed11_full_25906', 'eval_long_25955', 'analysis/eval_long_25955'),
        (22, 'L4_seed22_replica_26634', 'eval_long', 'analysis'),
        (33, 'L4_seed33_replica_26635', 'eval_long', 'analysis')]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/sps_a_seed_replication')
    args = parser.parse_args()
    protocol = json.loads((ROOT/'configs/sps_a_replication.json').read_text())
    rows, reports = [], []
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    for seed, folder, evaluation, audit_path in RUNS:
        path = ROOT/'results/sps_a'/folder
        report = json.loads((path/evaluation/'summary.json').read_text())
        audit = json.loads((path/audit_path/'audit.json').read_text())
        if (report['config'] != dict(protocol['baseline_config'], seed=seed)
                or report['settings'] != protocol['evaluation_settings']
                or report['reference'] != protocol['reference_metadata']
                or not audit['all_audit_checks_pass']):
            raise ValueError(f'Seed {seed} does not match the verified protocol')
        reports.append(report)
        ns = [r['steps'] for r in report['runs']]
        color = {11: 'tab:blue', 22: 'tab:orange', 33: 'tab:green'}[seed]
        vals = [[100*r['accepted_fraction'] for r in report['runs']],
                [r['sampling_seconds'] for r in report['runs']],
                [r['ess_per_second']['energy'] for r in report['runs']],
                [r['summaries']['pairs']['mean'] for r in report['runs']]]
        for panel, values in zip(axes.ravel(), vals):
            panel.plot(ns, values, 'o-', color=color, label=f'Seed {seed}')
            for n, value, run in zip(ns, values, report['runs']):
                if not run['screening_pass']:
                    panel.scatter(n, value, marker='x', color='crimson', s=110, zorder=5)
        pair_sem = [r['summaries']['pairs']['sem'] for r in report['runs']]
        axes[1, 1].errorbar(ns, vals[3], yerr=1.96*np.asarray(pair_sem), fmt='none', color=color, capsize=3)
        for run, details in zip(report['runs'], audit['runs']):
            individual_ess = np.asarray(run['summaries']['energy']['ess_by_chain'])
            harmonic_ess = len(individual_ess)**2 / np.sum(1/individual_ess)
            rows.append(dict(seed=seed, steps=run['steps'], acceptance=run['accepted_fraction'],
                sampling_seconds=run['sampling_seconds'], energy_ess_per_second=run['ess_per_second']['energy'],
                equal_variance_pooled_mean_energy_ess_per_second=float(harmonic_ess/run['sampling_seconds']),
                magnetization_ess_per_second=run['ess_per_second']['magnetization'],
                pairs_ess_per_second=run['ess_per_second']['pairs'], G2_ess_per_second=run['ess_per_second']['G_2'],
                energy=run['summaries']['energy']['mean'], energy_sem=run['summaries']['energy']['sem'],
                pairs=run['summaries']['pairs']['mean'], pairs_sem=run['summaries']['pairs']['sem'],
                screening_pass=run['screening_pass'],
                failed_observable_checks=details['failed_observable_checks'],
                longest_retained_rejection_streak=max(details['longest_rejection_streak_by_chain']),
                importance_ess_fraction=details['importance_ess_fraction']))
    ref = protocol['reference_metadata']['summaries']['pairs']
    axes[1, 1].axhline(ref['mean'], color='black', ls='--', label='Wolff')
    axes[1, 1].axhspan(ref['mean']-1.96*ref['sem'], ref['mean']+1.96*ref['sem'], color='black', alpha=.1)
    labels = [('A: Acceptance increases for every seed', 'Accepted proposals (%)'),
              ('B: More path steps cost more time', 'Generation + MH time (seconds)'),
              ('C: Energy efficiency depends on training seed', 'Estimated energy ESS / second'),
              ('D: Vortex counts (approximate 95% intervals)', 'Mean vortex pairs')]
    for ax, (title, ylabel) in zip(axes.ravel(), labels):
        ax.set(title=title, ylabel=ylabel, xlabel='Path steps N')
        ax.set_xscale('log', base=2)
        ax.set_xticks([32, 64, 128], ['32', '64', '128'])
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    axes[1, 1].legend(fontsize=8)
    fig.suptitle('XY SPS | L=4 | fixed training/evaluation budgets | three preselected seeds')
    fig.text(.5, .01, 'Red x: failed full screening; corresponding ESS/s and intervals are provisional. No chains pooled across models.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, .96))
    args.out.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out/'three_seed_comparison.png', dpi=170)
    plt.close(fig)
    note = ('One evaluation stream per model with common evaluation RNG seeds; three training seeds do not '
            'prove a universal optimum. ESS/CI assume sufficiently stationary chains. Retained repeated states included.')
    (args.out/'comparison.json').write_text(json.dumps(dict(rows=rows, note=note), indent=2)+'\n')
    sensitivity = [dict(seed=row['seed'], steps=row['steps'],
                        sum_chain_ess_per_second=row['energy_ess_per_second'],
                        equal_variance_pooled_mean_ess_per_second=row['equal_variance_pooled_mean_energy_ess_per_second'],
                        screening_pass=row['screening_pass']) for row in rows]
    (args.out/'ess_sensitivity.json').write_text(json.dumps(dict(rows=sensitivity,
        note='Sensitivity diagnostic only: m^2/sum(1/ESS_i) assumes independent chains with a common stationary '
             'single-draw variance. Poorly mixed cases remain provisional; not a definitive corrected estimator.'), indent=2)+'\n')
    with (args.out/'comparison.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    # Explain the most severe observed stay with other chains' independent proposals.
    path = ROOT/'results/sps_a/L4_seed22_replica_26634'
    audit = json.loads((path/'analysis/audit.json').read_text())
    stay = max(audit['runs'][0]['worst_streaks'], key=lambda x: x['length'])
    with np.load(path/'eval_long/steps_32.npz') as archive:
        held = np.delete(archive['proposal_log_weight'], stay['chain']-1, axis=0).ravel()
        current_w = stay['log_weight']
        alpha = np.exp(np.minimum(0, held-current_w))
        probability = float(alpha.mean())
        xs = np.linspace(np.quantile(held, .05), current_w, 80)
        ys = [float(np.exp(np.minimum(0, held-x)).mean()) for x in xs]
        chain_energy = archive['energy'][stay['chain']-1]
    mechanism = dict(seed=22, steps=32, held_out_proposal_count=len(held), stalled_state=stay,
        estimated_next_acceptance=probability,
        mc_sem=float(alpha.std(ddof=1)/np.sqrt(len(alpha))),
        estimated_mean_attempts_until_acceptance=1/probability,
        estimated_probability_of_at_least_observed_rejections=float(np.exp(stay['length']*np.log1p(-probability))),
        note='Finite proposal-bank estimate excluding the stuck chain. Geometric formulas assume the exact '
             'fixed-model independent proposal law. Heavy tails may make this estimate unstable. Not a mixing proof.')
    (args.out/'stall_mechanism.json').write_text(json.dumps(mechanism, indent=2)+'\n')
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(np.arange(1, len(chain_energy)+1), chain_energy, lw=.5)
    ax[0].axhline(protocol['reference_metadata']['summaries']['energy']['mean'], color='black', ls='--', label='Wolff mean')
    ax[0].axvspan(stay['first_retained_draw'], stay['last_retained_draw'], color='tab:red', alpha=.15,
                  label=f'{stay["length"]:,} consecutive rejections')
    ax[0].set(title='Seed 22, N=32, chain 5: prolonged stay', xlabel='Retained draw', ylabel='Energy per bond')
    ax[0].legend(fontsize=8)
    ax[1].plot(xs, np.asarray(ys)*100)
    ax[1].scatter([current_w], [probability*100], color='tab:red', zorder=4)
    ax[1].set_yscale('log')
    ax[1].set(title='Other seven chains estimate replacement probability', xlabel='Current path log weight', ylabel='Estimated next acceptance (%)')
    ax[1].annotate(f'Stuck state: {probability*100:.4f}%\nNo vortices in this state',
                   xy=(current_w, probability*100), xytext=(.1, .12), textcoords='axes fraction',
                   arrowprops=dict(arrowstyle='->'))
    for panel in ax:
        panel.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.out/'stall_mechanism.png', dpi=170)
    plt.close(fig)
    print(f'Saved {len(rows)} seed/step comparisons and held-out stall diagnosis to {args.out}')


if __name__ == '__main__':
    main()
