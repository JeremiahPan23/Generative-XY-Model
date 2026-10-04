"""CPU-only saved-data comparison of three paired training budgets.

Keeps the existing two-budget analysis intact. Requires per-run independent MH
and observable audits. Does not train, sample, or remove repeated states.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from compare_sps_training_budget import ROOT, read, digest, verify as verify_first_pair
from sps_budget_15000 import preflight_15000

FINAL_JOBS = {11: 26746, 22: 26747, 33: 26748}


def verify_final(seed, protocol):
    before, config, sources = preflight_15000(ROOT, protocol, seed)
    after = ROOT/f'results/sps_a/L4_seed{seed}_continue15000_{FINAL_JOBS[seed]}'
    old = read(before/'eval_long/summary.json')
    new = read(after/'eval_long/summary.json')
    spec = protocol['baselines'][str(seed)]
    provenance = read(after/'continuation_provenance.json')
    complete = read(after/'completion.json')
    checkpoint = torch.load(after/'train/checkpoint.pt', map_location='cpu', weights_only=False)
    prior = torch.load(before/'train/checkpoint.pt', map_location='cpu', weights_only=False)
    audit = read(after/'analysis/audit/audit.json')
    stderr = (after/'logs'/f'slurm-xy-sps-A-15000-{FINAL_JOBS[seed]}.err').read_text()
    checks = {
        '10000_preflight': True,
        'copied_baseline_summary': digest(after/'baseline_summary.json') == spec['summary_sha256'],
        'copied_protocol': read(after/'continuation_protocol.json') == protocol,
        'protocol_hash': provenance['protocol_sha256'] == digest(ROOT/'configs/sps_a_15000.json'),
        'runner_hash': provenance['runner_sha256'] == digest(ROOT/'scripts/sps_budget_15000.py'),
        'helper_hash': provenance['staging_helper_sha256'] == digest(ROOT/'scripts/sps_training_continuation.py'),
        'slurm_hash': provenance['slurm_script_sha256'] == digest(ROOT/'scripts/slurm/sps_a_15000.slurm'),
        'baseline_model_hash': provenance['baseline_checkpoint_sha256'] == spec['checkpoint_sha256'],
        'phase': provenance['start_update'] == 10000 and provenance['target_update'] == 15000,
        'expected_config': new['config'] == dict(config, updates=15000) == checkpoint['config'],
        'finished_updates': checkpoint['update'] == new['trained_update'] == 15000,
        'preserved_history': checkpoint['history'][:len(prior['history'])] == prior['history'],
        'first_added_record': checkpoint['history'][len(prior['history'])]['update'] == 10001,
        'history_matches_json': checkpoint['history'] == read(after/'train/training_log.json'),
        'adam_step_count': {int(s['step'].item()) for s in checkpoint['optimizer']['state'].values()} == {15000},
        'finite_model': all(bool(torch.isfinite(v).all()) for v in checkpoint['model'].values()),
        'changed_model': any(not torch.equal(v, prior['model'][k]) for k, v in checkpoint['model'].items()),
        'same_evaluation': old['settings'] == new['settings'] == protocol['evaluation_settings'],
        'same_reference': old['reference'] == new['reference'] == protocol['reference_metadata'],
        'same_core_sources': old['sources'] == new['sources'] == checkpoint['sources'] == sources,
        'new_checkpoint_hash': new['checkpoint_sha256'] == digest(after/'train/checkpoint.pt'),
        'completion': complete['software_complete'] and complete['trained_update'] == 15000
            and complete['training_seed'] == seed and complete['final_training_budget_point'],
        'completion_summary_hash': complete['evaluation_summary_sha256'] == digest(after/'eval_long/summary.json'),
        'cost_accounting': complete['baseline_training_seconds'] == old['training_seconds']
            and complete['total_training_seconds'] == new['training_seconds'] == checkpoint['training_seconds']
            and complete['added_training_seconds'] == new['training_seconds']-old['training_seconds'],
        'three_test_suites': all(f'Ran {count} test' in stderr for count in (11, 1, 6))
            and stderr.count('\nOK\n') == 3,
        'completion_log': f'CONTINUATION COMPLETE: seed={seed}; update=15000' in (
            after/'logs'/f'slurm-xy-sps-A-15000-{FINAL_JOBS[seed]}.out').read_text(),
        'raw_data_audit': audit['all_audit_checks_pass'],
    }
    for key in ('python', 'torch', 'numpy', 'gpu', 'device'):
        checks[f'environment:{key}'] = (provenance['environment'][key] == new['environment'][key]
            == checkpoint['environment'][key] == prior['environment'][key] == protocol['baseline_environment'][key])
    if not all(checks.values()):
        raise ValueError(f'Seed {seed} final-point mismatch: {[k for k, v in checks.items() if not v]}')
    return new, audit, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/sps_a_training_budget_three_points')
    args = parser.parse_args()
    first_protocol = read(ROOT/'configs/sps_a_continuation.json')
    final_protocol = read(ROOT/'configs/sps_a_15000.json')
    rows, checks = [], {}
    for seed in FINAL_JOBS:
        first, second, a0, a1, c0 = verify_first_pair(seed, first_protocol)
        third, a2, c1 = verify_final(seed, final_protocol)
        checks[str(seed)] = {'5000_to_10000': c0, '10000_to_15000': c1}
        for budget, summary, audit in [(5000, first, a0), (10000, second, a1), (15000, third, a2)]:
            for run, detail in zip(summary['runs'], audit['runs']):
                if run['steps'] != detail['steps']:
                    raise ValueError('Mismatched generation steps')
                energy = run['summaries']['energy']
                ess = np.asarray(energy['ess_by_chain'])
                rows.append(dict(seed=seed, updates=budget, steps=run['steps'],
                    acceptance=run['accepted_fraction'], energy_ess_per_second=run['ess_per_second']['energy'],
                    min_energy_chain_ess=float(ess.min()),
                    pooled_mean_energy_ess_per_second=float(len(ess)**2/np.sum(1/ess)/run['sampling_seconds']),
                    longest_rejections=max(detail['longest_rejection_streak_by_chain']),
                    rejections_by_chain=detail['longest_rejection_streak_by_chain'],
                    screening_pass=run['screening_pass'], energy=energy['mean'], energy_sem=energy['sem'],
                    sampling_seconds=run['sampling_seconds'], training_seconds=summary['training_seconds'],
                    importance_ess_fraction=detail['importance_ess_fraction'],
                    largest_normalized_importance_weight=detail['largest_normalized_importance_weight'],
                    failed_observables=detail['failed_observable_checks'],
                    worst_stay=max(detail['worst_streaks'], key=lambda x: x['length']),
                    observables=run['summaries'], ess_per_second=run['ess_per_second']))
    transitions = []
    for seed in FINAL_JOBS:
        for steps in (32, 64, 128):
            for b0, b1 in [(5000, 10000), (10000, 15000)]:
                before = next(r for r in rows if (r['seed'], r['steps'], r['updates']) == (seed, steps, b0))
                after = next(r for r in rows if (r['seed'], r['steps'], r['updates']) == (seed, steps, b1))
                transitions.append(dict(seed=seed, steps=steps, from_updates=b0, to_updates=b1,
                    acceptance_change=after['acceptance']-before['acceptance'],
                    energy_ess_rate_change=after['energy_ess_per_second']-before['energy_ess_per_second'],
                    energy_ess_rate_ratio=after['energy_ess_per_second']/before['energy_ess_per_second'],
                    longest_rejection_change=after['longest_rejections']-before['longest_rejections']))
    limits = [
        'Third point selected after seeing 10000-update results; not a three-point preregistered experiment.',
        'Three paired training histories with common evaluation RNG seeds; no cross-model pooling or independence-based significance claims.',
        'One finite evaluation stream and timing per checkpoint. Observed extrema and ESS estimates can fluctuate; no proof of training convergence or tail control.',
        'Failed 5000-update screening makes corresponding ESS/CI provisional. All retained repeated states included.',
        'ESS/s excludes training; total and added training costs must be reported separately.',
        'MH replay uses saved weights; intermediate paths are unavailable for independent full path-density recomputation.',
        'Only L=4 and one temperature; no cost-matched Wolff speedup or BKT critical-slowing-down scaling claim.',
    ]
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'comparison.json').write_text(json.dumps(dict(all_provenance_checks_pass=True,
        checks_by_seed=checks, budgets=[5000, 10000, 15000], rows=rows, transitions=transitions,
        limitations=limits), indent=2)+'\n', encoding='utf-8')
    columns = ['seed', 'updates', 'steps', 'acceptance', 'energy_ess_per_second',
               'min_energy_chain_ess', 'pooled_mean_energy_ess_per_second', 'longest_rejections',
               'screening_pass', 'energy', 'energy_sem', 'sampling_seconds', 'training_seconds',
               'importance_ess_fraction', 'largest_normalized_importance_weight']
    with (args.out/'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: row[k] for k in columns} for row in rows)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    colors = {11: '#2673aa', 22: '#ce7419', 33: '#22854a'}
    for index, seed in enumerate(FINAL_JOBS):
        selected = [r for r in rows if r['seed'] == seed and r['steps'] == 32]
        x = [r['updates']/1000 for r in selected]
        for ax, field, factor in [(axes[0, 0], 'acceptance', 100),
                                  (axes[0, 1], 'energy_ess_per_second', 1),
                                  (axes[1, 0], 'longest_rejections', 1)]:
            y = [factor*r[field] for r in selected]
            ax.plot(x, y, 'o-', color=colors[seed], label=f'Seed {seed}')
            if field == 'longest_rejections':
                for a, b in zip(x, y):
                    ax.annotate(f'{b:,}', (a, b), textcoords='offset points', xytext=(0, 7),
                                ha='center', fontsize=8, color=colors[seed])
            for row in selected:
                if not row['screening_pass']:
                    ax.scatter(row['updates']/1000, factor*row[field], marker='x', color='crimson', s=100, zorder=5)
        axes[1, 1].errorbar(np.asarray(x)+(index-1)*.2, [r['energy'] for r in selected],
                            yerr=[1.96*r['energy_sem'] for r in selected], fmt='o-',
                            color=colors[seed], capsize=3)
    failed = next(r for r in rows if r['seed'] == 22 and r['steps'] == 32 and r['updates'] == 5000)
    axes[1, 1].scatter(5, failed['energy'], color='crimson', marker='x', s=100, zorder=5)
    ref = final_protocol['reference_metadata']['summaries']['energy']
    axes[1, 1].axhline(ref['mean'], ls=':', color='black')
    axes[1, 1].axhspan(ref['mean']-1.96*ref['sem'], ref['mean']+1.96*ref['sem'], color='black', alpha=.1)
    for ax, title, ylabel in [(axes[0, 0], 'A. Acceptance is not monotonic in training budget', 'Accepted proposals (%)'),
                              (axes[0, 1], 'B. Seeds 11/22 improve; seed 33 drops after 10k', 'Estimated energy ESS / sampling second'),
                              (axes[1, 0], 'C. Longer training does not remove long stays', 'Longest retained rejection streak'),
                              (axes[1, 1], 'D. Energy estimates and approximate 95% intervals', 'Energy per bond')]:
        ax.set(title=title, xlabel='Parameter updates (thousands)', ylabel=ylabel)
        ax.set_xticks([5, 10, 15], ['5k', '10k', '15k'])
        ax.grid(alpha=.2)
    axes[1, 0].set(yscale='log', ylim=(150, 23000))
    legend = [Line2D([0], [0], color=colors[s], marker='o', label=f'Seed {s}') for s in FINAL_JOBS]
    legend += [Line2D([0], [0], color='crimson', marker='x', ls='', label='Failed preset screening'),
               Line2D([0], [0], color='black', ls=':', label='Wolff energy reference')]
    fig.legend(handles=legend, loc='upper center', bbox_to_anchor=(.5, .945), ncol=5, fontsize=9)
    fig.suptitle('Data-free XY SPS | L=4, N=32 | three paired training budgets', y=.985, fontsize=15)
    fig.text(.5, .02, 'N=32 is the same illustrative subset used previously; all 27 settings are saved in CSV/JSON.\n'
             'Failed-case ESS / intervals are provisional. ESS/s excludes training; no convergence or mixing proof.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .91))
    fig.savefig(args.out/'three_budget_comparison.png', dpi=180)
    fig.savefig(args.out/'three_budget_comparison.pdf')
    plt.close(fig)
    print('All provenance checks passed; saved 27 settings and 18 paired transitions.')
    for seed in FINAL_JOBS:
        for steps in (32, 64, 128):
            selected = [r for r in rows if r['seed'] == seed and r['steps'] == steps]
            print(seed, steps, 'ESS/s', [round(r['energy_ess_per_second'], 1) for r in selected],
                  'max stay', [r['longest_rejections'] for r in selected])


if __name__ == '__main__':
    main()
