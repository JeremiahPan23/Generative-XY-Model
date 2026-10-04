"""CPU-only paired analysis of saved 5000/10000-update SPS experiments.

Requires independent saved-result audits from review_sps_run.py. Does not train
or generate any samples. Keeps all MH repeated states and all three seeds.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
JOBS = {11: 26736, 22: 26737, 33: 26738}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(seed, protocol):
    spec = protocol['baselines'][str(seed)]
    before = ROOT/spec['directory']
    after = ROOT/f'results/sps_a/L4_seed{seed}_continue10000_{JOBS[seed]}'
    old = read(before/spec['evaluation_directory']/'summary.json')
    new = read(after/'eval_long/summary.json')
    provenance = read(after/'continuation_provenance.json')
    complete = read(after/'completion.json')
    old_checkpoint = torch.load(before/'train/checkpoint.pt', map_location='cpu', weights_only=False)
    checkpoint = torch.load(after/'train/checkpoint.pt', map_location='cpu', weights_only=False)
    expected_config = dict(protocol['baseline_config'], seed=seed)
    checks = {
        'old_checkpoint': digest(before/'train/checkpoint.pt') == spec['checkpoint_sha256'],
        'old_summary': digest(before/spec['evaluation_directory']/'summary.json') == spec['summary_sha256'],
        'copied_old_summary': digest(after/'baseline_summary.json') == spec['summary_sha256'],
        'protocol': read(after/'continuation_protocol.json') == protocol,
        'protocol_hash': provenance['protocol_sha256'] == digest(ROOT/'configs/sps_a_continuation.json'),
        'runner_hash': provenance['runner_sha256'] == digest(ROOT/'scripts/sps_training_continuation.py'),
        'slurm_hash': provenance['slurm_script_sha256'] == digest(ROOT/'scripts/slurm/sps_a_continue.slurm'),
        'original_model_provenance': provenance['baseline_checkpoint_sha256'] == spec['checkpoint_sha256'],
        'old_config': old['config'] == expected_config == old_checkpoint['config'],
        'new_config': new['config'] == dict(expected_config, updates=10000) == checkpoint['config'],
        'finished_updates': old_checkpoint['update'] == 5000 and checkpoint['update'] == new['trained_update'] == 10000,
        'copied_history': checkpoint['history'][:len(old_checkpoint['history'])] == old_checkpoint['history'],
        'new_history_matches_log': checkpoint['history'] == read(after/'train/training_log.json'),
        'first_added_record': checkpoint['history'][len(old_checkpoint['history'])]['update'] == 5001,
        'optimizer_update_count': {int(s['step'].item()) for s in checkpoint['optimizer']['state'].values()} == {10000},
        'finite_model': all(bool(torch.isfinite(v).all()) for v in checkpoint['model'].values()),
        'model_changed': any(not torch.equal(v, old_checkpoint['model'][k]) for k, v in checkpoint['model'].items()),
        'same_evaluation': old['settings'] == new['settings'] == protocol['evaluation_settings'],
        'same_reference': old['reference'] == new['reference'] == protocol['reference_metadata'],
        'same_core_sources': old['sources'] == new['sources'] == checkpoint['sources'] == old_checkpoint['sources'],
        'new_checkpoint_hash': new['checkpoint_sha256'] == digest(after/'train/checkpoint.pt'),
        'completion': complete['software_complete'] and complete['trained_update'] == 10000 and complete['training_seed'] == seed,
        'completion_summary_hash': complete['evaluation_summary_sha256'] == digest(after/'eval_long/summary.json'),
        'cost_accounting': complete['baseline_training_seconds'] == old['training_seconds']
            and complete['total_training_seconds'] == new['training_seconds'] == checkpoint['training_seconds']
            and complete['added_training_seconds'] == new['training_seconds']-old['training_seconds'],
    }
    for name, expected in protocol['source_sha256'].items():
        checks[f'source:{name}'] = digest(ROOT/name) == expected
    for key in ('python', 'torch', 'numpy', 'gpu', 'device'):
        checks[f'environment:{key}'] = (provenance['environment'][key]
            == new['environment'][key] == checkpoint['environment'][key]
            == old_checkpoint['environment'][key] == protocol['baseline_environment'][key])
    checks['new_tests'] = 'Ran 11 tests' in (after/'logs'/f'slurm-xy-sps-A-continue-{JOBS[seed]}.err').read_text()
    checks['completion_log'] = f'CONTINUATION COMPLETE: seed={seed}; update=10000' in (
        after/'logs'/f'slurm-xy-sps-A-continue-{JOBS[seed]}.out').read_text()
    old_audit_path = ('analysis/eval_long_25955/audit.json' if seed == 11 else 'analysis/audit.json')
    old_audit = read(before/old_audit_path)
    new_audit = read(after/'analysis/audit/audit.json')
    checks['both_data_audits'] = old_audit['all_audit_checks_pass'] and new_audit['all_audit_checks_pass']
    if not all(checks.values()):
        raise ValueError(f'Seed {seed} provenance mismatch: {[k for k, v in checks.items() if not v]}')
    return old, new, old_audit, new_audit, checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT/'results/sps_a_training_budget')
    args = parser.parse_args()
    protocol = read(ROOT/'configs/sps_a_continuation.json')
    rows, checks_by_seed = [], {}
    for seed in JOBS:
        old, new, old_audit, new_audit, checks = verify(seed, protocol)
        checks_by_seed[str(seed)] = checks
        for before, after, d0, d1 in zip(old['runs'], new['runs'], old_audit['runs'], new_audit['runs']):
            if not before['steps'] == after['steps'] == d0['steps'] == d1['steps']:
                raise ValueError('Mismatched step settings')
            row = dict(seed=seed, steps=before['steps'])
            for tag, run, audit in [('before', before, d0), ('after', after, d1)]:
                e = run['summaries']['energy']
                ess = np.asarray(e['ess_by_chain'])
                row.update({f'{tag}_acceptance': run['accepted_fraction'],
                    f'{tag}_sampling_seconds': run['sampling_seconds'],
                    f'{tag}_energy_ess_per_second': run['ess_per_second']['energy'],
                    f'{tag}_energy_ess': e['ess'],
                    f'{tag}_energy': e['mean'], f'{tag}_energy_sem': e['sem'],
                    f'{tag}_screening_pass': run['screening_pass'],
                    f'{tag}_min_energy_chain_ess': float(ess.min()),
                    f'{tag}_pooled_mean_energy_ess_per_second': float(len(ess)**2/np.sum(1/ess)/run['sampling_seconds']),
                    f'{tag}_longest_rejections': max(audit['longest_rejection_streak_by_chain']),
                    f'{tag}_rejections_by_chain': audit['longest_rejection_streak_by_chain'],
                    f'{tag}_importance_ess_fraction': audit['importance_ess_fraction'],
                    f'{tag}_largest_normalized_importance_weight': audit['largest_normalized_importance_weight'],
                    f'{tag}_failed_observables': audit['failed_observable_checks'],
                    f'{tag}_worst_stay': max(audit['worst_streaks'], key=lambda x: x['length']),
                    f'{tag}_observables': run['summaries'],
                    f'{tag}_ess_per_second': run['ess_per_second'],
                })
            row['energy_ess_rate_ratio'] = row['after_energy_ess_per_second']/row['before_energy_ess_per_second']
            rows.append(row)
    args.out.mkdir(parents=True, exist_ok=True)
    limits = [
        'Three paired models, with common evaluation random seeds. Before/after estimates are correlated; no independence-based significance claim.',
        'Only one finite evaluation stream per checkpoint; a reduction of observed maxima does not bound unseen rare stalls.',
        'All repeated states retained. Screening and ESS/CI assume sufficient stationarity; failed pre-training cases remain provisional.',
        'ESS/s includes generation and MH, excludes training and diagnostics. Hardware timings were not replicated.',
        'Intermediate paths not saved: MH replay uses saved weights; complete path densities cannot be independently reconstructed.',
        'This is L=4 at one temperature, not a BKT critical-slowing-down scaling result or a matched-cost Wolff speedup.',
    ]
    payload = dict(all_provenance_checks_pass=True, checks_by_seed=checks_by_seed,
                   before_updates=5000, after_updates=10000, rows=rows, limitations=limits)
    (args.out/'comparison.json').write_text(json.dumps(payload, indent=2)+'\n', encoding='utf-8')
    columns = ['seed', 'steps', 'before_acceptance', 'after_acceptance',
               'before_energy_ess_per_second', 'after_energy_ess_per_second', 'energy_ess_rate_ratio',
               'before_longest_rejections', 'after_longest_rejections',
               'before_screening_pass', 'after_screening_pass',
               'before_energy', 'after_energy', 'before_energy_sem', 'after_energy_sem',
               'before_min_energy_chain_ess', 'after_min_energy_chain_ess',
               'before_pooled_mean_energy_ess_per_second', 'after_pooled_mean_energy_ess_per_second',
               'before_importance_ess_fraction', 'after_importance_ess_fraction']
    with (args.out/'comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({k: row[k] for k in columns} for row in rows)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.5))
    colors = {11: '#2673aa', 22: '#ce7419', 33: '#22854a'}
    for seed in JOBS:
        selected = [r for r in rows if r['seed'] == seed]
        for stage, style, marker in [('before', '--', 'o'), ('after', '-', 's')]:
            x = [r['steps'] for r in selected]
            color = colors[seed]
            axes[0, 0].plot(x, [100*r[f'{stage}_acceptance'] for r in selected],
                            style, marker=marker, color=color)
            axes[0, 1].plot(x, [r[f'{stage}_energy_ess_per_second'] for r in selected],
                            style, marker=marker, color=color)
            for row in selected:
                if not row[f'{stage}_screening_pass']:
                    for ax, metric, factor in [(axes[0, 0], 'acceptance', 100),
                                               (axes[0, 1], 'energy_ess_per_second', 1)]:
                        ax.scatter(row['steps'], factor*row[f'{stage}_{metric}'],
                                   color='crimson', marker='x', s=100, zorder=5)
    for ax, title, label in [(axes[0, 0], 'A. Acceptance: all nine paired settings increased', 'Accepted proposals (%)'),
                             (axes[0, 1], 'B. Estimated energy efficiency: all nine increased', 'Energy ESS / sampling second')]:
        ax.set(title=title, xlabel='Generation path steps N', ylabel=label)
        ax.set_xscale('log', base=2)
        ax.set_xticks([32, 64, 128], ['32', '64', '128'])
        ax.grid(alpha=.2)
    n32 = [r for r in rows if r['steps'] == 32]
    xs = np.arange(3)
    for stage, offset, color in [('before', -.18, '#9cabb9'), ('after', .18, '#22854a')]:
        y = [r[f'{stage}_longest_rejections'] for r in n32]
        bars = axes[1, 0].bar(xs+offset, y, width=.34, color=color,
                               label='5000 updates' if stage == 'before' else '10000 updates')
        axes[1, 0].bar_label(bars, labels=[f'{v:,}' for v in y], padding=3, fontsize=9)
        axes[1, 1].errorbar(xs+offset, [r[f'{stage}_energy'] for r in n32],
                            yerr=[1.96*r[f'{stage}_energy_sem'] for r in n32],
                            fmt='o' if stage == 'before' else 's', color=color, capsize=4)
    axes[1, 0].set(yscale='log', ylim=(100, 26000),
                   title='C. N=32: longest observed retained rejection streak',
                   ylabel='Consecutive rejected proposals')
    ref = protocol['reference_metadata']['summaries']['energy']
    axes[1, 1].axhline(ref['mean'], color='black', ls=':', label='Wolff reference')
    axes[1, 1].axhspan(ref['mean']-1.96*ref['sem'], ref['mean']+1.96*ref['sem'], color='black', alpha=.1)
    axes[1, 1].scatter([1-.18], [n32[1]['before_energy']], marker='x', color='crimson', s=100, zorder=5)
    axes[1, 1].set(title='D. N=32: energy estimates and approximate 95% intervals', ylabel='Energy per bond')
    for ax in axes[1]:
        ax.set_xticks(xs, ['Seed 11', 'Seed 22', 'Seed 33'])
        ax.grid(axis='y', alpha=.2)
    axes[1, 0].legend(fontsize=9)
    axes[1, 1].legend(fontsize=9)
    handles = [Line2D([0], [0], color=colors[s], lw=2, label=f'Seed {s}') for s in JOBS]
    handles += [Line2D([0], [0], color='gray', ls='--', marker='o', label='5000 updates'),
                Line2D([0], [0], color='gray', ls='-', marker='s', label='10000 updates'),
                Line2D([0], [0], color='crimson', ls='', marker='x', label='Failed preset screening')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .945), ncol=6, fontsize=9)
    fig.suptitle('Data-free XY SPS | L=4 | paired training-budget comparison', y=.985, fontsize=15)
    fig.text(.5, .02, 'One evaluation per checkpoint; repeated states retained. Failed-case ESS / intervals are provisional.\n'
             'Screening is not a mixing proof. ESS/s excludes training; rare stalls may still occur.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, .91))
    fig.savefig(args.out/'training_budget_comparison.png', dpi=180)
    fig.savefig(args.out/'training_budget_comparison.pdf')
    plt.close(fig)
    print('Paired provenance checks passed; saved nine setting comparisons and PNG/PDF.')
    for row in rows:
        print(f"seed={row['seed']} N={row['steps']}: acceptance {row['before_acceptance']:.4f}->{row['after_acceptance']:.4f}; "
              f"energy ESS/s {row['before_energy_ess_per_second']:.1f}->{row['after_energy_ess_per_second']:.1f}; "
              f"max stay {row['before_longest_rejections']}->{row['after_longest_rejections']}")


if __name__ == '__main__':
    main()
