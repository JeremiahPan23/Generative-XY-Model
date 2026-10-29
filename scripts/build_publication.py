"""Export selected public summaries, or verify them without the private archive.

Packaging never trains, samples, changes scientific code or rewrites raw data.
Existing historical artifacts stay byte-identical. New summaries use explicit
field allowlists; their manifest distinguishes projections from byte copies.
--check needs public files; --check-sources also needs the local archive.
"""
import argparse
import copy
import csv
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = {
    'results/training_budget.csv': 'results/sps_a_training_budget_three_points/comparison.csv',
    'results/training_budget.json': 'results/sps_a_training_budget_three_points/comparison.json',
    'results/generation_steps.csv': 'results/sps_a_seed_replication/comparison.csv',
    'results/generation_steps.json': 'results/sps_a_seed_replication/comparison.json',
    'results/sticking_mechanism.json': 'results/sps_sticking_mechanism/mechanism.json',
    'results/training_budget_import_manifest.json': 'results/sps_a_training_budget_three_points/import_manifest.json',
    'results/generation_steps_provenance.json': 'results/sps_a_seed_replication/provenance_checks.json',
    'results/ess_sensitivity.json': 'results/sps_a_seed_replication/ess_sensitivity.json',
    'results/legacy_fm_wolff.json': 'results/fm_wolff_baseline/summary.json',
    'results/legacy_fm_wolff_manifest.json': 'results/fm_wolff_baseline/manifest.json',
    'figures/training_budget.png': 'results/sps_a_training_budget_three_points/three_budget_comparison.png',
    'figures/training_budget.pdf': 'results/sps_a_training_budget_three_points/three_budget_comparison.pdf',
    'figures/generation_steps.png': 'results/sps_a_seed_replication/three_seed_comparison.png',
    'figures/sticking_mechanism.png': 'results/sps_sticking_mechanism/mechanism_explained.png',
    'figures/sticking_mechanism.pdf': 'results/sps_sticking_mechanism/mechanism_explained.pdf',
    'figures/legacy_fm_wolff.png': 'results/fm_wolff_baseline/comparison.png',
    'figures/legacy_fm_wolff.pdf': 'results/fm_wolff_baseline/comparison.pdf',
    'figures/reference_multitemperature.png': 'figures/multitemp_example_grid.png',
    'figures/reference_statistics.png': 'figures/multitemp_vortex_energy.png',
}


LEGACY_ARTIFACTS = ARTIFACTS.copy()
# Never glob the checkout: untracked local tooling is not public evidence.
IMPLEMENTATION_FILES = (
    'src/generate_wolff_sweep.py', 'src/model_dit.py', 'src/sampler_wolff.py',
    'src/sps_diagnostics.py', 'src/sps_xy.py', 'src/train_flow.py',
    'src/xy_observables.py', 'scripts/compare_fm_wolff.py',
    'scripts/compare_sps_seeds.py', 'scripts/compare_sps_training_budget.py',
    'scripts/compare_sps_training_budget_three_points.py',
    'scripts/explain_sps_sticking.py', 'scripts/plot_loss.py',
    'scripts/review_sps_run.py', 'scripts/sps_budget_15000.py',
    'scripts/sps_experiment.py', 'scripts/sps_seed_replication.py',
    'scripts/sps_training_continuation.py', 'scripts/verify_sps_a.py',
    'scripts/slurm/sps_a.slurm', 'scripts/slurm/sps_a_15000.slurm',
    'scripts/slurm/sps_a_continue.slurm', 'scripts/slurm/sps_a_evaluate.slurm',
    'scripts/slurm/sps_a_replicate.slurm', 'tests/test_sps_budget_15000.py',
    'tests/test_sps_continuation.py', 'tests/test_sps_replication.py',
    'tests/test_sps_xy.py', 'tests/test_xy_diagnostics.py',
    'configs/sps_a_15000.json', 'configs/sps_a_continuation.json',
    'configs/sps_a_l4.json', 'configs/sps_a_l8.json', 'configs/sps_a_replication.json',
    'src/sps_nelson.py', 'src/sps_intermediate.py',
    'src/sps_intermediate_efficiency.py', 'tests/test_sps_nelson.py',
    'tests/test_sps_intermediate_public.py', 'tests/test_sps_intermediate_efficiency.py',
    'tests/test_publication.py',
)
CSV_COLUMNS = {
    'path_steps_2000': (
        'seed', 'training_updates', 'L', 'beta', 'generation_steps', 'h',
        'candidate_path_count', 'retained_mh_count', 'acceptance',
        'proposal_importance_ess', 'proposal_importance_ess_fraction',
        'physical_screening_pass', 'failed_physical_observables',
        'energy_per_bond', 'energy_sem', 'vortex_pairs', 'vortex_pairs_sem',
        'auxiliary_mh_energy_ess', 'auxiliary_mh_pairs_ess', 'sampling_seconds',
        'summary_sha256', 'samples_sha256', 'checkpoint_sha256',
        'auxiliary_mh_G_1_ess', 'auxiliary_mh_G_1_ess_fraction',
        'auxiliary_mh_G_2_ess', 'auxiliary_mh_G_2_ess_fraction',
        'auxiliary_mh_energy_ess_fraction', 'auxiliary_mh_magnetization_ess',
        'auxiliary_mh_magnetization_ess_fraction', 'auxiliary_mh_pairs_ess_fraction',
        'proposal_importance_ess_denominator', 'auxiliary_mh_ess_denominator',
    ),
    'three_temperatures': (
        'temperature_tag', 'T', 'beta', 'L', 'J', 'training_seed',
        'training_updates', 'generation_steps', 'candidate_path_count',
        'retained_mh_count', 'acceptance', 'proposal_importance_ess',
        'proposal_importance_ess_fraction', 'auxiliary_importance_ess_per_generation_step',
        'energy_per_bond', 'energy_sem', 'vortex_pairs', 'vortex_pairs_sem',
        'auxiliary_mh_energy_ess', 'auxiliary_mh_pairs_ess',
        'auxiliary_mh_energy_ess_fraction', 'auxiliary_mh_pairs_ess_fraction',
        'sampling_seconds', 'training_seconds', 'physical_screening_pass',
        'failed_physical_observables', 'summary_sha256', 'samples_sha256',
        'checkpoint_sha256', 'reference_json_sha256', 'reference_samples_sha256',
    ),
    'shared_nelson': (
        'arm', 'training_seed', 'training_updates', 'temperature_tag', 'T',
        'beta', 't_stop', 'generation_steps', 'actual_dt', 'candidate_count',
        'retained_count', 'acceptance', 'importance_ess', 'importance_ess_fraction',
        'energy', 'energy_sem', 'pairs', 'pairs_sem',
        'auxiliary_energy_chain_ess_fraction', 'auxiliary_pairs_chain_ess_fraction',
        'sampling_seconds', 'physical_screening_pass', 'samples_sha256',
    ),
    'multitemperature_efficiency': (
        'arm', 'temperature', 'time', 'accepted_fraction', 'importance_ess_fraction',
    ),
    'mh_observables': (
        'arm', 'index', 'temperature', 'observable', 'mh', 'mh_low95',
        'mh_high95', 'mh_bootstrap_sem', 'raw', 'wolff', 'difference',
        'standardized_difference', 'retained_records',
    ),
}
NEW_ARTIFACTS = {
    'results/path_steps_2000.csv': ('results/sps_path_steps_2000/combined_formal/comparison.csv', 'path_steps_2000'),
    'figures/path_steps_2000.png': ('results/sps_path_steps_2000/combined_formal/path_steps_comparison.png', 'copy'),
    'results/three_temperatures.csv': ('results/sps_temperatures/combined_formal/comparison.csv', 'three_temperatures'),
    'figures/three_temperatures.png': ('results/sps_temperatures/combined_formal/temperatures_physics.png', 'copy'),
    'results/shared_nelson.csv': ('results/sps_nelson/combined_full_28346_28347/comparison.csv', 'shared_nelson'),
    'results/shared_nelson.json': ('results/sps_nelson/combined_full_28346_28347/comparison.json', 'shared_nelson_json'),
    'figures/shared_nelson_efficiency.png': ('results/sps_nelson/combined_full_28346_28347/comparison.png', 'copy'),
    'figures/shared_nelson_physics.png': ('results/sps_nelson/combined_full_28346_28347/physics.png', 'copy'),
    'results/multitemperature_efficiency.csv': ('results/sps_intermediate/efficiency_28965/comparison/curves.csv', 'multitemperature_efficiency'),
    'figures/multitemperature_efficiency.png': ('results/sps_intermediate/efficiency_28965/comparison/acceptance_ess.png', 'copy'),
    'results/mh_observables.csv': ('results/sps_intermediate/mh_observables_20261021/curves.csv', 'mh_observables'),
    'results/mh_observables.json': ('results/sps_intermediate/mh_observables_20261021/summary.json', 'mh_observables_json'),
    'figures/mh_observables.png': ('results/sps_intermediate/mh_observables_20261021/physics_mh_main.png', 'copy'),
    'figures/raw_mh_observables.png': ('results/sps_intermediate/mh_observables_20261021/physics_raw_mh_backup.png', 'copy'),
    'figures/mh_low_temperature.png': ('results/sps_intermediate/mh_observables_20261021/physics_mh_low_temperature.png', 'copy'),
}
ARTIFACTS = {**{key: (value, 'copy') for key, value in LEGACY_ARTIFACTS.items()}, **NEW_ARTIFACTS}
PRIMARY = ('energy_per_bond', 'vorticity_density', 'chi_radial')
DIAGNOSTIC_OBSERVABLES = (
    'energy_per_bond', 'vorticity_density', 'Mx', 'My', 'Mx2', 'My2',
    'M2', 'M_abs', 'm_abs', 'chi_radial_influence_proxy',
)
DIAGNOSTIC_FIELDS = (
    'mean', 'sd', 'sem', 'ci95', 'ess', 'ess_by_chain', 'split_rhat',
    'chain_means', 'first_half_mean', 'last_half_mean', 'screening_pass',
)
ESTIMATE_FIELDS = ('value', 'bootstrap_sem', 'ci95')
DIFFERENCE_FIELDS = ('difference', 'combined_bootstrap_sem', 'standardized_difference',
                     'within_three_combined_sem')
# Secondary fail-closed protection, not a substitute for field selection.
PRIVATE_PATH = re.compile(r'(?:[A-Za-z]:[\\/]|/(?:home|Users|mnt)/|file://|\\\\[^\\])')


def _inside(root, relative):
    parsed = PurePosixPath(relative)
    if (not relative or parsed.is_absolute() or '..' in parsed.parts
            or '\\' in relative or ':' in relative or parsed.as_posix() != relative):
        raise ValueError('Manifest paths must be canonical repository-relative paths')
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Manifest path escapes repository root')
    return path


def fingerprint_bytes(payload):
    return dict(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))


def fingerprint(path):
    return fingerprint_bytes(path.read_bytes())


def _pick(mapping, keys):
    return {key: copy.deepcopy(mapping[key]) for key in keys if key in mapping}


def _ensure_public_values(value):
    if isinstance(value, dict):
        for item in value.values():
            _ensure_public_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _ensure_public_values(item)
    elif isinstance(value, str) and PRIVATE_PATH.search(value):
        raise ValueError('Selected public field contains a private absolute path')


def project_csv(payload, columns):
    reader = csv.DictReader(io.StringIO(payload.decode('utf-8-sig'), newline=''))
    if (reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames)
            or not set(columns).issubset(reader.fieldnames)):
        raise ValueError('CSV schema is missing required columns or has duplicate columns')
    rows = []
    for row in reader:
        if None in row or any(value is None for value in row.values()):
            raise ValueError('CSV row does not match its header')
        selected = {key: row[key] for key in columns}
        _ensure_public_values(selected)
        rows.append(selected)
    output = io.StringIO(newline='')
    writer = csv.DictWriter(output, fieldnames=columns, lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode('utf-8'), {
        'kind': 'csv_projection', 'columns': list(columns),
        'dropped_csv_columns': [key for key in reader.fieldnames if key not in columns],
    }


def project_shared_nelson(data):
    result = _pick(data, ('status', 'mode', 'seed', 'replicated_training', 'protocol_sha256'))
    result['rows'] = [_pick(row, CSV_COLUMNS['shared_nelson']) for row in data['rows']]
    result['references'] = {}
    for tag in ('T092', 'Tcentral', 'T087'):
        reference = data['references'][tag]
        result['references'][tag] = {'T': reference['T'], 'summaries': {
            name: _pick(reference['summaries'][name], DIAGNOSTIC_FIELDS)
            for name in ('energy', 'pairs', 'magnetization', 'G_1', 'G_2')
            if name in reference['summaries']}}
    result['model_costs'] = {}
    for arm in ('free', 'nelson'):
        cost = data['model_costs'][arm]
        result['model_costs'][arm] = {
            'training_seconds': cost['training_seconds'],
            'parameter_counts': _pick(cost['parameter_counts'],
                                      ('total', 'trainable', 'forward', 'noise', 'backward'))}
    result['public_export_note'] = (
        'Explicit field projection of a saved analysis; not a new measurement. '
        'Both arms share training targets and paired noise streams. '
        'Only seed 11; these three temperatures participated in training.')
    _ensure_public_values(result)
    return result


def project_mh_observables(data):
    result = _pick(data, ('status', 'job', 'created_utc', 'no_new_candidates', 'no_training'))
    result['bootstrap'] = _pick(data['bootstrap'], (
        'method', 'draws', 'seed', 'primary_block_size', 'sensitivity_block_sizes',
        'shared_indices_across_temperatures', 'truncate_synthetic_series_only'))
    result['scope'] = {
        'L': 4, 'J': 1, 'training_seed': 11,
        'trained_temperatures': [0.92, 1 / 1.12, 0.87],
        'candidate_paths_per_arm': 4096, 'full_path_steps': 1035,
        'chains': 4, 'retained_draws_per_chain': 895,
        'retained_records_per_temperature': 3580,
        'scope_source': 'Fixed export protocol for the saved job-28965 records; '
                        'chain/frame counts are checked against the source summary.',
        'importance_ess_denominator': '4096 candidate paths including initialization and burn-in',
        'acceptance_denominator': '3580 retained attempted transitions',
        'retained_rejection_repeats_included': True,
        'primary_observables': list(PRIMARY),
        'energy_definition': 'H/(2*L^2)',
        'vorticity_definition': 'sum(abs(plaquette_charge))/L^2',
        'chi_radial_definition': 'beta/L^2*(mean(M_abs^2)-mean(M_abs)^2)',
        'comparison_note': 'Pointwise descriptive 3-SEM screen; not an equivalence test, '
                           'multiplicity-adjusted hypothesis test or mixing proof.',
        'uncertainty_note': 'Conditional on four finite chains; excludes training-seed '
                            'variation and cannot exclude unobserved long dependence.',
    }
    result['arms'] = {}
    for arm in ('free', 'nelson'):
        source = data['arms'][arm]
        if (len(source['frames']) != 13
                or [row['index'] for row in source['frames']] != list(range(13))):
            raise ValueError('Saved MH export requires all 13 ordered frames per arm')
        frames = []
        for frame in source['frames']:
            if (frame['chains'] != 4 or frame['retained_draws_per_chain'] != 895
                    or frame['retained_count'] != 3580):
                raise ValueError('Saved MH frame count differs from the fixed public scope')
            selected = _pick(frame, ('index', 'time', 'beta', 'temperature', 'retained_count',
                                     'chains', 'retained_draws_per_chain', 'acceptance_fraction'))
            for key in ('mh_estimates', 'raw_estimates', 'wolff_estimates'):
                selected[key] = {name: _pick(frame[key][name], ESTIMATE_FIELDS) for name in PRIMARY}
            selected['differences'] = {
                name: {comparison: _pick(frame['differences'][name][comparison], DIFFERENCE_FIELDS)
                       for comparison in ('mh_vs_wolff', 'raw_vs_wolff')}
                for name in PRIMARY}
            selected['per_chain_estimates'] = _pick(frame['per_chain_estimates'], PRIMARY)
            selected['between_chain_mean_sem_proxy'] = _pick(frame['between_chain_mean_sem_proxy'], PRIMARY)
            selected['chain_diagnostics'] = {
                name: _pick(frame['chain_diagnostics'][name], DIAGNOSTIC_FIELDS)
                for name in DIAGNOSTIC_OBSERVABLES if name in frame['chain_diagnostics']}
            selected['rejection_diagnostics'] = [_pick(row, (
                'accepted', 'records', 'longest_rejection_run',
                'terminal_rejection_run', 'rejection_runs', 'boundary_note'))
                for row in frame['rejection_diagnostics']]
            if (len(frame['vortex_events_by_chain']) != 4
                    or any(type(value) is not int or value < 0
                           for value in frame['vortex_events_by_chain'])):
                raise ValueError('Vortex-event diagnostics require four nonnegative integer counts')
            selected['vortex_events_by_chain'] = copy.deepcopy(frame['vortex_events_by_chain'])
            selected['warnings'] = copy.deepcopy(frame['warnings'])
            frames.append(selected)
        sensitivity = {}
        for block in ('32', '64', '128'):
            series = source['block_sensitivity'][block]
            if len(series) != len(frames):
                raise ValueError('Block-sensitivity frame count does not match estimates')
            sensitivity[block] = [{name: _pick(row[name], ESTIMATE_FIELDS) for name in PRIMARY}
                                  for row in series]
        result['arms'][arm] = {
            'frames': frames, 'block_sensitivity': sensitivity,
            **_pick(source, ('split_rhat_note', 'chi_chain_note'))}
    _ensure_public_values(result)
    return result


def _render(payload, transform):
    if transform == 'copy':
        return payload, {'kind': 'copy'}
    if transform in CSV_COLUMNS:
        return project_csv(payload, CSV_COLUMNS[transform])
    data = json.loads(payload.decode('utf-8-sig'))
    if transform == 'shared_nelson_json':
        public = project_shared_nelson(data)
    elif transform == 'mh_observables_json':
        public = project_mh_observables(data)
    else:
        raise ValueError('Unknown public export transform: ' + transform)
    return (json.dumps(public, indent=2, allow_nan=False) + '\n').encode('utf-8'), {
        'kind': 'json_projection', 'schema': transform + '_v1'}


def build(root):
    root = Path(root).resolve()
    missing = [name for name, _ in ARTIFACTS.values() if not _inside(root, name).is_file()]
    missing += [name for name in IMPLEMENTATION_FILES if not _inside(root, name).is_file()]
    if missing:
        raise FileNotFoundError('Rebuilding needs the local archive and selected implementation: '
                                + ', '.join(missing))
    manifest_path = root / 'publication/manifest.json'
    prior = json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else None
    if prior:
        # Do not silently rebaseline previously pinned scientific source bytes.
        for name, expected in prior['implementation_files'].items():
            if name not in IMPLEMENTATION_FILES or fingerprint(_inside(root, name)) != expected:
                raise ValueError('Previously pinned implementation changed: ' + name)
    artifacts, planned = {}, {}
    for target, (original, transform) in ARTIFACTS.items():
        source_payload = _inside(root, original).read_bytes()
        output, transformation = _render(source_payload, transform)
        destination = _inside(root, 'publication/' + target)
        if destination.exists() and destination.read_bytes() != output:
            raise ValueError('Refusing to overwrite a different public artifact: ' + target)
        source_hash = fingerprint_bytes(source_payload)
        artifacts[target] = dict(source=original, source_sha256=source_hash['sha256'],
                                 source_bytes=source_hash['bytes'], transformation=transformation,
                                 **fingerprint_bytes(output))
        planned[target] = output
    implementation = {name: fingerprint(_inside(root, name)) for name in IMPLEMENTATION_FILES}
    manifest = dict(format_version=2, artifacts=artifacts, implementation_files=implementation,
                    note='Historical artifacts and new figures are byte copies. New tabular/JSON '
                         'summaries are explicit field projections, not new measurements. '
                         'Source and output hashes are distinct. --check verifies public integrity; '
                         '--check-sources additionally verifies the omitted source archive and '
                         'deterministic exports. Neither re-audits all original trajectories.')
    # Validate all inputs and overwrite guards before writing any public file.
    for name, payload in planned.items():
        path = _inside(root, 'publication/' + name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(payload)
    manifest_path.write_bytes((json.dumps(manifest, indent=2) + '\n').encode('utf-8'))
    check(root)


def check(root, check_sources=False):
    root = Path(root).resolve()
    manifest = json.loads((root / 'publication/manifest.json').read_text(encoding='utf-8'))
    failures = []
    if manifest.get('format_version') == 2:
        if set(manifest['artifacts']) != set(ARTIFACTS):
            failures.append('artifact allowlist')
        if set(manifest['implementation_files']) != set(IMPLEMENTATION_FILES):
            failures.append('implementation allowlist')
    elif manifest.get('format_version') != 1:
        raise ValueError('Unsupported publication manifest format')
    for name, expected in manifest['artifacts'].items():
        path = _inside(root, 'publication/' + name)
        if not path.is_file() or fingerprint(path) != {key: expected[key] for key in ('sha256', 'bytes')}:
            failures.append('publication/' + name)
        if check_sources:
            source = _inside(root, expected['source'])
            if not source.is_file():
                raise FileNotFoundError('--check-sources needs the omitted local archive: ' + expected['source'])
            source_payload = source.read_bytes()
            source_hash = fingerprint_bytes(source_payload)
            if manifest['format_version'] == 1:
                if source_hash != {key: expected[key] for key in ('sha256', 'bytes')}:
                    failures.append('source:' + expected['source'])
                continue
            if source_hash != {'sha256': expected['source_sha256'], 'bytes': expected['source_bytes']}:
                failures.append('source:' + expected['source'])
            original, transform = ARTIFACTS[name]
            output, transformation = _render(source_payload, transform)
            if (original != expected['source'] or transformation != expected['transformation']
                    or fingerprint_bytes(output) != {key: expected[key] for key in ('sha256', 'bytes')}):
                failures.append('projection:' + name)
    for name, expected in manifest['implementation_files'].items():
        path = _inside(root, name)
        if not path.is_file() or fingerprint(path) != expected:
            failures.append(name)
    if failures:
        raise ValueError('Published integrity mismatch: ' + ', '.join(failures))
    print(f"PASS: {len(manifest['artifacts'])} public artifacts and "
          f"{len(manifest['implementation_files'])} implementation files match their manifest.")
    if check_sources:
        print('Source hashes and deterministic exports match; original trajectories were not re-audited.')
    else:
        print('Public integrity verification only; the omitted archive is not required or re-audited.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--check', action='store_true')
    action.add_argument('--check-sources', action='store_true')
    args = parser.parse_args()
    if args.check or args.check_sources:
        check(args.root, check_sources=args.check_sources)
    else:
        build(args.root.resolve())


if __name__ == '__main__':
    main()
