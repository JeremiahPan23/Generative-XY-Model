"""Standard-library checks of public projections and clean-clone integrity.

Synthetic fixture values test packaging, not physical validity or sampler output.
"""
import contextlib
import copy
import csv
import importlib.util
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('publication_builder', ROOT / 'scripts/build_publication.py')
pub = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pub)


def shared_fixture():
    return {
        'status': 'fixture', 'mode': 'full', 'seed': 11, 'replicated_training': False,
        'rows': [dict.fromkeys(pub.CSV_COLUMNS['shared_nelson'], 0)],
        'references': {tag: {'T': 1.0, 'summaries': {
            'energy': {'mean': -0.75, 'sem': 0.01, 'private_field': 'not public'}}}
                       for tag in ('T092', 'Tcentral', 'T087')},
        'model_costs': {arm: {'training_seconds': 1.0,
                             'parameter_counts': {'total': 1, 'private_field': 999}}
                        for arm in ('free', 'nelson')},
        'private_field': 'not public', 'notes': ['private meeting wording'],
    }


def mh_fixture():
    estimate = {'value': 1.25, 'bootstrap_sem': 0.1, 'ci95': [1.0, 1.5]}
    frame = {
        'index': 0, 'time': 0.0, 'beta': 0.0, 'temperature': None,
        'retained_count': 3580, 'chains': 4, 'retained_draws_per_chain': 895,
        'acceptance_fraction': 0.5,
        **{key: {name: dict(estimate, private_field='not public')
                 for name in (*pub.PRIMARY, 'private_observable')}
           for key in ('mh_estimates', 'raw_estimates', 'wolff_estimates')},
        'differences': {name: {comparison: {
            'difference': 0.0, 'combined_bootstrap_sem': 0.2,
            'standardized_difference': 0.0, 'within_three_combined_sem': True,
            'private_field': 'not public'}
            for comparison in ('mh_vs_wolff', 'raw_vs_wolff')}
            for name in pub.PRIMARY},
        'per_chain_estimates': {name: [1, 2, 3, 4] for name in pub.PRIMARY},
        'between_chain_mean_sem_proxy': {name: 0.2 for name in pub.PRIMARY},
        'chain_diagnostics': {name: {'ess': 50, 'ess_by_chain': [10, 12, 13, 15],
                                   'split_rhat': 1.15, 'screening_pass': False,
                                   'private_field': 'not public'}
                              for name in pub.DIAGNOSTIC_OBSERVABLES},
        'rejection_diagnostics': [{'accepted': 10, 'records': 895,
                                  'longest_rejection_run': 215,
                                  'terminal_rejection_run': 9,
                                  'rejection_runs': [1, 215, 9],
                                  'boundary_note': 'Finite retained window.',
                                  'private_field': 'not public'} for _ in range(4)],
        'vortex_events_by_chain': [1, 2, 3, 4], 'warnings': ['Long rejection dependence.'],
        'private_field': 'not public',
    }
    frames = [dict(copy.deepcopy(frame), index=i, time=i / 12,
                   temperature=None if i == 0 else 1.0) for i in range(13)]
    arm = {'frames': frames,
           'block_sensitivity': {block: [{name: copy.deepcopy(estimate)
                                         for name in pub.PRIMARY} for _ in frames]
                                 for block in ('32', '64', '128')},
           'split_rhat_note': 'Finite diagnostic.', 'chi_chain_note': 'Influence proxy.',
           'private_field': 'not public'}
    return {'status': 'fixture', 'job': 'fixture', 'no_training': True,
            'no_new_candidates': True,
            'bootstrap': {'draws': 512, 'primary_block_size': 64,
                          'sensitivity_block_sizes': [32, 64, 128],
                          'private_field': 'not public'},
            'arms': {name: copy.deepcopy(arm) for name in ('free', 'nelson')},
            'private_field': 'not public'}


def fixture_tree(root):
    for relative in pub.IMPLEMENTATION_FILES:
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b'# synthetic implementation fixture\n')
    for source, transform in pub.ARTIFACTS.values():
        target = root / source
        target.parent.mkdir(parents=True, exist_ok=True)
        if transform in pub.CSV_COLUMNS:
            stream = io.StringIO(newline='')
            writer = csv.DictWriter(stream, fieldnames=[*pub.CSV_COLUMNS[transform], 'summary_path'])
            writer.writeheader()
            writer.writerow({**dict.fromkeys(pub.CSV_COLUMNS[transform], '0.12500000000000001'),
                             'summary_path': r'C:\Users\private\summary.json'})
            payload = stream.getvalue().encode('utf-8')
        elif transform == 'shared_nelson_json':
            payload = json.dumps(shared_fixture()).encode('utf-8')
        elif transform == 'mh_observables_json':
            payload = json.dumps(mh_fixture()).encode('utf-8')
        else:
            payload = ('Synthetic source bytes: ' + source).encode('utf-8')
        target.write_bytes(payload)


class PublicationChecks(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def quiet_build(self):
        with contextlib.redirect_stdout(io.StringIO()):
            pub.build(self.root)

    def quiet_check(self, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            pub.check(self.root, **kwargs)

    def test_csv_projection_preserves_numeric_strings_and_drops_unknown_columns(self):
        payload = b'value,summary_path,private_new_field\r\n0.12500000000000001,C:\\Users\\secret,x\r\n'
        projected, transformation = pub.project_csv(payload, ('value',))
        self.assertEqual(projected, b'value\n0.12500000000000001\n')
        self.assertEqual(transformation['dropped_csv_columns'], ['summary_path', 'private_new_field'])
        self.assertNotIn(b'secret', projected)

    def test_json_projection_keeps_diagnostics_sensitivity_and_does_not_mutate_source(self):
        source = mh_fixture()
        before = copy.deepcopy(source)
        projected = pub.project_mh_observables(source)
        self.assertEqual(source, before)
        self.assertNotIn('private_field', json.dumps(projected))
        for arm in ('free', 'nelson'):
            data = projected['arms'][arm]
            self.assertEqual(len(data['frames']), 13)
            self.assertEqual(set(data['block_sensitivity']), {'32', '64', '128'})
            frame = data['frames'][2]
            self.assertFalse(frame['chain_diagnostics']['energy_per_bond']['screening_pass'])
            self.assertIn('chi_radial_influence_proxy', frame['chain_diagnostics'])
            self.assertEqual(frame['rejection_diagnostics'][0]['longest_rejection_run'], 215)
            self.assertEqual(frame['rejection_diagnostics'][0]['terminal_rejection_run'], 9)
            self.assertEqual(frame['warnings'], ['Long rejection dependence.'])
            self.assertEqual(data['block_sensitivity']['128'][2]['chi_radial']['ci95'], [1.0, 1.5])
        shared = shared_fixture()
        shared_before = copy.deepcopy(shared)
        public_shared = pub.project_shared_nelson(shared)
        self.assertEqual(shared, shared_before)
        self.assertNotIn('private', json.dumps(public_shared))

    def test_known_field_private_paths_and_wrong_scope_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'private absolute path'):
            pub.project_csv(b'value\n/home/private/file\n', ('value',))
        bad = mh_fixture()
        bad['arms']['free']['frames'][0]['retained_count'] = 3581
        with self.assertRaisesRegex(ValueError, 'fixed public scope'):
            pub.project_mh_observables(bad)
        bad = mh_fixture()
        bad['arms']['free']['frames'].pop()
        with self.assertRaisesRegex(ValueError, '13 ordered frames'):
            pub.project_mh_observables(bad)

    def test_build_uses_allowlist_and_public_check_needs_no_archive(self):
        fixture_tree(self.root)
        private = self.root / 'src/unpublished_local_tool.py'
        private.write_text('private local work\n', encoding='utf-8')
        self.quiet_build()
        manifest = json.loads((self.root / 'publication/manifest.json').read_text(encoding='utf-8'))
        self.assertNotIn(b'\r\n', (self.root / 'publication/manifest.json').read_bytes())
        self.assertEqual(len(manifest['artifacts']), 34)
        self.assertEqual(len(manifest['implementation_files']), 41)
        self.assertNotIn('src/unpublished_local_tool.py', manifest['implementation_files'])
        for target, source in pub.LEGACY_ARTIFACTS.items():
            self.assertEqual((self.root / 'publication' / target).read_bytes(),
                             (self.root / source).read_bytes())
        self.quiet_check(check_sources=True)
        shutil.rmtree(self.root / 'results')
        shutil.rmtree(self.root / 'figures')
        self.quiet_check()
        with self.assertRaisesRegex(FileNotFoundError, 'omitted local archive'):
            self.quiet_check(check_sources=True)

    def test_existing_public_artifact_or_pinned_implementation_cannot_be_rebased(self):
        fixture_tree(self.root)
        self.quiet_build()
        target = self.root / 'publication/figures/mh_observables.png'
        target.write_bytes(b'unexpected change')
        with self.assertRaisesRegex(ValueError, 'different public artifact'):
            self.quiet_build()
        target.write_bytes((self.root / pub.NEW_ARTIFACTS['figures/mh_observables.png'][0]).read_bytes())
        (self.root / pub.IMPLEMENTATION_FILES[0]).write_bytes(b'changed scientific implementation')
        with self.assertRaisesRegex(ValueError, 'Previously pinned implementation changed'):
            self.quiet_build()

    def test_missing_or_modified_public_files_and_changed_sources_are_detected(self):
        fixture_tree(self.root)
        self.quiet_build()
        target = self.root / 'publication/results/multitemperature_efficiency.csv'
        original = target.read_bytes()
        target.unlink()
        with self.assertRaisesRegex(ValueError, 'Published integrity mismatch'):
            self.quiet_check()
        target.write_bytes(original + b'changed')
        with self.assertRaisesRegex(ValueError, 'Published integrity mismatch'):
            self.quiet_check()
        target.write_bytes(original)
        source = self.root / pub.NEW_ARTIFACTS['results/path_steps_2000.csv'][0]
        source.write_bytes(source.read_bytes().replace(b'private', b'changed'))
        self.quiet_check()  # Public integrity is deliberately independent of local sources.
        with self.assertRaisesRegex(ValueError, 'source:'):
            self.quiet_check(check_sources=True)

    def test_unsafe_manifest_paths_and_incomplete_allowlists_are_rejected(self):
        fixture_tree(self.root)
        self.quiet_build()
        path = self.root / 'publication/manifest.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        manifest['artifacts'].pop('results/mh_observables.json')
        path.write_text(json.dumps(manifest), encoding='utf-8')
        with self.assertRaisesRegex(ValueError, 'artifact allowlist'):
            self.quiet_check()
        with self.assertRaisesRegex(ValueError, 'repository-relative'):
            pub._inside(self.root, '../outside')


if __name__ == '__main__':
    unittest.main()
