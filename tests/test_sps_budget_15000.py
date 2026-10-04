"""Standard-library guards for the final budget point; does not train."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from sps_budget_15000 import preflight_15000
from sps_training_continuation import digest


class FinalBudgetGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.baseline = self.root/'baseline'
        for name in ('train', 'reference', 'eval_long'):
            (self.baseline/name).mkdir(parents=True, exist_ok=True)
        source = self.root/'model.py'
        source.write_text('unchanged source')
        (self.baseline/'train/checkpoint.pt').write_bytes(b'original 10000-update model')
        (self.baseline/'reference/samples.npz').write_bytes(b'validation only')
        ref = dict(L=4, beta=1.12, samples_sha256=digest(self.baseline/'reference/samples.npz'))
        source_hashes = {'model.py': digest(source)}
        config = dict(seed=11, updates=10000)
        summary = dict(config=config, settings={'chains': 8}, trained_update=10000,
                       checkpoint_sha256=digest(self.baseline/'train/checkpoint.pt'),
                       sources=source_hashes, reference=ref)
        self.write('reference/reference.json', ref)
        self.write('eval_long/summary.json', summary)
        self.completion = dict(software_complete=True, training_seed=11, trained_update=10000,
                              evaluation_summary_sha256=digest(self.baseline/'eval_long/summary.json'))
        self.write('completion.json', self.completion)
        self.protocol = dict(start_update=10000, target_updates=15000, training_seeds=[11, 22, 33],
            baseline_config=config, evaluation_settings=summary['settings'], reference_metadata=ref,
            source_sha256=source_hashes,
            baselines={'11':dict(directory='baseline', evaluation_directory='eval_long',
                checkpoint_sha256=summary['checkpoint_sha256'],
                summary_sha256=digest(self.baseline/'eval_long/summary.json'))})

    def write(self, name, value):
        (self.baseline/name).write_text(json.dumps(value), encoding='utf-8')

    def test_completed_10000_checkpoint_is_accepted(self):
        path, config, sources = preflight_15000(self.root, self.protocol, 11)
        self.assertEqual(path, self.baseline)
        self.assertEqual(config['updates'], 10000)

    def test_larger_budget_cannot_be_requested(self):
        altered = copy.deepcopy(self.protocol)
        altered['target_updates'] = 20000
        with self.assertRaisesRegex(ValueError, '10000 -> 15000'):
            preflight_15000(self.root, altered, 11)

    def test_wrong_checkpoint_is_rejected_with_correct_phase_message(self):
        (self.baseline/'train/checkpoint.pt').write_bytes(b'wrong model')
        with self.assertRaisesRegex(ValueError, '10000-update checkpoint'):
            preflight_15000(self.root, self.protocol, 11)

    def test_incomplete_baseline_is_rejected(self):
        self.write('completion.json', dict(self.completion, software_complete=False))
        with self.assertRaisesRegex(ValueError, 'completed run'):
            preflight_15000(self.root, self.protocol, 11)

    def test_changed_core_source_is_rejected(self):
        (self.root/'model.py').write_text('changed implementation')
        with self.assertRaisesRegex(ValueError, 'source/configuration changed'):
            preflight_15000(self.root, self.protocol, 11)

    def test_changed_reference_is_rejected(self):
        (self.baseline/'reference/samples.npz').write_bytes(b'wrong reference')
        with self.assertRaisesRegex(ValueError, 'reference changed'):
            preflight_15000(self.root, self.protocol, 11)


if __name__ == '__main__':
    unittest.main()
