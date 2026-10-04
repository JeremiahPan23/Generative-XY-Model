"""Small CPU-only checks for replication provenance guards; no training."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
from sps_seed_replication import digest, preflight


class ReplicationGuards(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.baseline = self.root/'baseline'
        for path in ('src', 'baseline/train', 'baseline/reference', 'baseline/eval_long_25955'):
            (self.root/path).mkdir(parents=True, exist_ok=True)
        (self.root/'src/model.py').write_text('unchanged source')
        (self.baseline/'train/checkpoint.pt').write_bytes(b'original checkpoint')
        (self.baseline/'reference/samples.npz').write_bytes(b'original reference')
        ref = dict(L=4, beta=1.12, samples_sha256=digest(self.baseline/'reference/samples.npz'))
        source = {'src/model.py': digest(self.root/'src/model.py')}
        self.protocol = dict(new_training_seeds=[22, 33], source_sha256=source,
            baseline_checkpoint_sha256=digest(self.baseline/'train/checkpoint.pt'),
            reference_metadata=ref, baseline_config={'seed': 11, 'updates': 5000},
            evaluation_settings={'chains': 8, 'draws': 16384})
        prior = dict(config=self.protocol['baseline_config'], settings=self.protocol['evaluation_settings'],
                     checkpoint_sha256=self.protocol['baseline_checkpoint_sha256'], sources=source, reference=ref)
        (self.baseline/'reference/reference.json').write_text(json.dumps(ref))
        (self.baseline/'eval_long_25955/summary.json').write_text(json.dumps(prior))

    def test_both_preselected_seeds_pass(self):
        for seed in (22, 33):
            self.assertEqual(preflight(self.root, self.baseline, self.protocol, seed), self.protocol['reference_metadata'])

    def test_accidental_baseline_retraining_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'preselected'):
            preflight(self.root, self.baseline, self.protocol, 11)

    def test_changed_source_is_rejected(self):
        (self.root/'src/model.py').write_text('changed source')
        with self.assertRaisesRegex(ValueError, 'source/configuration mismatch'):
            preflight(self.root, self.baseline, self.protocol, 22)

    def test_wrong_checkpoint_is_rejected(self):
        (self.baseline/'train/checkpoint.pt').write_bytes(b'wrong model')
        with self.assertRaisesRegex(ValueError, 'Wrong baseline checkpoint'):
            preflight(self.root, self.baseline, self.protocol, 22)

    def test_changed_reference_samples_are_rejected(self):
        (self.baseline/'reference/samples.npz').write_bytes(b'wrong reference')
        with self.assertRaisesRegex(ValueError, 'reference samples changed'):
            preflight(self.root, self.baseline, self.protocol, 22)

    def test_different_long_chain_budget_is_rejected(self):
        protocol = copy.deepcopy(self.protocol)
        protocol['evaluation_settings']['draws'] = 2048
        with self.assertRaisesRegex(ValueError, 'long evaluation does not match'):
            preflight(self.root, self.baseline, protocol, 22)


if __name__ == '__main__':
    unittest.main()
