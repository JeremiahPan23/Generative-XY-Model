"""CPU-only verification of genuine continuation, not a physical experiment."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import tempfile
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import sps_experiment as experiment
from sps_xy import SPSConfig
from sps_training_continuation import digest, stage_checkpoint


class ContinuationChecks(unittest.TestCase):
    def test_continuation_matches_uninterrupted_training_and_preserves_original(self):
        with tempfile.TemporaryDirectory(prefix='sps-budget-check-') as directory:
            root = Path(directory)
            short_config = asdict(SPSConfig(L=3, hidden=4, train_steps=2, batch_size=3,
                                           updates=2, checkpoint_every=1, seed=22))
            long_config = dict(short_config, updates=4)
            def train(name, config, resume=False):
                config_path = root/(name+'.json')
                config_path.write_text(json.dumps(config), encoding='utf-8')
                experiment.train(argparse.Namespace(config=config_path, out=root/name,
                    device='cpu', updates=None, seed=None, resume=resume))
            train('short', short_config)
            original_path = root/'short/checkpoint.pt'
            original_hash = digest(original_path)
            (root/'continued').mkdir()
            config = stage_checkpoint(original_path, root/'continued/checkpoint.pt',
                                      short_config, experiment.sources(), 4)
            train('continued', config, resume=True)
            train('uninterrupted', long_config)
            continued = torch.load(root/'continued/checkpoint.pt', weights_only=False)
            full = torch.load(root/'uninterrupted/checkpoint.pt', weights_only=False)
            self.assertEqual(digest(original_path), original_hash)
            self.assertEqual(continued['update'], 4)
            self.assertEqual([x['update'] for x in continued['history']], [1, 2, 3, 4])
            def assert_equal(a, b):
                if isinstance(a, torch.Tensor):
                    torch.testing.assert_close(a, b, rtol=0, atol=0)
                elif isinstance(a, dict):
                    self.assertEqual(a.keys(), b.keys())
                    for key in a:
                        assert_equal(a[key], b[key])
                elif isinstance(a, (list, tuple)):
                    self.assertEqual(len(a), len(b))
                    for x, y in zip(a, b):
                        assert_equal(x, y)
                else:
                    self.assertEqual(a, b)
            for key in ('model', 'optimizer', 'rng_cpu', 'rng_cuda', 'config', 'sources'):
                assert_equal(continued[key], full[key])
            for a, b in zip(continued['history'], full['history']):
                assert_equal({k: v for k, v in a.items() if k != 'seconds'},
                             {k: v for k, v in b.items() if k != 'seconds'})
            with self.assertRaisesRegex(ValueError, 'already exists'):
                stage_checkpoint(original_path, root/'continued/checkpoint.pt',
                                 short_config, experiment.sources(), 4)
            with self.assertRaisesRegex(ValueError, 'exceed'):
                stage_checkpoint(original_path, root/'invalid.pt', short_config,
                                 experiment.sources(), 2)
            with self.assertRaisesRegex(ValueError, 'contents'):
                stage_checkpoint(original_path, root/'wrong.pt', dict(short_config, seed=33),
                                 experiment.sources(), 4)


if __name__ == '__main__':
    unittest.main()
