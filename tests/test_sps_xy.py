"""CPU-only physical/probabilistic checks for the XY SPS adaptation."""
from pathlib import Path
import math
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import i0, i1, logsumexp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from sps_xy import SPSConfig, XYPathSampler, energy, wrapped_log_prob, independence_mh
from sps_diagnostics import chain_summary, measurements

torch.set_num_threads(2)


class SPSChecks(unittest.TestCase):
    def test_wrapped_density_normalization_periodicity_and_fourier_moment(self):
        n = 32768
        x = (torch.arange(n, dtype=torch.float64)+.5)*(2*math.pi/n)-math.pi
        for sigma in (.07, .3, 1.0, 2.0):
            p = wrapped_log_prob(x, torch.tensor(2.9), sigma).exp()
            self.assertAlmostEqual(float(p.mean()*2*math.pi), 1, places=10)
            moment = (p*torch.exp(1j*x)).mean()*2*math.pi
            expected = np.exp(1j*float(torch.tensor(2.9))-.5*sigma*sigma)
            self.assertLess(abs(complex(moment)-expected), 1e-10)
            torch.testing.assert_close(wrapped_log_prob(x, x*.2, sigma),
                                       wrapped_log_prob(x+4*math.pi, x*.2-2*math.pi, sigma))

    def test_wrapped_density_against_wide_image_sum_near_cut(self):
        x = np.array([-math.pi+1e-7, math.pi-1e-7, .2])
        for sigma in (.1, 1, 2):
            expected = logsumexp(-.5*((x[:, None]+2*math.pi*np.arange(-30, 31))/sigma)**2,
                                axis=-1)-math.log(sigma*math.sqrt(2*math.pi))
            actual = wrapped_log_prob(torch.tensor(x), torch.zeros(3), sigma).numpy()
            np.testing.assert_allclose(actual, expected, atol=1e-11)
        with self.assertRaises(ValueError):
            wrapped_log_prob(torch.tensor(0.), torch.tensor(0.), 2.01)

    def test_total_energy_against_explicit_bond_loop(self):
        x = np.random.default_rng(4).uniform(-np.pi, np.pi, (4, 4))
        expected = sum(-np.cos(x[i,j]-x[(i+1)%4,j])-np.cos(x[i,j]-x[i,(j+1)%4])
                       for i in range(4) for j in range(4))
        self.assertAlmostEqual(float(energy(torch.tensor(x))), expected, places=12)

    def test_zero_drift_uniform_target_has_constant_path_weight(self):
        model = XYPathSampler(SPSConfig(beta=0, hidden=4)).double()
        with torch.no_grad():
            for n in (1, 4, 16):
                s = model.sample(16, n)
                torch.testing.assert_close(s['log_weight'], torch.full((16,), 16*math.log(2*math.pi), dtype=torch.float64), rtol=0, atol=2e-11)

    def test_path_weight_against_independent_scalar_gaussian_sum(self):
        model = XYPathSampler(SPSConfig(L=3, hidden=4)).double()
        with torch.no_grad():
            model.forward_drift.net[-1].bias.fill_(.2)
            model.backward_drift.net[-1].bias.fill_(-.15)
            sample = model.sample(2, 3, keep_path=True)
        paths = sample['path'].numpy()
        h = 1/3
        f = 12*np.tanh(.2/12)
        b = 12*np.tanh(-.15/12)
        def kernel(y, mu):
            d = (y-mu+np.pi)%(2*np.pi)-np.pi
            return logsumexp(-.5*((d[..., None]+2*np.pi*np.arange(-20, 21))/np.sqrt(h))**2,
                             axis=-1)-.5*np.log(2*np.pi*h)
        qf = np.full(2, -9*np.log(2*np.pi))
        qb = np.zeros(2)
        for j in range(3):
            qf += kernel(paths[:, j+1], paths[:, j]+h*f).sum((-2, -1))
            qb += kernel(paths[:, j], paths[:, j+1]+h*b).sum((-2, -1))
        expected = -1.12*sample['energy'].numpy()+qb-qf
        np.testing.assert_allclose(sample['log_weight'].numpy(), expected, atol=1e-10)

    def test_loss_has_finite_pathwise_gradients(self):
        torch.manual_seed(44)
        model = XYPathSampler(SPSConfig(L=3, hidden=4)).double()
        s = model.sample(8, 3)
        loss = -s['log_weight'].mean()/9
        loss.backward()
        for p in model.parameters():
            self.assertIsNotNone(p.grad)
            self.assertTrue(torch.isfinite(p.grad).all())
        self.assertGreater(model.forward_drift.net[-1].weight.grad.abs().sum().item(), 0)

    def test_mh_rejection_preserves_old_weight_and_repeated_state(self):
        theta = np.arange(4).reshape(1, 4, 1, 1)
        weights = np.array([[0., -100., -1., 2.]])
        chain = independence_mh(theta, weights, np.array([[-1., -.5, -.1]]), burn=0)
        np.testing.assert_array_equal(chain['accepted'], [[False, False, True]])
        np.testing.assert_array_equal(chain['theta'].ravel(), [0, 0, 3])
        np.testing.assert_array_equal(chain['log_weight'].ravel(), [0, 0, 2])

    def test_mh_corrects_uniform_proposals_to_known_von_mises_target(self):
        rng = np.random.default_rng(911)
        theta = rng.uniform(-np.pi, np.pi, (4, 14001, 1, 1))
        weight = 1.2*np.cos(theta[..., 0, 0])
        result = independence_mh(theta, weight, np.log(rng.random((4, 14000))), burn=2000)
        corrected = result['theta'][..., 0, 0]
        self.assertLess(abs(np.cos(corrected).mean()-i1(1.2)/i0(1.2)), .015)
        self.assertLess(abs(np.sin(corrected).mean()), .015)
        self.assertLess(abs(np.cos(theta).mean()), .015)
        self.assertTrue((~result['accepted']).any())

    def test_ess_detects_sticking_and_constant_chains(self):
        x = np.random.default_rng(23).normal(size=(4, 512))
        independent = chain_summary(x)
        repeated = chain_summary(np.repeat(x, 8, axis=1))
        self.assertLess(repeated['ess']/4096, independent['ess']/512/5)
        self.assertIsNone(chain_summary(np.zeros((4, 512)))['ess'])
        self.assertFalse(chain_summary(np.zeros((4, 512)))['screening_pass'])

    def test_high_temperature_xy_path_mh_against_wolff(self):
        from numba import njit
        from sampler_wolff import wolff_step_xy
        torch.manual_seed(710)
        model = XYPathSampler(SPSConfig(L=3, beta=.2, hidden=4)).double()
        # Nonzero state-dependent drifts exercise the full path probability ratio.
        with torch.no_grad():
            model.forward_drift.net[-1].weight.normal_(0, .03)
            model.backward_drift.net[-1].weight.normal_(0, .03)
            proposals = [model.sample(2000, 2) for _ in range(12)]
        theta = np.concatenate([p['theta'].numpy() for p in proposals]).reshape(4, 6000, 3, 3)
        weights = np.concatenate([p['log_weight'].numpy() for p in proposals]).reshape(4, 6000)
        rng = np.random.default_rng(711)
        chain = independence_mh(theta, weights, np.log(rng.random((4, 5999))), 999)
        seed = njit(lambda n: np.random.seed(n))
        reference = []
        seed(712)
        a = np.zeros((3, 3))
        for j in range(20500):
            wolff_step_xy(a, .2)
            if j >= 500 and j % 5 == 0:
                reference.append(a.copy())
        ref = measurements(np.array(reference))
        obs = measurements(chain['theta'])
        self.assertLess(abs(obs['energy'].mean()-ref['energy'].mean()), .02)
        self.assertLess(abs(obs['magnetization'].mean()-ref['magnetization'].mean()), .025)

    def test_interrupted_training_resume_reproduces_uninterrupted_weights(self):
        import argparse
        import json
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
        import sps_experiment as cli
        with tempfile.TemporaryDirectory(prefix='sps-resume-') as d:
            root = Path(d)
            config = root/'config.json'
            config.write_text(json.dumps(dict(L=3, hidden=4, train_steps=2, batch_size=3,
                                              updates=3, checkpoint_every=1)), encoding='utf-8')
            def args(out, resume=False):
                return argparse.Namespace(config=str(config), out=str(root/out), device='cpu',
                                          updates=None, seed=None, resume=resume)
            cli.train(args('continuous'))
            real_write = cli.write_json
            def simulate_interruption(path, obj):
                if Path(path).name == 'training_log.json':
                    raise InterruptedError('Simulated interruption after atomic checkpoint')
                real_write(path, obj)
            with patch.object(cli, 'write_json', side_effect=simulate_interruption):
                with self.assertRaises(InterruptedError):
                    cli.train(args('resumed'))
            cli.train(args('resumed', resume=True))
            a = torch.load(root/'continuous/checkpoint.pt', weights_only=False)
            b = torch.load(root/'resumed/checkpoint.pt', weights_only=False)
            for key in a['model']:
                torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
