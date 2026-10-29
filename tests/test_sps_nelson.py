"""Small CPU checks for the bounded Nelson comparison, not research results."""
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sps_xy import energy
from sps_nelson import (TARGETS, MultiTemperaturePathSampler, NelsonConfig,
                        grad_total_energy, multi_temperature_loss)

torch.set_num_threads(2)


class NelsonChecks(unittest.TestCase):
    def model(self, arm='nelson', **kwargs):
        return MultiTemperaturePathSampler(
            NelsonConfig(L=3, hidden=4, train_steps=5, **kwargs), arm).double()

    def test_total_energy_gradient_matches_autograd_and_explicit_bonds(self):
        torch.manual_seed(83)
        theta = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
        analytic = grad_total_energy(theta, J=1.7)
        exact, = torch.autograd.grad(energy(theta, J=1.7).sum(), theta,
                                     create_graph=True)
        torch.testing.assert_close(analytic, exact, rtol=1e-12, atol=1e-12)
        values = theta.detach().numpy()
        expected = np.zeros_like(values)
        for batch in range(2):
            for i in range(3):
                for j in range(3):
                    # Differentiate explicitly enumerated right/down bonds,
                    # accumulating both endpoints' contributions.
                    for ni, nj in (((i + 1) % 3, j), (i, (j + 1) % 3)):
                        derivative = 1.7 * np.sin(values[batch, i, j] - values[batch, ni, nj])
                        expected[batch, i, j] += derivative
                        expected[batch, ni, nj] -= derivative
        np.testing.assert_allclose(analytic.detach().numpy(), expected, atol=1e-12)
        self.assertTrue(torch.isfinite(torch.autograd.grad(analytic.square().sum(), theta)[0]).all())

    def test_energy_gradient_periodicity_global_rotation_and_zero_total_torque(self):
        theta = torch.randn(4, 4, 4, dtype=torch.float64)
        shift = torch.randint(-3, 4, theta.shape).double() * (2 * math.pi)
        torch.testing.assert_close(grad_total_energy(theta + shift), grad_total_energy(theta))
        torch.testing.assert_close(grad_total_energy(theta + 1.217), grad_total_energy(theta))
        torch.testing.assert_close(grad_total_energy(theta).sum((-2, -1)),
                                   torch.zeros(4, dtype=torch.float64), atol=1e-12, rtol=0)

    def test_nelson_relation_uses_squared_nonunit_noise_and_preserves_gradients(self):
        model = self.model()
        with torch.no_grad():
            probability = (1.5 - model.config.sigma_min) / (model.config.sigma_max - model.config.sigma_min)
            model.noise_net[-1].bias.fill_(math.log(probability / (1 - probability)))
            model.forward_drift.net[-1].bias.fill_(0.37)
        theta = torch.randn(2, 3, 3, dtype=torch.float64, requires_grad=True)
        time = 0.617
        self.assertAlmostEqual(float(model.sigma(time).detach()), 1.5, places=12)
        expected = -model.sigma(time).square() * (time / .87) * grad_total_energy(theta)
        actual = model.forward_drift(theta, time) + model.drift_backward(theta, time)
        torch.testing.assert_close(actual, expected)
        gradient, = torch.autograd.grad(model.drift_backward(theta, time).square().sum(), theta)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0)

    def test_derived_backward_is_not_clamped_to_network_bound(self):
        model = self.model(drift_bound=0.1)
        theta = torch.zeros(1, 3, 3, dtype=torch.float64)
        theta[0, 0, 0] = math.pi / 2
        result = model.drift_backward(theta, 1).detach()
        self.assertGreater(float(result.abs().max()), 0.1)
        expected = -float(model.sigma(1).detach())**2 * 4 / .87
        self.assertAlmostEqual(float(result[0, 0, 0]), expected, places=12)

    def test_common_forward_and_noise_initialization_and_paired_paths_match(self):
        torch.manual_seed(11)
        free = self.model('free')
        torch.manual_seed(11)
        nelson = self.model('nelson')
        for module in ('forward_drift', 'noise_net'):
            left = getattr(free, module).state_dict()
            right = getattr(nelson, module).state_dict()
            self.assertEqual(left.keys(), right.keys())
            for name in left:
                torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
        a = free.sample_prefixes(3, generator=torch.Generator().manual_seed(12), keep_path=True)
        b = nelson.sample_prefixes(3, generator=torch.Generator().manual_seed(12), keep_path=True)
        torch.testing.assert_close(a['path'], b['path'], rtol=0, atol=0)
        for p, q in zip(a['prefixes'], b['prefixes']):
            torch.testing.assert_close(p['log_qf'], q['log_qf'], rtol=0, atol=0)

    def test_default_training_grid_and_prefixes_hit_all_exact_targets(self):
        model = MultiTemperaturePathSampler(NelsonConfig(hidden=4)).double()
        grid = model.training_grid()
        self.assertEqual(len(grid), 33)
        self.assertEqual(grid[0], 0)
        self.assertEqual(grid[-1], 1)
        self.assertTrue(all(a < b for a, b in zip(grid[:-1], grid[1:])))
        np.testing.assert_allclose(np.diff(grid[:31]), .87 / .92 / 30, atol=1e-16)
        batch = model.sample_prefixes(2, keep_path=True)
        self.assertEqual(batch['path'].shape, (2, 33, 4, 4))
        for index, prefix, target in zip((30, 31, 32), batch['prefixes'], TARGETS):
            self.assertEqual(prefix['tag'], target['tag'])
            self.assertEqual(prefix['T'], target['T'])
            self.assertEqual(prefix['path_index'], index)
            self.assertAlmostEqual(prefix['t_stop'], .87 / target['T'], places=14)
            self.assertAlmostEqual(prefix['beta'], 1 / target['T'], places=14)
            torch.testing.assert_close(prefix['theta'], batch['path'][:, index])

    def test_prefix_path_weights_match_independent_periodized_density_sum(self):
        for arm in ('free', 'nelson'):
            model = self.model(arm)
            with torch.no_grad():
                model.forward_drift.net[-1].bias.fill_(.2)
                if arm == 'free':
                    model.backward_drift.net[-1].bias.fill_(-.15)
                batch = model.sample_prefixes(2, keep_path=True)
            path = batch['path'].numpy()
            grid = batch['grid']
            qf = np.full(2, -9 * np.log(2 * np.pi))
            qb = np.zeros(2)
            forward = 12 * np.tanh(.2 / 12)
            backward = 12 * np.tanh(-.15 / 12)
            def kernel(value, mean, std):
                residual = (value - mean + np.pi) % (2 * np.pi) - np.pi
                images = residual[..., None] + 2 * np.pi * np.arange(-24, 25)
                return logsumexp(-.5 * (images / std)**2, axis=-1) - np.log(std * np.sqrt(2 * np.pi))
            prefixes = {p['path_index']: p for p in batch['prefixes']}
            for index, (t0, t1) in enumerate(zip(grid[:-1], grid[1:]), 1):
                dt = t1 - t0
                theta, next_theta = path[:, index - 1], path[:, index]
                sigma_mid = float(model.sigma((t0 + t1) / 2).detach())
                std = sigma_mid * np.sqrt(dt)
                mean_f = theta + dt * forward
                if arm == 'free':
                    reverse_drift = backward
                else:
                    gradient = sum(np.sin(next_theta - np.roll(next_theta, shift, axis))
                                   for axis in (-2, -1) for shift in (-1, 1))
                    sigma_right = float(model.sigma(t1).detach())
                    reverse_drift = -forward - sigma_right**2 * (t1 / .87) * gradient
                mean_b = next_theta + dt * reverse_drift
                qf += kernel(next_theta, mean_f, std).sum((-2, -1))
                qb += kernel(theta, mean_b, std).sum((-2, -1))
                if index in prefixes:
                    prefix = prefixes[index]
                    expected = -prefix['beta'] * prefix['energy'].numpy() + qb - qf
                    np.testing.assert_allclose(prefix['log_qf'].numpy(), qf, atol=1e-10, rtol=0)
                    np.testing.assert_allclose(prefix['log_qb'].numpy(), qb, atol=1e-10, rtol=0)
                    np.testing.assert_allclose(prefix['log_weight'].numpy(), expected, atol=1e-10, rtol=0)

    def test_sampling_uses_absolute_clock_and_terminal_beta(self):
        model = self.model()
        forward_times, backward_times = [], []
        forward, backward = model.forward_drift.forward, model.drift_backward
        def record_forward(theta, t):
            forward_times.append(t)
            return forward(theta, t)
        def record_backward(theta, t):
            backward_times.append(t)
            return backward(theta, t)
        with patch.object(model.forward_drift, 'forward', side_effect=record_forward), \
                patch.object(model, 'drift_backward', side_effect=record_backward):
            batch = model.sample(2, 3, t_stop=.6, keep_path=True)
        # Nelson backward evaluation also invokes forward at the right endpoint.
        np.testing.assert_allclose(forward_times, [0, .2, .2, .4, .4, .6], atol=1e-15)
        np.testing.assert_allclose(backward_times, [.2, .4, .6], atol=1e-15)
        self.assertEqual(batch['grid'][-1], .6)
        self.assertAlmostEqual(batch['beta'], .6 / .87, places=14)
        self.assertEqual(batch['path'].shape, (2, 4, 3, 3))

    def test_both_arms_have_finite_pathwise_gradients_for_all_parameters(self):
        for arm in ('free', 'nelson'):
            model = self.model(arm)
            batch = model.sample_prefixes(4)
            loss = multi_temperature_loss(batch, 3)
            self.assertTrue(torch.isfinite(loss))
            loss.backward()
            for name, param in model.named_parameters():
                self.assertIsNotNone(param.grad, f'{arm}: {name}')
                self.assertTrue(torch.isfinite(param.grad).all(), f'{arm}: {name}')
            self.assertGreater(float(model.forward_drift.net[-1].weight.grad.abs().sum()), 0)
            self.assertGreater(float(model.noise_net[-1].bias.grad.abs().sum()), 0)

    def test_loss_averages_three_objectives_without_lattice_or_beta_mixup(self):
        batch = dict(prefixes=[dict(log_weight=torch.tensor([value, value + 2.], dtype=torch.float64))
                               for value in (10., 20., 30.)])
        self.assertAlmostEqual(float(multi_temperature_loss(batch, 4)), -21 / 16, places=14)

    def test_zero_energy_zero_drift_uniform_target_has_constant_prefix_and_endpoint_weights(self):
        for arm in ('free', 'nelson'):
            model = self.model(arm, J=0)
            with torch.no_grad():
                batch = model.sample_prefixes(5)
                samples = [model.sample(5, steps, t_stop=stop)
                           for steps, stop in ((1, .5), (4, .9), (16, 1))]
            expected = torch.full((5,), 9 * math.log(2 * math.pi), dtype=torch.float64)
            for sample in batch['prefixes'] + samples:
                torch.testing.assert_close(sample['log_weight'], expected, atol=3e-11, rtol=0)
                torch.testing.assert_close(sample['energy'], torch.zeros_like(expected), atol=0, rtol=0)

    def test_invalid_arm_config_or_sampling_clock_is_rejected(self):
        with self.assertRaises(ValueError):
            self.model('unknown')
        for changes in (dict(train_steps=2), dict(T_min=.9), dict(J=-1),
                        dict(sigma_max=2.1), dict(seed=-1), dict(hidden=True)):
            with self.assertRaises(ValueError):
                NelsonConfig(**changes).validate()
        model = self.model()
        for stop in (0, -1, 1.1, math.nan):
            with self.assertRaises(ValueError):
                model.sample(2, 3, t_stop=stop)
        for batch_size, steps in ((0, 3), (1.5, 3), (2, 0), (2, True)):
            with self.assertRaises(ValueError):
                model.sample(batch_size, steps)


if __name__ == '__main__':
    unittest.main()
