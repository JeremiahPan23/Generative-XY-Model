"""Portable synthetic checks of intermediate frames and physical estimators.

No historical configuration, checkpoint, reference archive or cluster CLI is
required. These tiny CPU checks test implementation, not research performance.
"""
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from sps_intermediate import (collect_raw_frames, frame_metadata, histogram_frames,
                              make_grid, measure_frames, summarize_frames)
from sps_nelson import MultiTemperaturePathSampler, NelsonConfig
from sps_xy import wrap

torch.set_num_threads(1)


def synthetic_model(arm='nelson'):
    torch.manual_seed(61)
    model = MultiTemperaturePathSampler(
        NelsonConfig(L=4, hidden=4, train_steps=3), arm)
    # Nonzero drift and time-dependent noise expose clock/grid regressions.
    with torch.no_grad():
        model.forward_drift.net[-1].weight.normal_(std=.12)
        model.forward_drift.net[-1].bias.fill_(.17)
        model.noise_net[-1].weight.fill_(.13)
    return model.eval()


class GridChecks(unittest.TestCase):
    def test_requested_capture_nodes_are_exact_and_insertion_is_counted(self):
        times = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, .87 / .92, .9744, 1]
        mesh = make_grid(512, times)
        np.testing.assert_array_equal(mesh['grid'][mesh['capture_indices']], times)
        self.assertEqual(len(mesh['dt']), 522)
        self.assertTrue((mesh['dt'] > 0).all())
        self.assertAlmostEqual(mesh['dt'].sum(), 1)
        with self.assertRaises(ValueError):
            make_grid(512, times, 'strict')

    def test_invalid_capture_times_and_infinity_temperature_label(self):
        for times in ([0, .2, .1, 1], [0, .1, .1, 1], [.1, 1],
                      [0, math.nan, 1], [0, 1.1, 1]):
            with self.subTest(times=times), self.assertRaises(ValueError):
                make_grid(16, times)
        frames = frame_metadata([0, .1, .2, 1])
        self.assertIsNone(frames[0]['temperature'])
        self.assertEqual(frames[0]['beta'], 0)
        np.testing.assert_allclose([row['temperature'] for row in frames[1:]],
                                   [8.7, 4.35, .87])


class PathChecks(unittest.TestCase):
    def test_uniform_grid_matches_same_kernel_full_path_both_arms(self):
        for arm in ('free', 'nelson'):
            with self.subTest(arm=arm):
                model = synthetic_model(arm)
                mesh = make_grid(16, [0, .125, .5, 1], 'strict')
                with torch.no_grad():
                    reference = model.sample(5, 16,
                        generator=torch.Generator().manual_seed(832), keep_path=True)
                captured = collect_raw_frames(model, 5, mesh, 5, 832)
                np.testing.assert_array_equal(captured,
                    reference['path'][:, mesh['capture_indices']].numpy())

    def test_inserted_grid_matches_same_kernel_and_omits_backward_scoring(self):
        model = synthetic_model()
        mesh = make_grid(8, [0, .1, .2, .51, 1])
        with torch.no_grad():
            reference, _ = model._sample_grid(3, mesh['grid'].tolist(),
                                              torch.Generator().manual_seed(317), True)
        with patch.object(model, 'drift_backward',
                          side_effect=AssertionError('Raw capture must not score backward')):
            captured = collect_raw_frames(model, 3, mesh, 3, 317)
        np.testing.assert_array_equal(captured,
            reference['path'][:, mesh['capture_indices']].numpy())

    def test_manual_step_uses_left_drift_midpoint_noise_and_actual_dt(self):
        class ClockModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.p = torch.nn.Parameter(torch.tensor(0., dtype=torch.float64))
                self.config = type('Config', (), {'L': 3})()

            def forward_drift(self, theta, t):
                return torch.ones_like(theta) * (1 + 3 * t)

            def sigma(self, t):
                return .2 + t

        model = ClockModel()
        mesh = make_grid(1, [0, .2, 1])
        rng = torch.Generator().manual_seed(29)
        theta = 2 * math.pi * torch.rand((2, 3, 3), dtype=torch.float64,
                                        generator=rng) - math.pi
        expected = [theta.clone()]
        for t0, t1 in ((0, .2), (.2, 1)):
            dt = t1 - t0
            eps = torch.randn(theta.shape, dtype=theta.dtype, generator=rng)
            theta = wrap(theta + dt * (1 + 3 * t0)
                         + (.2 + (t0 + t1) / 2) * math.sqrt(dt) * eps)
            expected.append(theta.clone())
        np.testing.assert_array_equal(collect_raw_frames(model, 2, mesh, 2, 29),
                                      torch.stack(expected, 1).numpy())

    def test_batch_rng_does_not_restart_and_collection_has_no_gradients(self):
        model = synthetic_model()
        mesh = make_grid(4, [0, .5, 1])
        first = collect_raw_frames(model, 8, mesh, 4, 41)
        repeated = collect_raw_frames(model, 8, mesh, 4, 41)
        np.testing.assert_array_equal(first, repeated)
        self.assertFalse(np.array_equal(first[:4], first[4:]))
        self.assertFalse(any(parameter.grad is not None for parameter in model.parameters()))


class ObservableChecks(unittest.TestCase):
    def test_aligned_fields_have_known_energy_charge_and_magnetization(self):
        values = measure_frames(np.full((6, 2, 4, 4), .73))
        np.testing.assert_allclose(values['energy_per_bond'], -1)
        np.testing.assert_allclose(values['energy_total'], -32)
        np.testing.assert_array_equal(values['vorticity_density'], 0)
        np.testing.assert_allclose(values['M2'], 256)
        np.testing.assert_allclose(values['M_abs'], 16)

    def test_observables_respect_angle_periodicity_and_charge_reflection(self):
        rng = np.random.default_rng(19)
        theta = rng.uniform(-math.pi, math.pi, (12, 3, 4, 4))
        original = measure_frames(theta)
        rotated = measure_frames(theta + .327)
        shifted = measure_frames(theta + 2 * math.pi * rng.integers(-2, 3, theta.shape))
        reflected = measure_frames(-theta)
        for key in ('energy_per_bond', 'vorticity_density', 'M2'):
            np.testing.assert_allclose(original[key], rotated[key], atol=2e-12)
            np.testing.assert_allclose(original[key], shifted[key], atol=2e-12)
        np.testing.assert_array_equal(original['vortex_charge'], -reflected['vortex_charge'])
        np.testing.assert_array_equal(original['vortex_net_charge'], 0)
        np.testing.assert_allclose(original['vorticity_density'],
                                   2 * original['vortex_pair_count'] / 16)
        self.assertTrue(np.any(original['vorticity_density'] > 0))

    def test_vector_and_radial_susceptibilities_differ_for_rotated_ordered_fields(self):
        # Four aligned configurations point in orthogonal directions. Their
        # vector magnetization fluctuates, but all four magnitudes equal V.
        theta = np.broadcast_to(np.arange(4)[:, None, None, None] * math.pi / 2,
                                (4, 2, 4, 4))
        summary = summarize_frames(measure_frames(theta), [0, 1], 4,
                                    bootstrap_draws=64)
        estimates = summary['frames'][1]['estimates']
        self.assertAlmostEqual(estimates['chi_vector_trace']['value'], 16 / .87)
        self.assertAlmostEqual(estimates['chi_component_average']['value'], 8 / .87)
        self.assertAlmostEqual(estimates['chi_radial']['value'], 0)
        self.assertAlmostEqual(estimates['chi_spin_unscaled']['value'], 16)
        self.assertEqual(summary['frames'][0]['estimates']['chi_vector_trace']['value'], 0)
        self.assertGreater(estimates['chi_vector_trace']['bootstrap_sem'], 0)

    def test_path_bootstrap_preserves_frame_pairs_and_nonlinear_moment_scaling(self):
        rng = np.random.default_rng(39)
        theta = rng.uniform(-math.pi, math.pi, (16, 1, 4, 4)).repeat(2, axis=1)
        summary = summarize_frames(measure_frames(theta), [.5, 1], 4,
                                    bootstrap_draws=128)
        early, late = [frame['estimates'] for frame in summary['frames']]
        self.assertEqual(early['energy_per_bond'], late['energy_per_bond'])
        self.assertAlmostEqual(2 * early['chi_vector_trace']['bootstrap_sem'],
                               late['chi_vector_trace']['bootstrap_sem'])
        self.assertEqual([2 * x for x in early['chi_vector_trace']['ci95']],
                         late['chi_vector_trace']['ci95'])

    def test_histograms_keep_all_records_and_blocks_preserve_chain_layout(self):
        theta = np.random.default_rng(3).uniform(-math.pi, math.pi, (16, 3, 4, 4))
        observations = measure_frames(theta)
        for histogram in histogram_frames(observations, 4).values():
            np.testing.assert_array_equal(np.asarray(histogram['counts']).sum(axis=1), 16)
        summary = summarize_frames(observations, [0, .5, 1], 4, chains=2,
                                    block_size=4, paired_frames=False, bootstrap_draws=32)
        self.assertEqual(summary['blocks_per_chain'], 2)
        with self.assertRaises(ValueError):
            summarize_frames(observations, [0, .5, 1], 4, chains=3, block_size=4)

    def test_zero_observed_vortices_has_a_nonzero_iid_upper_bound(self):
        summary = summarize_frames(measure_frames(np.zeros((16, 2, 4, 4))),
                                    [0, 1], 4, bootstrap_draws=32)
        rare = summary['frames'][1]['vortex_event_counts']
        self.assertEqual(rare['configurations_with_vortices'], 0)
        self.assertAlmostEqual(rare['iid_zero_count_upper95'], 1 - .05 ** (1 / 16))


if __name__ == '__main__':
    unittest.main()
