"""Small CPU checks for prefix-density replay and finite-bank IMH evaluation.

These tests use toy models and synthetic banks, not production sampling.
"""
import copy
import math
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from sps_intermediate import collect_raw_frames, frame_metadata, make_grid
from sps_intermediate_efficiency import collect_scored_replay, evaluate_scored_paths
from sps_nelson import MultiTemperaturePathSampler, NelsonConfig

torch.set_num_threads(1)


def toy_model(arm='nelson', *, J=1., dtype=torch.float64):
    torch.manual_seed(93)
    model = MultiTemperaturePathSampler(
        NelsonConfig(L=3, hidden=4, train_steps=3, J=J), arm).to(dtype=dtype)
    with torch.no_grad():
        model.forward_drift.net[-1].weight.normal_(std=.035)
        model.forward_drift.net[-1].bias.fill_(.14)
        if arm == 'free':
            model.backward_drift.net[-1].weight.normal_(std=.03)
            model.backward_drift.net[-1].bias.fill_(-.09)
        model.noise_net[-1].weight.fill_(.11)
    return model.eval()


def synthetic_bank():
    """Chain-major independent candidates, with visible unique labels."""
    paths, frames, L = 12, 3, 3
    theta = np.broadcast_to(np.linspace(-2, 2, paths)[:, None, None, None],
                            (paths, frames, L, L)).copy()
    weights = np.empty((paths, frames))
    weights[:, 0] = 9 * math.log(2 * math.pi)
    weights[:, 1] = [0, 8, -8, 0, 7, -9, 2, -9, 10, -10, 9, -8]
    weights[:, 2] = np.linspace(-3, 2, paths) + 10000
    return theta, weights, frame_metadata([0, .37, 1])


def manual_mh(theta, weights, logu, burn):
    """Independent scalar chain loop; initial state is not a transition."""
    C, K = weights.shape
    draws = K - 1 - burn
    result = np.empty((C, draws, *theta.shape[2:]), dtype=theta.dtype)
    retained_weight = np.empty((C, draws))
    accepted = np.zeros((C, draws), dtype=bool)
    alpha = np.empty((C, draws))
    for chain in range(C):
        current = theta[chain, 0].copy()
        weight = float(weights[chain, 0])
        for index in range(1, K):
            probability = math.exp(min(0., float(weights[chain, index]) - weight))
            take = float(logu[chain, index - 1]) < math.log(probability)
            if take:
                current = theta[chain, index].copy()
                weight = float(weights[chain, index])
            if index > burn:
                out = index - burn - 1
                result[chain, out] = current
                retained_weight[chain, out] = weight
                accepted[chain, out] = take
                alpha[chain, out] = probability
    return dict(theta=result, log_weight=retained_weight,
                accepted=accepted, alpha=alpha)


class PrefixReplayTests(unittest.TestCase):
    def test_nonuniform_replay_matches_saved_frames_and_frozen_prefix_scores(self):
        mesh = make_grid(5, [0, .13, .61, 1])
        for arm in ('free', 'nelson'):
            for dtype in (torch.float32, torch.float64):
                with self.subTest(arm=arm, dtype=dtype):
                    model = toy_model(arm, dtype=dtype)
                    raw = collect_raw_frames(model, 7, mesh, 3, 401)
                    saved = raw.copy()
                    scored = collect_scored_replay(model, mesh, saved, 3, 401)
                    np.testing.assert_array_equal(saved, raw)  # never mutate the input
                    np.testing.assert_array_equal(scored['theta'], raw)
                    np.testing.assert_array_equal(scored['path_id'], np.arange(7))
                    np.testing.assert_array_equal(scored['grid'], mesh['grid'])
                    np.testing.assert_array_equal(scored['capture_indices'], mesh['capture_indices'])
                    expected = {key: [] for key in ('log_qf', 'log_qb', 'log_weight', 'energy_total')}
                    rng = torch.Generator().manual_seed(401)
                    with torch.no_grad():
                        for first in range(0, 7, 3):
                            n = min(3, 7-first)
                            end, prefixes = model._sample_grid(n, mesh['grid'].tolist(), rng,
                                                               True, tuple(mesh['capture_indices'][1:]))
                            initial = model._endpoint(end['path'][:, 0],
                                                      torch.full((n,), -9*math.log(2*math.pi), dtype=torch.float64),
                                                      torch.zeros(n, dtype=torch.float64), 0.)
                            for key in expected:
                                source = 'energy' if key == 'energy_total' else key
                                expected[key].append(torch.stack([p[source] for p in [initial]+prefixes], 1).numpy())
                    for key, batches in expected.items():
                        self.assertEqual(scored[key].dtype, np.float64)
                        np.testing.assert_allclose(scored[key], np.concatenate(batches), atol=1e-10, rtol=0)
                    self.assertFalse(any(p.grad is not None for p in model.parameters()))

    def test_independent_density_sum_uses_actual_dt_left_forward_and_right_backward(self):
        # Full toy path is kept only by the independent reference, enabling an
        # explicit 49-image NumPy kernel sum at every transition.
        model = toy_model('free')
        mesh = make_grid(3, [0, .17, .73, 1])
        raw = collect_raw_frames(model, 2, mesh, 2, 613)
        replay = collect_scored_replay(model, mesh, raw, 2, 613)
        with torch.no_grad():
            path = model._sample_grid(2, mesh['grid'].tolist(),
                                     torch.Generator().manual_seed(613), True)[0]['path']
        qf = np.full(2, -9*math.log(2*math.pi))
        qb = np.zeros(2)
        def kernel(value, mean, std):
            residual = (value-mean+math.pi) % (2*math.pi)-math.pi
            images = residual[..., None] + 2*math.pi*np.arange(-24, 25)
            return logsumexp(-.5*(images/std)**2, axis=-1)-math.log(std*math.sqrt(2*math.pi))
        for index, (left, right) in enumerate(zip(mesh['grid'][:-1], mesh['grid'][1:]), 1):
            dt = float(right-left)
            old, new = path[:, index-1], path[:, index]
            with torch.no_grad():
                std = float(model.sigma(float((left+right)/2)))*math.sqrt(dt)
                fmean = (old+dt*model.forward_drift(old, float(left))).numpy()
                bmean = (new+dt*model.drift_backward(new, float(right))).numpy()
            qf += kernel(new.numpy(), fmean, std).sum((-2, -1))
            qb += kernel(old.numpy(), bmean, std).sum((-2, -1))
            if index in mesh['capture_indices']:
                slot = int(np.flatnonzero(mesh['capture_indices'] == index)[0])
                np.testing.assert_allclose(replay['log_qf'][:, slot], qf, atol=1e-10, rtol=0)
                np.testing.assert_allclose(replay['log_qb'][:, slot], qb, atol=1e-10, rtol=0)
                a = new.numpy()
                energy = -sum(np.cos(a-np.roll(a, -1, axis)) for axis in (-2, -1)).sum((-2, -1))
                expected = -float(right)/.87*energy+qb-qf
                np.testing.assert_allclose(replay['log_weight'][:, slot], expected, atol=1e-10, rtol=0)

    def test_tampered_capture_at_every_position_fails_closed(self):
        model = toy_model()
        mesh = make_grid(3, [0, .17, .73, 1])
        raw = collect_raw_frames(model, 5, mesh, 2, 421)
        for frame in range(raw.shape[1]):
            changed = raw.copy()
            changed[-1, frame, 1, 2] += 1e-10
            with self.subTest(frame=frame), self.assertRaises(ValueError):
                collect_scored_replay(model, mesh, changed, 2, 421)

    def test_wrong_seed_batch_partition_or_forward_drift_cannot_score_saved_bank(self):
        model = toy_model('free')
        mesh = make_grid(4, [0, .21, 1])
        raw = collect_raw_frames(model, 7, mesh, 3, 421)
        for batch, seed in ((3, 422), (4, 421)):
            with self.subTest(batch=batch, seed=seed), self.assertRaises(ValueError):
                collect_scored_replay(model, mesh, raw, batch, seed)
        forward = model.forward_drift.forward
        with patch.object(model.forward_drift, 'forward', side_effect=lambda theta, t: forward(theta, t)+.01):
            with self.assertRaises(ValueError):
                collect_scored_replay(model, mesh, raw, 3, 421)

    def test_uniform_zero_energy_case_has_constant_weight_unit_ess_and_all_mh_accept(self):
        for arm in ('free', 'nelson'):
            model = MultiTemperaturePathSampler(
                NelsonConfig(L=3, hidden=4, train_steps=3, J=0), arm).double()
            mesh = make_grid(4, [0, .13, .68, 1])
            raw = collect_raw_frames(model, 12, mesh, 5, 829)
            replay = collect_scored_replay(model, mesh, raw, 5, 829)
            np.testing.assert_allclose(replay['energy_total'], 0, atol=0, rtol=0)
            np.testing.assert_allclose(replay['log_weight'], 9*math.log(2*math.pi), atol=3e-11, rtol=0)
            evaluated = evaluate_scored_paths(raw, replay['log_weight'], frame_metadata(mesh['capture_times']),
                                              chains=2, candidates_per_chain=6, burn=1, seed=823)
            for row, frame in zip(evaluated['summary'], evaluated['frames']):
                self.assertAlmostEqual(row['importance_ess_fraction'], 1, places=12)
                self.assertAlmostEqual(row['importance_ess'], 12, places=12)
                self.assertEqual(row['accepted_fraction'], 1)
                self.assertTrue(frame['accepted'].all())

    def test_invalid_replay_shape_finite_seed_or_mesh_is_rejected(self):
        model = toy_model()
        mesh = make_grid(3, [0, .2, 1])
        raw = collect_raw_frames(model, 4, mesh, 2, 23)
        for seed in (-1, True, 2**63, 3.1):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                collect_scored_replay(model, mesh, raw, 2, seed)
        for batch in (0, True, 1.5):
            with self.subTest(batch=batch), self.assertRaises(ValueError):
                collect_scored_replay(model, mesh, raw, batch, 23)
        for bad in (raw[:, :-1], raw[:, :, :-1], raw[:, :, :, :2], raw[0]):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                collect_scored_replay(model, mesh, bad, 2, 23)
        bad = raw.copy(); bad[0, 0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            collect_scored_replay(model, mesh, bad, 2, 23)
        malformed = copy.deepcopy(mesh); malformed['capture_indices'][1] = 0
        with self.assertRaises(ValueError):
            collect_scored_replay(model, malformed, raw, 2, 23)

    def test_invalid_model_config_dtype_parameters_or_density_fails_closed(self):
        mesh = make_grid(3, [0, .2, 1])
        model = toy_model()
        raw = collect_raw_frames(model, 4, mesh, 2, 23)
        with self.assertRaises(ValueError):
            collect_scored_replay(model, mesh, raw.astype(np.float32), 2, 23)
        model.config.T_min = .9
        with self.assertRaises(ValueError):
            collect_scored_replay(model, mesh, raw, 2, 23)
        model.config.T_min = .87
        with patch.object(model, 'drift_backward', side_effect=lambda theta, t: torch.full_like(theta, math.nan)):
            with self.assertRaises(ValueError):
                collect_scored_replay(model, mesh, raw, 2, 23)
        with torch.no_grad():
            next(model.parameters()).flatten()[0] = math.inf
        with self.assertRaises(ValueError):
            collect_scored_replay(model, mesh, raw, 2, 23)


class FiniteBankMHTests(unittest.TestCase):
    def evaluate(self, theta=None, weights=None, frames=None, **kwargs):
        saved, w, metadata = synthetic_bank()
        return evaluate_scored_paths(saved if theta is None else theta,
                                     w if weights is None else weights,
                                     metadata if frames is None else frames,
                                     chains=2, candidates_per_chain=6,
                                     burn=kwargs.pop('burn', 1), seed=kwargs.pop('seed', 711), **kwargs)

    def test_chain_axes_initial_exclusion_burn_and_rejected_repeats(self):
        theta, weights, _ = synthetic_bank()
        evaluated = self.evaluate()
        for slot, frame in enumerate(evaluated['frames']):
            proposals = theta[:, slot].reshape(2, 6, 3, 3)
            scores = weights[:, slot].reshape(2, 6)
            np.testing.assert_array_equal(frame['proposal_theta'], proposals)
            np.testing.assert_array_equal(frame['proposal_log_weight'], scores)
            expected_uniforms = np.log1p(-np.random.default_rng(711+1009*slot).random((2, 5)))
            np.testing.assert_array_equal(frame['log_uniform'], expected_uniforms)
            expected = manual_mh(proposals, scores, expected_uniforms, 1)
            for key in expected:
                np.testing.assert_allclose(frame[key], expected[key], atol=0, rtol=0)
            self.assertEqual(frame['theta'].shape, (2, 4, 3, 3))
            row = evaluated['summary'][slot]
            self.assertEqual(row['candidate_count'], 12)
            self.assertEqual(row['retained_count'], 8)
            self.assertEqual(row['retained_draws_per_chain'], 4)
            self.assertEqual(row['accepted_count'], int(expected['accepted'].sum()))
            self.assertEqual(row['accepted_fraction'], expected['accepted'].mean())
            np.testing.assert_array_equal(row['acceptance_by_chain'], expected['accepted'].mean(axis=1))
        # Frame one climbs to a large weight then rejects low-weight candidates.
        self.assertTrue((~evaluated['frames'][1]['accepted']).any())
        repeated = np.diff(evaluated['frames'][1]['theta'], axis=1) == 0
        self.assertTrue(repeated.all(axis=(-2, -1)).any())

    def test_ess_uses_all_candidates_including_initial_and_burn_and_is_stable(self):
        _, weights, _ = synthetic_bank()
        result = self.evaluate()
        for index, row in enumerate(result['summary']):
            w = np.exp(weights[:, index]-weights[:, index].max())
            expected = float(w.sum()**2 / np.square(w).sum())
            self.assertAlmostEqual(row['importance_ess'], expected, places=12)
            self.assertAlmostEqual(row['importance_ess_fraction'], expected/12, places=12)
            self.assertAlmostEqual(row['largest_normalized_importance_weight'], float(w.max()/w.sum()), places=12)
        no_burn = self.evaluate(burn=0)
        for row, other in zip(result['summary'], no_burn['summary']):
            self.assertEqual(row['importance_ess'], other['importance_ess'])
            self.assertEqual(row['importance_ess_fraction'], other['importance_ess_fraction'])
            self.assertEqual(other['retained_count'], 10)

    def test_constant_shift_of_log_weights_preserves_ess_and_mh_states(self):
        theta, weights, frames = synthetic_bank()
        first = self.evaluate()
        second = self.evaluate(theta, weights + 123456, frames)
        for a, b, sa, sb in zip(first['frames'], second['frames'], first['summary'], second['summary']):
            np.testing.assert_array_equal(a['theta'], b['theta'])
            np.testing.assert_array_equal(a['accepted'], b['accepted'])
            self.assertAlmostEqual(sa['importance_ess_fraction'], sb['importance_ess_fraction'], places=12)

    def test_infinite_temperature_is_recorded_but_not_plotted(self):
        rows = self.evaluate()['summary']
        self.assertIsNone(rows[0]['temperature'])
        self.assertFalse(rows[0]['include_in_temperature_plot'])
        self.assertTrue(all(row['include_in_temperature_plot'] for row in rows[1:]))

    def test_invalid_bank_finite_inputs_metadata_and_mh_configuration_fail(self):
        theta, weights, metadata = synthetic_bank()
        for bad in (theta[:-1], theta[:, :-1], theta[:, :, :-1], theta[0]):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                self.evaluate(theta=bad)
        for bad in (weights[:-1], weights[:, :-1], weights[0]):
            with self.subTest(shape=bad.shape), self.assertRaises(ValueError):
                self.evaluate(weights=bad)
        for value in (np.nan, np.inf, -np.inf):
            bad = weights.copy(); bad[-1, -1] = value
            with self.subTest(nonfinite=value), self.assertRaises(ValueError):
                self.evaluate(weights=bad)
        for burn in (-1, True, 1.5, 5, 6):
            with self.subTest(burn=burn), self.assertRaises(ValueError):
                self.evaluate(burn=burn)
        for seed in (-1, True, 1.5, 2**63):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                self.evaluate(seed=seed)
        changed = copy.deepcopy(metadata); changed[1]['beta'] = 7
        with self.assertRaises(ValueError):
            self.evaluate(frames=changed)
        with self.assertRaises(ValueError):
            self.evaluate(frames=metadata[:-1])

    def test_invalid_chain_partition_and_derived_seed_are_rejected(self):
        theta, weights, metadata = synthetic_bank()
        for chains, candidates in ((0, 6), (True, 6), (1.5, 6), (2, 0),
                                   (2, True), (2, 3.5), (3, 6), (2, 1)):
            with self.subTest(chains=chains, candidates=candidates), self.assertRaises(ValueError):
                evaluate_scored_paths(theta, weights, metadata, chains=chains,
                                      candidates_per_chain=candidates, burn=0, seed=1)
        with self.assertRaisesRegex(ValueError, 'seed'):
            self.evaluate(seed=2**63-2)


if __name__ == '__main__':
    unittest.main()
