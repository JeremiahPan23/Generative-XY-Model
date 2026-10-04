"""Independent physical checks and reproducibility checks for the saved diagnostic."""
from pathlib import Path
import sys
import unittest
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
from xy_observables import calculate_vorticity, energy_density, observables, mean_summary
from compare_fm_wolff import seed_wolff
from sampler_wolff import wolff_step_xy


class PhysicsChecks(unittest.TestCase):
    def test_aligned_and_checkerboard_energy(self):
        aligned = np.full((8, 8), .31)
        self.assertAlmostEqual(float(energy_density(aligned)), -1)
        self.assertEqual(float(observables(aligned)['pairs']), 0)
        checkerboard = np.indices((8, 8)).sum(axis=0) % 2 * np.pi
        self.assertAlmostEqual(float(energy_density(checkerboard)), 1)

    def test_energy_against_explicit_unique_bonds(self):
        a = np.random.default_rng(71).uniform(-np.pi, np.pi, (7, 7))
        total = sum(-np.cos(a[i,j]-a[(i+1)%7,j])-np.cos(a[i,j]-a[i,(j+1)%7])
                    for i in range(7) for j in range(7))
        self.assertAlmostEqual(float(energy_density(a)), total/(2*7*7))

    def test_charge_against_complex_bond_angles(self):
        a = np.random.default_rng(42).uniform(-np.pi, np.pi, (8, 8))
        z = np.exp(1j*a)
        expected = np.zeros((8, 8), int)
        for i in range(8):
            for j in range(8):
                loop = [z[i,j], z[(i+1)%8,j], z[(i+1)%8,(j+1)%8], z[i,(j+1)%8]]
                expected[i,j] = round(sum(np.angle(loop[(k+1)%4]/loop[k])
                                         for k in range(4))/(2*np.pi))
        np.testing.assert_array_equal(calculate_vorticity(a), expected)
        self.assertEqual(int(expected.sum()), 0)

    def test_global_rotation_and_periodic_translation(self):
        a = np.random.default_rng(7).uniform(-np.pi, np.pi, (3, 8, 8))
        b = np.roll(a+.79, (2, 3), axis=(-2, -1))
        x, y = observables(a), observables(b)
        for key in x:
            np.testing.assert_allclose(x[key], y[key], atol=1e-12)

    def test_block_uncertainty_detects_repeated_correlated_samples(self):
        independent = np.random.default_rng(1).normal(size=(4, 32))
        repeated = np.repeat(independent, 16, axis=1)
        blocked = mean_summary(repeated, block_size=16)
        naive = mean_summary(repeated)
        self.assertGreater(blocked['sem'], 3*naive['sem'])
        self.assertEqual(blocked, mean_summary(repeated, block_size=16))

    def test_numba_rng_reproducible(self):
        a, b = np.zeros((8, 8)), np.zeros((8, 8))
        seed_wolff(88)
        for _ in range(20):
            wolff_step_xy(a, 1.12)
        seed_wolff(88)
        for _ in range(20):
            wolff_step_xy(b, 1.12)
        np.testing.assert_array_equal(a, b)


if __name__ == '__main__':
    unittest.main()
