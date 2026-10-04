"""Generate a temperature sweep of Wolff configurations for the 2D XY model.

This is the mainline data-generation script for the quantitative BKT
visualisation: for every requested T we store N_samples independent
(cos, sin) configurations with fixed burn-in and decorrelation interval.

Example:
    python src/generate_wolff_sweep.py --T 0.5 0.7 0.8 0.89 1.0 1.2 1.5 --N 2000
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from sampler_wolff import wolff_step_xy

DEFAULT_DATA_DIR = str(Path(__file__).resolve().parents[1] / 'data')


def generate_temperature(T, N_samples, L=32, burn_in=1000, interval=50,
                         seed=0, out_dir=DEFAULT_DATA_DIR):
    beta = 1.0 / T
    rng = np.random.default_rng(seed)

    theta = rng.uniform(0.0, 2.0 * np.pi, size=(L, L)).astype(np.float64)
    wolff_step_xy(theta.copy(), beta)  # Numba warm-up

    for _ in range(burn_in):
        theta = wolff_step_xy(theta, beta)

    samples = np.zeros((N_samples, L, L), dtype=np.float64)
    for i in tqdm(range(N_samples), desc=f'T={T}', leave=False):
        for _ in range(interval):
            theta = wolff_step_xy(theta, beta)
        samples[i] = theta

    tensor = torch.stack([
        torch.from_numpy(np.cos(samples)).float(),
        torch.from_numpy(np.sin(samples)).float(),
    ], dim=1)

    os.makedirs(out_dir, exist_ok=True)
    tag = f'{T:g}'.replace('.', '_')
    path = os.path.join(out_dir, f'xy_L{L}_T{tag}_N{N_samples}.pt')
    torch.save(tensor, path)
    print(f'saved {tuple(tensor.shape)} -> {path}')
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--T', nargs='+', type=float,
                        default=[0.5, 0.7, 0.8, 0.89, 1.0, 1.2, 1.5])
    parser.add_argument('--N', type=int, default=2000)
    parser.add_argument('--L', type=int, default=32)
    parser.add_argument('--burn-in', type=int, default=1000)
    parser.add_argument('--interval', type=int, default=50)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--out-dir', type=str, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    for i, T in enumerate(args.T):
        t0 = time.time()
        seed = args.seed + i
        generate_temperature(T, args.N, L=args.L, burn_in=args.burn_in,
                             interval=args.interval, seed=seed,
                             out_dir=args.out_dir)
        print(f'T={T}: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
