"""Create or reuse the saved 128-sample FM/Wolff diagnostic shared by notebooks 02/03.

Run from any working directory. Existing complete runs are verified and reused;
use --output results/another-name for a fresh run. No training occurs here.
"""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import sys
import time
from datetime import datetime, timezone

import numpy as np
import torch
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from model_dit import PhysicsInformedDiT
from sampler_wolff import wolff_step_xy
from xy_observables import observables, mean_summary, chain_checks

DEFAULT_OUTPUT = ROOT / 'results' / 'fm_wolff_baseline'
CONFIG = dict(L=32, J=1.0, beta=1.12, fm_samples=128, fm_steps=50,
              fm_batch_size=8, fm_seed=20260923, wolff_chains=4,
              wolff_samples_per_chain=256, wolff_burn_in=2000,
              wolff_interval=50, wolff_seed=20261000,
              ci_block_size=16, bootstrap_seed=20262000, bootstrap_draws=4000)
TEMPERATURES = [.5, .7, .8, .89, 1., 1.2, 1.5]


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sources():
    paths = [ROOT / 'checkpoints/best_dit_flow.pth', Path(__file__),
             ROOT / 'src/model_dit.py', ROOT / 'src/sampler_wolff.py',
             ROOT / 'src/xy_observables.py', ROOT / 'src/train_flow.py']
    paths += [ROOT / 'data' / f'xy_L32_T{str(t).removesuffix(".0").replace(".", "_")}_N1000.pt'
              for t in TEMPERATURES]
    return {str(p.relative_to(ROOT)).replace('\\', '/'): sha256(p) for p in paths}


@njit
def seed_wolff(seed):
    # Numba uses a separate RNG: seeding Python/NumPy alone is insufficient.
    np.random.seed(seed)


def generate_fm(config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(config['fm_seed'])
    torch.set_num_threads(4)
    model = PhysicsInformedDiT().to(device)
    model.load_state_dict(torch.load(ROOT / 'checkpoints/best_dit_flow.pth',
                                    map_location=device, weights_only=True))
    model.eval()
    # Generate all initial noise on CPU with a separate RNG; batch order is recorded.
    generator = torch.Generator().manual_seed(config['fm_seed'])
    noise = torch.randn(config['fm_samples'], 2, config['L'], config['L'], generator=generator)
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(noise), config['fm_batch_size']):
            x = noise[start:start+config['fm_batch_size']].to(device)
            for step in range(config['fm_steps']):
                t = torch.full((len(x),), step/config['fm_steps'], device=device)
                x = x + model(x, t)/config['fm_steps']
            outputs.append(x.cpu().numpy())
            print(f'FM: {min(start+len(x), len(noise))}/{len(noise)}', flush=True)
    raw = np.concatenate(outputs)
    if not np.isfinite(raw).all():
        raise ValueError('Non-finite FM endpoint')
    theta = np.arctan2(raw[:, 1].astype(float), raw[:, 0].astype(float))
    return raw, theta, noise.numpy(), str(device)


def generate_reference(config):
    samples, traces = [], []
    for chain in range(config['wolff_chains']):
        seed = config['wolff_seed'] + chain
        rng = np.random.default_rng(seed)
        theta = (np.zeros((config['L'], config['L'])) if chain % 2 == 0
                 else rng.uniform(0, 2*np.pi, (config['L'], config['L'])))
        seed_wolff(seed)
        trace = []
        for step in range(config['wolff_burn_in']):
            wolff_step_xy(theta, config['beta'])
            if (step+1) % config['wolff_interval'] == 0:
                o = observables(theta)
                trace.append([step+1, float(o['pairs']), float(o['energy'])])
        saved = []
        for _ in range(config['wolff_samples_per_chain']):
            for _ in range(config['wolff_interval']):
                wolff_step_xy(theta, config['beta'])
            saved.append(theta.copy())
        samples.append(saved)
        traces.append(trace)
        print(f'Wolff chain {chain+1}/{config["wolff_chains"]} complete', flush=True)
    return np.asarray(samples), np.asarray(traces)


def compute_report(arrays, config):
    fm, ref = observables(arrays['fm_theta']), observables(arrays['wolff_theta'])
    summaries = {}
    for k in ('pairs', 'energy'):
        summaries[k] = {
            'fm': mean_summary(fm[k], config['bootstrap_seed'], draws=config['bootstrap_draws']),
            'wolff': mean_summary(ref[k], config['bootstrap_seed']+1,
                                  block_size=config['ci_block_size'], draws=config['bootstrap_draws']),
            'wolff_checks': chain_checks(ref[k])}
    sweep = []
    for temperature in TEMPERATURES:
        tag = f'{temperature:g}'.replace('.', '_')
        tensor = torch.load(ROOT / 'data' / f'xy_L32_T{tag}_N1000.pt',
                            map_location='cpu', weights_only=True).double().numpy()
        obs = observables(np.arctan2(tensor[:, 1], tensor[:, 0]))
        sweep.append(dict(T=temperature, n=len(tensor), **{
            k: dict(mean=float(obs[k].mean()), sd=float(obs[k].std(ddof=1)))
            for k in ('pairs', 'energy')}))
    warnings = []
    for k in ('pairs', 'energy'):
        checks = summaries[k]['wolff_checks']
        if checks['split_rhat'] is None or checks['split_rhat'] > 1.05:
            warnings.append(f'{k}: split R-hat exceeds 1.05 or is undefined; extend reference chains.')
        if max(abs(v) for v in checks['lag1_by_chain']) > .2:
            warnings.append(f'{k}: appreciable residual lag-one correlation; review blocks.')
        sems = [checks['block_sem'][str(b)] for b in (8, 16, 32)]
        if max(sems) > 1.5*min(sems):
            warnings.append(f'{k}: block SEM not stable across 8/16/32; extend chains.')
    return dict(config=config, target_temperature=1/config['beta'], summaries=summaries,
                sweep=sweep, warnings=warnings,
                uncertainty='FM: sample bootstrap. Fresh Wolff: bootstrap of non-overlapping '
                            '16-sample blocks within chains, pooled across four chains. '
                            '95% intervals describe mean uncertainty, conditional on equilibration '
                            'and adequate block length. Sweep ribbon: sample SD, not a CI.',
                interpretation='This diagnoses the current checkpoint and Euler settings; '
                               'it is not an acceptance-rate estimate or a proof against all Euclidean FM.')


def write_observables(output, arrays):
    with (output/'observables.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ensemble', 'chain', 'sample', 'n_plus', 'n_minus', 'pairs', 'energy_per_bond'])
        for label, theta in [('FM', arrays['fm_theta'][None]), ('Wolff', arrays['wolff_theta'])]:
            obs = observables(theta)
            for c in range(len(theta)):
                for i in range(theta.shape[1]):
                    writer.writerow([label, c, i, *[obs[k][c, i] for k in
                                                   ('n_plus', 'n_minus', 'pairs', 'energy')]])


def load_run(output=DEFAULT_OUTPUT):
    output = Path(output)
    manifest = json.loads((output/'manifest.json').read_text(encoding='utf-8'))
    if manifest['config'] != CONFIG or manifest['sources'] != sources():
        raise RuntimeError('Saved run does not match current settings, source or checkpoint. '
                           'Preserve it and generate a fresh run with --output results/new-name.')
    for name, digest in manifest['artifacts'].items():
        if sha256(output/name) != digest:
            raise RuntimeError(f'Saved artifact changed: {name}')
    report = json.loads((output/'summary.json').read_text(encoding='utf-8'))
    with np.load(output/'samples.npz', allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files}
    return report, arrays


def ensure_run(output=DEFAULT_OUTPUT):
    output = Path(output)
    if (output/'manifest.json').exists():
        return load_run(output)
    if output.exists() and any(output.iterdir()):
        raise RuntimeError('Incomplete output directory; inspect it and use a new --output directory.')
    source_hashes = sources()
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    raw, theta, noise, device = generate_fm(CONFIG)
    wolff, burn_in = generate_reference(CONFIG)
    arrays = dict(fm_raw=raw, fm_theta=theta, fm_initial_noise=noise,
                  wolff_theta=wolff, wolff_burn_in=burn_in)
    np.savez_compressed(output/'samples.npz', **arrays)
    report = compute_report(arrays, CONFIG)
    (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    write_observables(output, arrays)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), config=CONFIG,
                    sources=source_hashes, runtime=dict(python=platform.python_version(),
                    numpy=np.__version__, torch=torch.__version__, device=device,
                    gpu=torch.cuda.get_device_name(0) if device=='cuda' else None,
                    elapsed_seconds=time.perf_counter()-started),
                    reference_initial_states=['ordered', 'random', 'ordered', 'random'],
                    reference_chain_seeds=[CONFIG['wolff_seed']+i for i in range(CONFIG['wolff_chains'])],
                    temperature_provenance='beta=1.12 from original sampler source; old training '
                                           'tensor/checkpoint lacks embedded temperature metadata.',
                    reproducibility='Saved initial noise and endpoints are authoritative. Floating-point '
                                    'results may differ across hardware or PyTorch versions.',
                    artifacts={p: sha256(output/p) for p in ('samples.npz', 'summary.json', 'observables.csv')})
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return report, arrays


def plot_comparison(report, arrays):
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, PercentFormatter
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout='constrained')
    blue, red, dark = '#277DA8', '#C64B38', '#263238'
    temperature = report['target_temperature']
    fm, ref = observables(arrays['fm_theta']), observables(arrays['wolff_theta'])
    for col, (key, label) in enumerate([('pairs', 'Vortex pair count'), ('energy', 'Energy per bond (J=1)')]):
        ax = axes[0, col]
        xs = np.array([r['T'] for r in report['sweep']])
        ys = np.array([r[key]['mean'] for r in report['sweep']])
        sd = np.array([r[key]['sd'] for r in report['sweep']])
        ax.plot(xs, ys, 'o-', color=blue, ms=4, label='Wolff temperature sweep: mean')
        ax.fill_between(xs, ys-sd, ys+sd, color=blue, alpha=.12, label='Sweep: mean +/- SD')
        for name, color, marker, legend in [('wolff', dark, 'o', 'Fresh Wolff: mean, 95% CI'),
                                          ('fm', red, 's', 'FM: mean, 95% CI')]:
            s = report['summaries'][key][name]
            ax.errorbar([temperature], [s['mean']], yerr=[[s['mean']-s['ci95'][0]],
                        [s['ci95'][1]-s['mean']]], fmt=marker, ms=7, color=color,
                        markerfacecolor='white' if name=='wolff' else color,
                        capsize=5, zorder=6, label=legend)
        ax.axvline(temperature, color=dark, ls=':', lw=1)
        ax.set(xlabel='Temperature T (J=kB=1)', ylabel='Mean '+label.lower(),
               title=('A  ' if col==0 else 'B  ')+label+' across temperatures')
        ax.legend(fontsize=8, loc='upper left' if col==0 else 'lower right')
        if key=='pairs':
            ax.set_ylim(bottom=0)
        ax = axes[1, col]
        a, b = fm[key].ravel(), ref[key].ravel()
        joined = np.concatenate([a, b])
        bins = (np.arange(np.floor(joined.min())-.5, np.ceil(joined.max())+1.5)
                if key=='pairs' else np.histogram_bin_edges(joined, bins=30))
        for values, color, name in [(b, blue, 'Fresh Wolff'), (a, red, 'FM')]:
            counts, _ = np.histogram(values, bins)
            ax.stairs(counts/len(values), bins, color=color, fill=True, alpha=.18)
            ax.stairs(counts/len(values), bins, color=color, label=f'{name} (n={len(values)})', linewidth=1.8)
            ax.axvline(values.mean(), color=color, ls='--', lw=1)
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.set(xlabel=label, ylabel='Fraction of configurations per bin',
               title=('C  ' if col==0 else 'D  ')+f'Distribution at T={temperature:.6f}')
        if key=='pairs':
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.legend(fontsize=9)
    for ax in axes.flat:
        ax.grid(alpha=.18)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Current Flow Matching baseline vs Wolff | L=32 | Euler 50 steps\n'
                 '128 saved FM samples; 1,024 fresh Wolff samples at matched temperature', fontsize=14)
    fig.supxlabel('Pair count = (N+ + N-)/2; no bound-pair identification. '
                  'Dashed histogram lines mark means. No acceptance rate inferred.', fontsize=9)
    return fig


def print_summary(report):
    print(f'Matched temperature: {report["target_temperature"]:.9f}, beta=1.12')
    for key in ('pairs', 'energy'):
        for label in ('fm', 'wolff'):
            s = report['summaries'][key][label]
            print(f'{label:5s} {key:6s}: mean={s["mean"]:.6f}, SD={s["sd"]:.6f}, '
                  f'95% CI=[{s["ci95"][0]:.6f}, {s["ci95"][1]:.6f}]')
    for warning in report['warnings']:
        print('REFERENCE CHECK:', warning)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    report, arrays = ensure_run(output)
    print_summary(report)
    fig = plot_comparison(report, arrays)
    for extension in ('png', 'pdf'):
        fig.savefig(output/f'comparison.{extension}', dpi=180)
    print(f'Saved comparison: {output}', flush=True)


if __name__ == '__main__':
    main()
