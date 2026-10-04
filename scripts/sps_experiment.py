"""Train, reference, and fixed-checkpoint step-size sweep for XY SPS."""
import argparse
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
import torch
from sps_xy import SPSConfig, XYPathSampler, independence_mh
from sps_diagnostics import measurements, chain_summary


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    return {p: digest(ROOT/p) for p in ('src/sps_xy.py', 'src/sps_diagnostics.py',
                                       'scripts/sps_experiment.py')}


def write_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    tmp.replace(path)


def fresh_directory(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f'Output is not empty: {path}; choose a new directory')
    path.mkdir(parents=True, exist_ok=True)
    return path


def device_setup(name):
    if name == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable; no silent CPU fallback')
    if name == 'cpu':
        torch.set_num_threads(2)
    return torch.device(name)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize()


def environment(device):
    return dict(python=platform.python_version(), torch=str(torch.__version__),
                numpy=np.__version__, platform=platform.platform(), device=str(device),
                gpu=torch.cuda.get_device_name() if device.type == 'cuda' else None)


def train(args):
    device = device_setup(args.device)
    c = SPSConfig(**json.loads(Path(args.config).read_text(encoding='utf-8')))
    if args.updates is not None:
        c.updates = args.updates
    if args.seed is not None:
        c.seed = args.seed
    c.validate()
    torch.manual_seed(c.seed)
    model = XYPathSampler(c).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=c.learning_rate)
    out = Path(args.out)
    start, elapsed, history = 0, 0.0, []
    if args.resume:
        # Only load checkpoints produced by this experiment, from a trusted source.
        checkpoint = torch.load(out/'checkpoint.pt', map_location=device, weights_only=False)
        if checkpoint['config'] != asdict(c) or checkpoint['sources'] != sources():
            raise ValueError('Resume config/source mismatch; preserve the old run and use a new name')
        if checkpoint['environment']['device'] != str(device):
            raise ValueError('Resume on the same device type used for training')
        model.load_state_dict(checkpoint['model'])
        opt.load_state_dict(checkpoint['optimizer'])
        start, elapsed = checkpoint['update'], checkpoint['training_seconds']
        history = checkpoint['history']
        torch.set_rng_state(checkpoint['rng_cpu'].cpu())
        if device.type == 'cuda':
            if checkpoint['rng_cuda'] is None:
                raise ValueError('Resume on the same device type used for training')
            torch.cuda.set_rng_state_all([x.cpu() for x in checkpoint['rng_cuda']])
    else:
        fresh_directory(out)
    write_json(out/'config.json', asdict(c))
    write_json(out/'environment.json', environment(device))
    print(f'DATA-FREE XY SPS | L={c.L} beta={c.beta} path_steps={c.train_steps} device={device}', flush=True)
    sync(device)
    begin = time.perf_counter()
    model.train()
    for update in range(start+1, c.updates+1):
        opt.zero_grad(set_to_none=True)
        batch = model.sample(c.batch_size, c.train_steps)
        loss = -batch['log_weight'].mean()/(c.L*c.L)
        if not torch.isfinite(loss):
            raise FloatingPointError(f'Nonfinite loss at update {update}')
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0, error_if_nonfinite=True)
        opt.step()
        if update == start+1 or update % 10 == 0:
            print(f'update={update} loss_per_site={loss.item():.6f} logw_sd={batch["log_weight"].std(unbiased=False).item():.3f}', flush=True)
        if update == start+1 or update % c.checkpoint_every == 0 or update == c.updates:
            sync(device)
            seconds = elapsed+time.perf_counter()-begin
            history.append(dict(update=update, loss_per_site=loss.item(),
                                log_weight_sd=batch['log_weight'].std(unbiased=False).item(),
                                energy_per_bond=batch['energy'].mean().item()/(2*c.L*c.L),
                                grad_norm=float(grad_norm), seconds=seconds))
            payload = dict(format_version=1, config=asdict(c), update=update,
                           model=model.state_dict(), optimizer=opt.state_dict(),
                           rng_cpu=torch.get_rng_state(),
                           rng_cuda=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
                           sources=sources(), history=history, training_seconds=seconds,
                           environment=environment(device), objective='E_q[-log_weight]/L^2; no target dataset')
            temp = out/'checkpoint.pt.tmp'
            torch.save(payload, temp)
            temp.replace(out/'checkpoint.pt')
            write_json(out/'training_log.json', history)
    print(f'Saved {out / "checkpoint.pt"}. Training loss is NOT a physical validation.', flush=True)


def reference(args):
    from numba import njit
    from sampler_wolff import wolff_step_xy
    if args.L < 3 or args.chains < 2 or args.draws < 4 or args.burn < 0 or args.interval < 1:
        raise ValueError('Invalid reference settings')
    if not math.isfinite(args.beta) or args.beta < 0 or not math.isfinite(args.J) or args.J <= 0:
        raise ValueError('Require finite beta>=0 and J>0')
    out = fresh_directory(args.out)
    seed_numba = njit(lambda seed: np.random.seed(seed))
    wolff_step_xy(np.zeros((args.L, args.L)), args.beta, args.J)  # exclude JIT warm-up
    theta = np.empty((args.chains, args.draws, args.L, args.L), dtype=np.float64)
    start = time.perf_counter()
    for k in range(args.chains):
        seed = args.seed+k
        seed_numba(seed)
        a = np.zeros((args.L, args.L)) if k % 2 == 0 else np.random.default_rng(seed).uniform(-np.pi, np.pi, (args.L, args.L))
        for _ in range(args.burn):
            wolff_step_xy(a, args.beta, args.J)
        for j in range(args.draws):
            for _ in range(args.interval):
                wolff_step_xy(a, args.beta, args.J)
            theta[k, j] = a
        print(f'Wolff reference chain {k+1}/{args.chains}', flush=True)
    duration = time.perf_counter()-start
    np.savez_compressed(out/'samples.npz', theta=theta)
    report = dict(L=args.L, beta=args.beta, J=args.J, chains=args.chains, draws=args.draws,
                  burn=args.burn, interval=args.interval, seed=args.seed, seconds=duration,
                  sampler_sha256=digest(ROOT/'src/sampler_wolff.py'),
                  samples_sha256=digest(out/'samples.npz'),
                  summaries={key: chain_summary(v) for key, v in measurements(theta, args.J).items()},
                  purpose='Independent validation only; NEVER used as SPS training data')
    write_json(out/'reference.json', report)


def draw_proposals(model, count, steps, batch_size, seed, device):
    generator = torch.Generator(device=device).manual_seed(seed)
    endpoints, weights = [], []
    sync(device)
    begin = time.perf_counter()
    with torch.no_grad():
        for first in range(0, count, batch_size):
            batch = model.sample(min(batch_size, count-first), steps, generator)
            endpoints.append(batch['theta'].cpu().numpy())
            weights.append(batch['log_weight'].cpu().numpy())
    sync(device)
    return np.concatenate(endpoints), np.concatenate(weights), time.perf_counter()-begin


def evaluate(args):
    device = device_setup(args.device)
    if args.chains < 2 or args.draws < 4 or args.burn < 0 or args.batch_size < 1:
        raise ValueError('Require chains>=2, draws>=4, burn>=0, batch_size>=1')
    if len(set(args.steps)) != len(args.steps) or min(args.steps) < 1:
        raise ValueError('Step counts must be positive and distinct')
    checkpoint_path = Path(args.checkpoint)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if ckpt['sources'] != sources():
        raise ValueError('Source files changed since training; do not mix implementations in experiment A')
    c = SPSConfig(**ckpt['config'])
    model = XYPathSampler(c).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    ref_dir = Path(args.reference)
    ref = json.loads((ref_dir/'reference.json').read_text(encoding='utf-8'))
    if any(ref[key] != getattr(c, key) for key in ('L', 'beta', 'J')):
        raise ValueError('Wolff reference L/beta/J does not match checkpoint')
    if digest(ref_dir/'samples.npz') != ref['samples_sha256']:
        raise ValueError('Reference samples hash mismatch')
    with np.load(ref_dir/'samples.npz') as archive:
        ref_summary = {k: chain_summary(v) for k, v in measurements(archive['theta'], c.J).items()}
    out = fresh_directory(args.out)
    report = dict(config=asdict(c), trained_update=ckpt['update'], checkpoint_sha256=digest(checkpoint_path),
                  sources=sources(), environment=environment(device), training_seconds=ckpt['training_seconds'],
                  reference=ref, reference_summaries=ref_summary,
                  settings=dict(steps=args.steps, chains=args.chains, draws=args.draws, burn=args.burn,
                                batch_size=args.batch_size, seed=args.seed), runs=[],
                  note='Fixed checkpoint. Only N/h change; horizon and all learned coefficient functions stay fixed. '
                       'Independent random streams per N. ESS/s includes proposal generation, host transfer, '
                       'initialization and burn-in, MH; excludes diagnostic plotting and training. '
                       'Approximate errors require equilibrium. Screening is not a proof of correctness.')
    count = args.chains*(args.burn+args.draws+1)
    # Separate untimed warm-up; re-seeding in draw_proposals makes it reproducible.
    with torch.no_grad():
        model.sample(min(args.batch_size, 4), 1)
    for n in sorted(args.steps):
        seed = args.seed+1009*n
        raw_theta, raw_w, generation_seconds = draw_proposals(model, count, n, args.batch_size, seed, device)
        raw_theta = raw_theta.reshape(args.chains, -1, c.L, c.L)
        raw_w = raw_w.reshape(args.chains, -1)
        logu = np.log(np.maximum(np.random.default_rng(seed+1).random(raw_w[:, 1:].shape), np.finfo(float).tiny))
        start = time.perf_counter()
        chain = independence_mh(raw_theta, raw_w, logu, args.burn)
        mh_seconds = time.perf_counter()-start
        seconds = generation_seconds+mh_seconds
        obs = measurements(chain['theta'], c.J)
        summaries = {k: chain_summary(v) for k, v in obs.items()}
        checks = {}
        for key, s in summaries.items():
            r = ref_summary[key]
            uncertainty = math.hypot(s['sem'], r['sem']) if s['sem'] is not None and r['sem'] is not None else None
            bias = s['mean']-r['mean']
            checks[key] = dict(bias=bias, combined_sem=uncertainty,
                               screening_pass=bool(s['screening_pass'] and r['screening_pass']
                                                   and uncertainty and abs(bias) <= 3*uncertainty))
        lw = raw_w.ravel()
        w = np.exp(lw-lw.max())
        acceptance = chain['accepted'].mean(1)
        alpha_summary = chain_summary(chain['alpha'])
        run = dict(steps=n, h=c.horizon/n, proposal_count=count,
                   generation_seconds=generation_seconds, mh_seconds=mh_seconds, sampling_seconds=seconds,
                   accepted_fraction=float(acceptance.mean()), acceptance_by_chain=acceptance.tolist(),
                   mean_alpha=alpha_summary,
                   log_weight_sd=float(lw.std(ddof=1)), importance_ess_fraction=float(w.sum()**2/(len(w)*(w*w).sum())),
                   summaries=summaries, physical_checks=checks,
                   ess_per_second={k: s['ess']/seconds if s['ess'] is not None else None for k, s in summaries.items()},
                   screening_pass=all(x['screening_pass'] for x in checks.values()))
        np.savez_compressed(out/f'steps_{n}.npz', proposal_theta=raw_theta, proposal_log_weight=raw_w,
                            log_uniform=logu, chain_theta=chain['theta'], chain_log_weight=chain['log_weight'],
                            accepted=chain['accepted'], alpha=chain['alpha'], **obs)
        run['samples_sha256'] = digest(out/f'steps_{n}.npz')
        report['runs'].append(run)
        write_json(out/'summary.json', report)
        print(f'N={n} acceptance={run["accepted_fraction"]:.4f} '
              f'energy_ESS/s={run["ess_per_second"]["energy"]} screening={run["screening_pass"]}', flush=True)
    write_csv_and_plot(out, report)


def write_csv_and_plot(out, report):
    rows = []
    for r in report['runs']:
        rows.append(dict(steps=r['steps'], h=r['h'], acceptance=r['accepted_fraction'],
                         mean_alpha=r['mean_alpha']['mean'], alpha_sem=r['mean_alpha']['sem'],
                         energy=r['summaries']['energy']['mean'], pairs=r['summaries']['pairs']['mean'],
                         energy_ess_per_second=r['ess_per_second']['energy'],
                         sampling_seconds=r['sampling_seconds'], screening_pass=r['screening_pass']))
    with (out/'summary.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ns = [r['steps'] for r in report['runs']]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    for r in report['runs']:
        axes[0].plot([r['steps']]*len(r['acceptance_by_chain']), r['acceptance_by_chain'], '.', color='0.7')
    axes[0].plot(ns, [r['accepted_fraction'] for r in report['runs']], 'o-')
    axes[0].set(ylabel='Accepted fraction', ylim=(0, 1), title='A: acceptance (dots: individual chains)')
    for key in ('energy', 'magnetization', 'pairs'):
        axes[1].plot(ns, [r['ess_per_second'][key] if r['ess_per_second'][key] is not None else np.nan
                         for r in report['runs']], 'o-', label=key)
    axes[1].set(ylabel='Estimated chain ESS / sampling second', title='B: useful samples versus cost')
    axes[1].legend()
    r = report['reference_summaries']['energy']
    axes[2].axhline(r['mean'], color='black', label='Wolff')
    if r['sem'] is not None:
        axes[2].axhspan(r['mean']-1.96*r['sem'], r['mean']+1.96*r['sem'], color='black', alpha=.12)
    for x in report['runs']:
        s = x['summaries']['energy']
        axes[2].errorbar(x['steps'], s['mean'], yerr=1.96*s['sem'] if s['sem'] is not None else None, fmt='o', color='tab:blue')
    axes[2].set(ylabel='Energy per bond', title='C: corrected physics (approx. 95% CI)')
    axes[2].legend()
    for ax in axes:
        ax.set_xlabel('Path steps N (fixed duration)')
        ax.set_xscale('log', base=2)
        ax.set_xticks(ns, [str(n) for n in ns])
        ax.grid(alpha=.2)
    status = 'SCREENING PASSED (not proof of mixing)' if all(r['screening_pass'] for r in report['runs']) else 'DIAGNOSTICS INCOMPLETE / FAILED: do not infer an efficiency gain'
    fig.suptitle(f'XY SPS | L={report["config"]["L"]} | fixed checkpoint | {status}', fontsize=10)
    fig.tight_layout()
    fig.savefig(out/'comparison.png', dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    p = subs.add_parser('train')
    p.add_argument('--config', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--updates', type=int)
    p.add_argument('--seed', type=int)
    p.add_argument('--resume', action='store_true')
    p.set_defaults(func=train)
    p = subs.add_parser('reference')
    p.add_argument('--L', type=int, required=True)
    p.add_argument('--beta', type=float, default=1.12)
    p.add_argument('--J', type=float, default=1.0)
    p.add_argument('--out', required=True)
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--draws', type=int, default=4096)
    p.add_argument('--burn', type=int, default=2000)
    p.add_argument('--interval', type=int, default=20)
    p.add_argument('--seed', type=int, default=7000)
    p.set_defaults(func=reference)
    p = subs.add_parser('evaluate')
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--reference', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--device', choices=['cpu', 'cuda'], required=True)
    p.add_argument('--steps', type=int, nargs='+', default=[32, 64, 128])
    p.add_argument('--chains', type=int, default=4)
    p.add_argument('--draws', type=int, default=2048)
    p.add_argument('--burn', type=int, default=1024)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--seed', type=int, default=12000)
    p.set_defaults(func=evaluate)
    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
