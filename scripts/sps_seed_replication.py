"""Pinned seed-only replication of job 25906, with job 25955 evaluation settings.

--verify-only uses the standard library and never trains or samples.
Actual runs use the existing experiment entry point and CUDA on the server.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def preflight(root, baseline, protocol, seed):
    if seed not in protocol['new_training_seeds']:
        raise ValueError('Use the preselected training seeds 22 or 33')
    changed = [name for name, expected in protocol['source_sha256'].items()
               if not (root/name).is_file() or digest(root/name) != expected]
    if changed:
        raise ValueError(f'Baseline source/configuration mismatch: {changed}')
    checkpoint = baseline/'train/checkpoint.pt'
    if digest(checkpoint) != protocol['baseline_checkpoint_sha256']:
        raise ValueError('Wrong baseline checkpoint')
    ref = read_json(baseline/'reference/reference.json')
    if ref != protocol['reference_metadata']:
        raise ValueError('Wolff reference metadata changed')
    if digest(baseline/'reference/samples.npz') != ref['samples_sha256']:
        raise ValueError('Wolff reference samples changed')
    prior = read_json(baseline/'eval_long_25955/summary.json')
    if (prior['config'] != protocol['baseline_config']
            or prior['settings'] != protocol['evaluation_settings']
            or prior['checkpoint_sha256'] != protocol['baseline_checkpoint_sha256']
            or prior['reference'] != ref):
        raise ValueError('Baseline long evaluation does not match the pinned protocol')
    if any(prior['sources'][name] != protocol['source_sha256'][name]
           for name in prior['sources']):
        raise ValueError('Pinned source hashes do not match the baseline evaluation')
    return ref


def command(*arguments):
    subprocess.run([sys.executable, '-u', *map(str, arguments)], cwd=ROOT, check=True)


def write_json(path, payload):
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--baseline', type=Path, default=ROOT/'results/sps_a/L4_seed11_full_25906')
    p.add_argument('--protocol', type=Path, default=ROOT/'configs/sps_a_replication.json')
    p.add_argument('--out', type=Path)
    p.add_argument('--verify-only', action='store_true')
    args = p.parse_args()
    protocol = read_json(args.protocol)
    preflight(ROOT, args.baseline, protocol, args.seed)
    print(f'PREFLIGHT PASS: seed={args.seed}, fixed baseline implementation and reference.', flush=True)
    if args.verify_only:
        return
    if args.out is None:
        p.error('--out is required for a server run')
    if args.out.exists():
        raise ValueError(f'Refusing to reuse output: {args.out}; choose a fresh job directory')

    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; no CPU training fallback')
    actual_environment = dict(python='.'.join(map(str, sys.version_info[:3])),
                              torch=str(torch.__version__), numpy=np.__version__,
                              gpu=torch.cuda.get_device_name(0))
    expected_environment = {key: protocol['baseline_environment'][key] for key in actual_environment}
    if actual_environment != expected_environment:
        raise ValueError(f'Environment changed: {actual_environment}; expected {expected_environment}')
    args.out.mkdir(parents=True)
    write_json(args.out/'replication_protocol.json', protocol)
    write_json(args.out/'replication_provenance.json', dict(training_seed=args.seed,
               baseline=str(args.baseline.resolve()), baseline_checkpoint_sha256=protocol['baseline_checkpoint_sha256'],
               protocol_sha256=digest(args.protocol), runner_sha256=digest(__file__),
               slurm_script_sha256=digest(ROOT/'scripts/slurm/sps_a_replicate.slurm'),
               job_id=os.environ.get('SLURM_JOB_ID'), environment=actual_environment,
               note='Fresh initialization; no resume or baseline model loading. Common evaluation RNG seeds across trained models.'))

    command('-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_sps_xy.py', '-v')
    entry = ROOT/'scripts/sps_experiment.py'
    command(entry, 'train', '--config', ROOT/'configs/sps_a_l4.json',
            '--out', args.out/'train', '--device', 'cuda', '--seed', args.seed)
    # Copy the independently generated reference only AFTER training.
    shutil.copytree(args.baseline/'reference', args.out/'reference')
    settings = protocol['evaluation_settings']
    command(entry, 'evaluate', '--checkpoint', args.out/'train/checkpoint.pt',
            '--reference', args.out/'reference', '--out', args.out/'eval_long',
            '--device', 'cuda', '--steps', *settings['steps'],
            '--chains', settings['chains'], '--draws', settings['draws'], '--burn', settings['burn'],
            '--batch-size', settings['batch_size'], '--seed', settings['seed'])
    result = read_json(args.out/'eval_long/summary.json')
    expected_config = dict(protocol['baseline_config'], seed=args.seed)
    expected_core_sources = {name: protocol['source_sha256'][name] for name in result['sources']}
    if (result['config'] != expected_config or result['settings'] != settings
            or result['trained_update'] != expected_config['updates']
            or result['sources'] != expected_core_sources
            or result['reference'] != protocol['reference_metadata']):
        raise ValueError('Completed run did not follow the seed-only replication protocol')
    write_json(args.out/'completion.json', dict(software_complete=True, training_seed=args.seed,
               evaluation_summary_sha256=digest(args.out/'eval_long/summary.json'),
               screening_by_steps={r['steps']: r['screening_pass'] for r in result['runs']},
               note='Physical screening failures are research results; they do not imply a failed software job.'))
    print(f'REPLICATION COMPLETE: seed={args.seed}; output={args.out}', flush=True)


if __name__ == '__main__':
    main()
