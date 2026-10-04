"""Paired 5000-to-10000-update continuation; original SPS sources stay fixed.

Only a NEW checkpoint copy's planned update limit changes. The existing train
entry point restores the complete model, Adam state, and CPU/CUDA RNG states.
--verify-only checks pinned inputs with the standard library, without training.
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


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False,
                                    allow_nan=False)+'\n', encoding='utf-8')


def preflight(root, protocol, seed):
    spec = protocol['baselines'][str(seed)]
    baseline = root/spec['directory']
    changed = [name for name, expected in protocol['source_sha256'].items()
               if not (root/name).is_file() or digest(root/name) != expected]
    if changed:
        raise ValueError(f'Original source/configuration changed: {changed}')
    if digest(baseline/'train/checkpoint.pt') != spec['checkpoint_sha256']:
        raise ValueError('Wrong or changed 5000-update checkpoint')
    summary_path = baseline/spec['evaluation_directory']/'summary.json'
    if digest(summary_path) != spec['summary_sha256']:
        raise ValueError('Original long evaluation changed')
    summary = read_json(summary_path)
    expected_config = dict(protocol['baseline_config'], seed=seed)
    expected_sources = {k: protocol['source_sha256'][k] for k in summary['sources']}
    if (summary['config'] != expected_config
            or summary['trained_update'] != protocol['start_update']
            or summary['checkpoint_sha256'] != spec['checkpoint_sha256']
            or summary['sources'] != expected_sources
            or summary['settings'] != protocol['evaluation_settings']
            or summary['reference'] != protocol['reference_metadata']):
        raise ValueError('Baseline does not match the paired continuation protocol')
    ref = read_json(baseline/'reference/reference.json')
    if (ref != protocol['reference_metadata']
            or digest(baseline/'reference/samples.npz') != ref['samples_sha256']):
        raise ValueError('Independent Wolff validation reference changed')
    return baseline, expected_config, expected_sources


def stage_checkpoint(source, destination, expected_config, expected_sources, target_updates):
    """Create a new continuation checkpoint, changing only config.updates.

This is needed because the original interrupted-run resume guard requires an
exact configuration match. No original checkpoint or source file is modified.
"""
    import torch
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError('Continuation checkpoint already exists; use a fresh output')
    if target_updates <= expected_config['updates']:
        raise ValueError('Target updates must exceed the original training budget')
    checkpoint = torch.load(source, map_location='cpu', weights_only=False)
    if (checkpoint['config'] != expected_config
            or checkpoint['update'] != expected_config['updates']
            or checkpoint['sources'] != expected_sources):
        raise ValueError('Checkpoint contents do not match the original completed run')
    checkpoint['config'] = dict(checkpoint['config'], updates=target_updates)
    torch.save(checkpoint, destination)
    return checkpoint['config']


def command(*arguments):
    subprocess.run([sys.executable, '-u', *map(str, arguments)], cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(11, 22, 33), required=True)
    parser.add_argument('--protocol', type=Path,
                        default=ROOT/'configs/sps_a_continuation.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    if protocol['start_update'] != 5000 or protocol['target_updates'] != 10000:
        raise ValueError('This bounded experiment is pinned to 5000 -> 10000 updates')
    baseline, original_config, core_sources = preflight(ROOT, protocol, args.seed)
    print(f'PREFLIGHT PASS: seed={args.seed}; verified original model and matched evaluation.', flush=True)
    if args.verify_only:
        return
    if args.out is None:
        parser.error('--out is required for a server run')
    output = args.out.resolve()
    baseline = baseline.resolve()
    if output == baseline or baseline in output.parents or output in baseline.parents:
        raise ValueError('Continuation output must be separate from the original run')
    if output.exists():
        raise ValueError(f'Output already exists: {output}; use a fresh job directory')

    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Server CUDA required; no local or CPU training fallback')
    actual_environment = dict(python='.'.join(map(str, sys.version_info[:3])),
                              torch=str(torch.__version__), numpy=np.__version__,
                              gpu=torch.cuda.get_device_name(0), device='cuda')
    expected_environment = {k: protocol['baseline_environment'][k] for k in actual_environment}
    if actual_environment != expected_environment:
        raise ValueError(f'Environment changed: {actual_environment}; expected {expected_environment}')
    command('-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_sps_xy.py', '-v')
    command('-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_sps_continuation.py', '-v')

    (output/'train').mkdir(parents=True)
    write_json(output/'continuation_protocol.json', protocol)
    config = stage_checkpoint(baseline/'train/checkpoint.pt', output/'train/checkpoint.pt',
                              original_config, core_sources, protocol['target_updates'])
    write_json(output/'continuation_config.json', config)
    shutil.copyfile(baseline/protocol['baselines'][str(args.seed)]['evaluation_directory']/'summary.json',
                    output/'baseline_summary.json')
    write_json(output/'continuation_provenance.json', dict(
        training_seed=args.seed, job_id=os.environ.get('SLURM_JOB_ID'),
        baseline=str(baseline), baseline_checkpoint_sha256=digest(baseline/'train/checkpoint.pt'),
        staged_checkpoint_sha256=digest(output/'train/checkpoint.pt'),
        protocol_sha256=digest(args.protocol), runner_sha256=digest(__file__),
        slurm_script_sha256=digest(ROOT/'scripts/slurm/sps_a_continue.slurm'),
        environment=actual_environment, start_update=5000, target_update=10000,
        note='Only config.updates changed in a new checkpoint copy; all model/Adam/RNG states retained. '
             'Original runs untouched. Training uses no Wolff data. Evaluation uses common RNG seeds.'))
    entry = ROOT/'scripts/sps_experiment.py'
    command(entry, 'train', '--config', output/'continuation_config.json',
            '--out', output/'train', '--device', 'cuda', '--resume')
    # Validation data are copied only after parameter training has finished.
    shutil.copytree(baseline/'reference', output/'reference')
    settings = protocol['evaluation_settings']
    command(entry, 'evaluate', '--checkpoint', output/'train/checkpoint.pt',
            '--reference', output/'reference', '--out', output/'eval_long',
            '--device', 'cuda', '--steps', *settings['steps'],
            '--chains', settings['chains'], '--draws', settings['draws'],
            '--burn', settings['burn'], '--batch-size', settings['batch_size'],
            '--seed', settings['seed'])
    result = read_json(output/'eval_long/summary.json')
    if (result['config'] != config or result['trained_update'] != 10000
            or result['settings'] != settings or result['sources'] != core_sources
            or result['reference'] != protocol['reference_metadata']
            or result['checkpoint_sha256'] != digest(output/'train/checkpoint.pt')):
        raise ValueError('Completed run does not match the continuation protocol')
    preflight(ROOT, protocol, args.seed)
    baseline_seconds = read_json(output/'baseline_summary.json')['training_seconds']
    write_json(output/'completion.json', dict(
        software_complete=True, training_seed=args.seed, trained_update=10000,
        baseline_training_seconds=baseline_seconds,
        total_training_seconds=result['training_seconds'],
        added_training_seconds=result['training_seconds']-baseline_seconds,
        evaluation_summary_sha256=digest(output/'eval_long/summary.json'),
        screening_by_steps={r['steps']: r['screening_pass'] for r in result['runs']},
        note='Paired within-seed comparison, not a new independent training seed. '
             'Physical screening failure is a scientific result, not software failure.'))
    print(f'CONTINUATION COMPLETE: seed={args.seed}; update=10000; output={output}', flush=True)


if __name__ == '__main__':
    main()
