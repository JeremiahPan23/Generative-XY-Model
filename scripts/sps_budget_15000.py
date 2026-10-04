"""Final training-budget point: continue the three verified 10000-update models.

Uses the unchanged checkpoint-staging helper, original SPS training/evaluation
entry point, and the same evaluation budget. No automatic further extension.
"""
import argparse
import os
from pathlib import Path
import shutil
import sys

from sps_training_continuation import ROOT, command, digest, preflight, read_json, stage_checkpoint, write_json


def preflight_15000(root, protocol, seed):
    if (protocol['start_update'] != 10000 or protocol['target_updates'] != 15000
            or protocol['baseline_config']['updates'] != 10000
            or protocol['training_seeds'] != [11, 22, 33]):
        raise ValueError('Final budget experiment requires 10000 -> 15000 and seeds 11/22/33')
    try:
        baseline, config, core_sources = preflight(root, protocol, seed)
    except ValueError as error:
        raise ValueError(str(error).replace('5000-update', '10000-update')) from error
    completion = read_json(baseline/'completion.json')
    summary_path = baseline/protocol['baselines'][str(seed)]['evaluation_directory']/'summary.json'
    if (completion['software_complete'] is not True
            or completion['trained_update'] != 10000 or completion['training_seed'] != seed
            or completion['evaluation_summary_sha256'] != digest(summary_path)):
        raise ValueError('The 10000-update baseline is not a verified completed run')
    return baseline, config, core_sources


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=(11, 22, 33), required=True)
    parser.add_argument('--protocol', type=Path, default=ROOT/'configs/sps_a_15000.json')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    protocol = read_json(args.protocol)
    baseline, original_config, core_sources = preflight_15000(ROOT, protocol, args.seed)
    print(f'PREFLIGHT PASS: seed={args.seed}; verified 10000-update model; final target=15000.', flush=True)
    if args.verify_only:
        return
    if args.out is None:
        parser.error('--out is required for a server run')
    baseline, output = baseline.resolve(), args.out.resolve()
    if output == baseline or baseline in output.parents or output in baseline.parents:
        raise ValueError('Output must be separate from the original run')
    if output.exists():
        raise ValueError(f'Output already exists: {output}; choose a fresh job directory')

    import numpy as np
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Server CUDA required; no CPU training fallback')
    actual_environment = dict(python='.'.join(map(str, sys.version_info[:3])),
                              torch=str(torch.__version__), numpy=np.__version__,
                              gpu=torch.cuda.get_device_name(0), device='cuda')
    expected_environment = {k: protocol['baseline_environment'][k] for k in actual_environment}
    if actual_environment != expected_environment:
        raise ValueError(f'Environment changed: {actual_environment}; expected {expected_environment}')
    for test in ('test_sps_xy.py', 'test_sps_continuation.py', 'test_sps_budget_15000.py'):
        command('-m', 'unittest', 'discover', '-s', 'tests', '-p', test, '-v')

    (output/'train').mkdir(parents=True)
    write_json(output/'continuation_protocol.json', protocol)
    config = stage_checkpoint(baseline/'train/checkpoint.pt', output/'train/checkpoint.pt',
                              original_config, core_sources, 15000)
    write_json(output/'continuation_config.json', config)
    shutil.copyfile(baseline/'eval_long/summary.json', output/'baseline_summary.json')
    write_json(output/'continuation_provenance.json', dict(
        training_seed=args.seed, job_id=os.environ.get('SLURM_JOB_ID'), baseline=str(baseline),
        baseline_checkpoint_sha256=digest(baseline/'train/checkpoint.pt'),
        staged_checkpoint_sha256=digest(output/'train/checkpoint.pt'),
        protocol_sha256=digest(args.protocol), runner_sha256=digest(__file__),
        staging_helper_sha256=digest(ROOT/'scripts/sps_training_continuation.py'),
        slurm_script_sha256=digest(ROOT/'scripts/slurm/sps_a_15000.slurm'),
        environment=actual_environment, start_update=10000, target_update=15000,
        note='Final third checkpoint in the 5000/10000/15000 budget study. '
             'Only config.updates changes in a fresh copy; complete model/Adam/RNG/history retained. '
             'No automatic further training; original runs untouched. Wolff only validates.'))
    entry = ROOT/'scripts/sps_experiment.py'
    command(entry, 'train', '--config', output/'continuation_config.json',
            '--out', output/'train', '--device', 'cuda', '--resume')
    shutil.copytree(baseline/'reference', output/'reference')
    settings = protocol['evaluation_settings']
    command(entry, 'evaluate', '--checkpoint', output/'train/checkpoint.pt',
            '--reference', output/'reference', '--out', output/'eval_long',
            '--device', 'cuda', '--steps', *settings['steps'],
            '--chains', settings['chains'], '--draws', settings['draws'],
            '--burn', settings['burn'], '--batch-size', settings['batch_size'], '--seed', settings['seed'])
    result = read_json(output/'eval_long/summary.json')
    if (result['config'] != config or result['trained_update'] != 15000
            or result['settings'] != settings or result['sources'] != core_sources
            or result['reference'] != protocol['reference_metadata']
            or result['checkpoint_sha256'] != digest(output/'train/checkpoint.pt')):
        raise ValueError('Completed run does not match the final-budget protocol')
    preflight_15000(ROOT, protocol, args.seed)
    baseline_seconds = read_json(output/'baseline_summary.json')['training_seconds']
    write_json(output/'completion.json', dict(
        software_complete=True, training_seed=args.seed, trained_update=15000,
        baseline_training_seconds=baseline_seconds, total_training_seconds=result['training_seconds'],
        added_training_seconds=result['training_seconds']-baseline_seconds,
        evaluation_summary_sha256=digest(output/'eval_long/summary.json'),
        screening_by_steps={r['steps']: r['screening_pass'] for r in result['runs']},
        final_training_budget_point=True,
        note='Paired third checkpoint, not a new independent training seed. '
             'Screening failure is a research result; no automatic retry or budget extension.'))
    print(f'CONTINUATION COMPLETE: seed={args.seed}; update=15000; output={output}', flush=True)


if __name__ == '__main__':
    main()
