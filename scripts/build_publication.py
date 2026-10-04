"""Copy selected public artifacts, or check their integrity on a clean clone.

Packaging never trains, samples, changes scientific code or rewrites raw data.
Rebuilding requires the original local archive. --check needs only published
files and code. All manifest paths are repository-relative.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = {
    'results/training_budget.csv': 'results/sps_a_training_budget_three_points/comparison.csv',
    'results/training_budget.json': 'results/sps_a_training_budget_three_points/comparison.json',
    'results/generation_steps.csv': 'results/sps_a_seed_replication/comparison.csv',
    'results/generation_steps.json': 'results/sps_a_seed_replication/comparison.json',
    'results/sticking_mechanism.json': 'results/sps_sticking_mechanism/mechanism.json',
    'results/training_budget_import_manifest.json': 'results/sps_a_training_budget_three_points/import_manifest.json',
    'results/generation_steps_provenance.json': 'results/sps_a_seed_replication/provenance_checks.json',
    'results/ess_sensitivity.json': 'results/sps_a_seed_replication/ess_sensitivity.json',
    'results/legacy_fm_wolff.json': 'results/fm_wolff_baseline/summary.json',
    'results/legacy_fm_wolff_manifest.json': 'results/fm_wolff_baseline/manifest.json',
    'figures/training_budget.png': 'results/sps_a_training_budget_three_points/three_budget_comparison.png',
    'figures/training_budget.pdf': 'results/sps_a_training_budget_three_points/three_budget_comparison.pdf',
    'figures/generation_steps.png': 'results/sps_a_seed_replication/three_seed_comparison.png',
    'figures/sticking_mechanism.png': 'results/sps_sticking_mechanism/mechanism_explained.png',
    'figures/sticking_mechanism.pdf': 'results/sps_sticking_mechanism/mechanism_explained.pdf',
    'figures/legacy_fm_wolff.png': 'results/fm_wolff_baseline/comparison.png',
    'figures/legacy_fm_wolff.pdf': 'results/fm_wolff_baseline/comparison.pdf',
    'figures/reference_multitemperature.png': 'figures/multitemp_example_grid.png',
    'figures/reference_statistics.png': 'figures/multitemp_vortex_energy.png',
}


def fingerprint(path):
    payload = path.read_bytes()
    return dict(sha256=hashlib.sha256(payload).hexdigest(), bytes=len(payload))


def build(root):
    out = root/'publication'
    missing = [name for name in ARTIFACTS.values() if not (root/name).is_file()]
    if missing:
        raise FileNotFoundError('Rebuilding needs the local archive: '+', '.join(missing))
    artifacts = {}
    for target, original in ARTIFACTS.items():
        source, destination = root/original, out/target
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != source.read_bytes():
            raise ValueError(f'Refusing to overwrite a different public artifact: {target}')
        shutil.copyfile(source, destination)
        artifacts[target] = dict(source=original, **fingerprint(source))
    implementation = {}
    for folder, extension in [('src', '*.py'), ('scripts', '*.py'),
                              ('scripts/slurm', '*.slurm'), ('tests', '*.py'),
                              ('configs', '*.json')]:
        for path in sorted((root/folder).glob(extension)):
            if path.name == 'build_publication.py':
                continue
            implementation[path.relative_to(root).as_posix()] = fingerprint(path)
    manifest = dict(format_version=1, artifacts=artifacts,
        implementation_files=implementation,
        note='Byte-identical copies of existing saved analyses, not new measurements. '
             'Source paths identify the omitted local archive. Integrity checks '
             'do not reproduce raw-data audits or full unrecorded paths.')
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    check(root)


def check(root):
    manifest = json.loads((root/'publication/manifest.json').read_text(encoding='utf-8'))
    failures = []
    for name, expected in manifest['artifacts'].items():
        path = root/'publication'/name
        if not path.is_file() or fingerprint(path) != {k: expected[k] for k in ('sha256', 'bytes')}:
            failures.append('publication/'+name)
    for name, expected in manifest['implementation_files'].items():
        path = root/name
        if not path.is_file() or fingerprint(path) != expected:
            failures.append(name)
    if failures:
        raise ValueError('Published integrity mismatch: '+', '.join(failures))
    print(f"PASS: {len(manifest['artifacts'])} public artifacts and "
          f"{len(manifest['implementation_files'])} implementation files match their manifest.")
    print('Integrity verification only; omitted raw experiments are not re-audited.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.check:
        check(args.root.resolve())
    else:
        build(args.root.resolve())


if __name__ == '__main__':
    main()
