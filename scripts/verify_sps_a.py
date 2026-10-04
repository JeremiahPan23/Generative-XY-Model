#!/usr/bin/env python3
"""Read-only verification against the original experiment-A server bundle.

Upload this file alone if BUNDLE_MANIFEST.json is missing. Run from the
project root, or pass --root /path/to/Generative-XY-Model. Standard library only.
"""
import argparse
import hashlib
from pathlib import Path
import socket
import sys

# Original bundle_manifest.json values, not hashes regenerated from server files.
EXPECTED = {
    'src/sps_xy.py': '89bf6b0f0480dbff25df79fb2617c670b834f90ed7d1ec4d7baf973abd56ea66',
    'src/sps_diagnostics.py': 'a001499fcdf49a1c5ca8dae35e4b780b7b09fa69b5af3358d39f872eed3176de',
    'src/sampler_wolff.py': 'aa939108a5595a4a963974ecad03d1bd98589baecfec728bde8906db5501fe92',
    'src/xy_observables.py': '37969eef23eb05d25330ef47c0064f0e030a2e64fedc206da988e49e20ebd779',
    'scripts/sps_experiment.py': 'b37b34d9d5403e67df9b42e589962f7c770831744f6de8e6227f738caa5e0485',
    'tests/test_sps_xy.py': '797b16e5e6fa59b7ad6fcf48975c079e962e0576c20b339fd52be4ea94e60242',
    'configs/sps_a_l4.json': '64f1079d0fd81cc9915dc9eb36712c6901fac172693d04abd410cb69f3875d6f',
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.root.resolve()
    print(f'Host: {socket.gethostname()}')
    print(f'Project: {root}')
    problems = []
    for relative, expected in EXPECTED.items():
        try:
            observed = hashlib.sha256((root/relative).read_bytes()).hexdigest()
        except OSError as exc:
            print(f'MISSING / UNREADABLE: {relative} ({exc})')
            problems.append(relative)
            continue
        if observed != expected:
            print(f'DIFFERENT: {relative}\n  expected: {expected}\n  observed: {observed}')
            problems.append(relative)
        else:
            print(f'OK: {relative}')
    if problems:
        print('CHECK NEEDED: share this output before submitting the full run.')
        print('A mismatch may include line-ending changes; it is not by itself proof of a physics-code change.')
        return 1
    print('PASS: all 7 core files match the original server bundle.')
    print('This verifies only the files at the host and path printed above.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
