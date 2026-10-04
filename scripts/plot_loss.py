"""Plot the most recent Flow Matching training history.

This script is intentionally data-driven now: it does NOT contain a
hard-coded training log.  It looks for the newest of:

    logs/training_log.json   (preferred on equal timestamps)
    logs/training_log.csv
    logs/training_log.txt / logs/training.log

and renders ``figures/loss_curve.png`` + ``figures/loss_curve.pdf``.

You can also point it at any stdout log from the cluster:

    python scripts/plot_loss.py --log logs/slurm-123456.out
"""

import argparse
import csv
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_PREFIX = PROJECT_ROOT / 'figures' / 'loss_curve'
LOG_CANDIDATES = [
    PROJECT_ROOT / 'logs' / 'training_log.json',
    PROJECT_ROOT / 'logs' / 'training_log.csv',
    PROJECT_ROOT / 'logs' / 'training_log.txt',
    PROJECT_ROOT / 'logs' / 'training.log',
]

# Matches the fixed stdout format printed by src/train_flow.py.
LOG_LINE_RE = re.compile(
    r'Epoch\s+(\d+)/\d+\s+\|\s+Train Loss:\s+([0-9.]+)\s+\|\s+Val Loss:\s+([0-9.]+)'
)


def _row_get(mapping, *keys):
    """Return the first key present in a dict-like object (case-insensitive)."""
    lower = {str(k).lower(): v for k, v in mapping.items()}
    for key in keys:
        if key.lower() in lower:
            return lower[key.lower()]
    return None


def parse_text_log(text):
    """Parse a plain-text training stdout log into (epoch, train, val) records."""
    records = []
    for line in text.strip().splitlines():
        match = LOG_LINE_RE.search(line)
        if match:
            records.append({
                'epoch': int(match.group(1)),
                'train_loss': float(match.group(2)),
                'val_loss': float(match.group(3)),
            })
    return records


def load_json_log(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Support both a bare list and {"history": [...]} wrappers.
    if isinstance(data, dict):
        for key in ('history', 'records', 'epochs'):
            if isinstance(data.get(key), list):
                data = data[key]
                break

    if not isinstance(data, list):
        return []

    records = []
    for i, item in enumerate(data, start=1):
        epoch = int(_row_get(item, 'epoch') or i)
        train_loss = float(_row_get(item, 'train_loss', 'train') or 0.0)
        val_loss = float(_row_get(item, 'val_loss', 'val', 'valid_loss') or 0.0)
        records.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_loss,
        })
    return records


def load_csv_log(path):
    records = []
    with open(path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader, start=1):
            epoch = int(_row_get(row, 'epoch') or i)
            train_loss = float(_row_get(row, 'train_loss', 'train') or 0.0)
            val_loss = float(_row_get(row, 'val_loss', 'val', 'valid_loss') or 0.0)
            records.append({
                'epoch': epoch,
                'train_loss': train_loss,
                'val_loss': val_loss,
            })
    return records


def load_log(path):
    """Load history from JSON, CSV, or plain text based on file extension."""
    suffix = path.suffix.lower()
    if suffix == '.json':
        return load_json_log(path)
    if suffix == '.csv':
        return load_csv_log(path)
    return parse_text_log(path.read_text(encoding='utf-8'))


def find_latest_log():
    """Return the newest available training log in the project's logs folder."""
    candidates = [Path(p) for p in LOG_CANDIDATES if Path(p).is_file()]
    candidates = [p for p in candidates if p.stat().st_size > 0]
    if not candidates:
        return None
    # On equal mtimes, list order prefers JSON over CSV over plain text.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def plot_and_save(records, source_label, out_prefix=DEFAULT_OUT_PREFIX):
    """Render train/val curves and write PNG + PDF files."""
    out_prefix = Path(out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    epochs = [r['epoch'] for r in records]
    train_losses = [r['train_loss'] for r in records]
    val_losses = [r['val_loss'] for r in records]

    best_idx = min(range(len(val_losses)), key=val_losses.__getitem__)
    best_epoch = epochs[best_idx]
    best_val_loss = val_losses[best_idx]

    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 14,
        'legend.fontsize': 12,
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'figure.figsize': (8, 5),
        'figure.dpi': 300,
    })

    fig, ax = plt.subplots()
    ax.plot(epochs, train_losses, label='Training Loss',
            color='#1f77b4', linewidth=2, linestyle='-')
    ax.plot(epochs, val_losses, label='Validation Loss',
            color='#ff7f0e', linewidth=2, linestyle='--')

    ax.scatter(best_epoch, best_val_loss, color='red', s=100,
               zorder=5, marker='*', label='Best Model')

    # Keep the annotation inside the axes for short runs or early best epochs.
    x_min, x_max = min(epochs), max(epochs)
    x_span = x_max - x_min
    val_span = (max(val_losses) - min(val_losses)) or 1.0
    if best_epoch - 25 >= x_min:
        text_x = best_epoch - 25
        text_ha = 'right'
    else:
        text_x = x_min + 0.5 * x_span
        text_ha = 'center'
    text_y = best_val_loss + 0.12 * val_span
    ax.annotate(
        f'Best Model\nEpoch {best_epoch}, Loss: {best_val_loss:.4f}',
        xy=(best_epoch, best_val_loss),
        xytext=(text_x, text_y),
        arrowprops=dict(facecolor='black', arrowstyle='->', lw=1.2),
        fontsize=11,
        horizontalalignment=text_ha,
        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='gray', alpha=0.9),
    )

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (MSE)')
    ax.set_title('Physics-Informed DiT Training Convergence (L=32)')
    ax.grid(True, linestyle=':', alpha=0.7)
    ax.legend(loc='upper right', frameon=True, edgecolor='black')

    plt.tight_layout()
    plt.savefig(f'{out_prefix}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{out_prefix}.pdf', dpi=300, bbox_inches='tight')
    plt.close(fig)

    print(f"Plotting completed! '{out_prefix}.png' and '{out_prefix}.pdf' "
          f"have been generated.")
    print(f'Source: {source_label}')
    print(f'The best model appears at Epoch {best_epoch}, '
          f'with a validation Loss of {best_val_loss:.6f}.')


def main():
    parser = argparse.ArgumentParser(
        description='Plot the most recent Flow Matching training history.'
    )
    parser.add_argument(
        '--log',
        default=None,
        help='Optional path to a JSON, CSV, or stdout text training log.',
    )
    parser.add_argument(
        '--out-prefix',
        default=str(DEFAULT_OUT_PREFIX),
        help='Output filename prefix (default: project figures/loss_curve).',
    )
    args = parser.parse_args()

    if args.log:
        path = Path(args.log)
        if not path.is_file():
            raise SystemExit(f'Log file not found: {path}')
    else:
        path = find_latest_log()
        if path is None:
            raise SystemExit(
                'No training_log.json / training_log.csv found in logs/.\n'
                'Run `python src/train_flow.py` first, or pass an explicit '
                'cluster stdout log with `--log PATH`.'
            )

    records = load_log(path)
    if not records:
        raise SystemExit(f'No parseable training records found in: {path}')

    # Sort by epoch in case a hand-written or concatenated log is unordered.
    records.sort(key=lambda r: r['epoch'])
    plot_and_save(records, source_label=str(path), out_prefix=args.out_prefix)


if __name__ == '__main__':
    main()
