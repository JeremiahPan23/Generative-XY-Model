# Published study artifacts

This directory contains lightweight copies of the project's own selected results. Original local experiments and their raw binary records have not been moved or overwritten.

## Results

| Files | Coverage |
|---|---|
| [training_budget.csv](results/training_budget.csv), [JSON](results/training_budget.json) | All 27 seed/budget/generation-step settings, including failed screening and unfavorable outcomes; JSON also contains observables, checks and limitations |
| [generation_steps.csv](results/generation_steps.csv), [JSON](results/generation_steps.json) | Three 5,000-update models evaluated at 32/64/128 steps |
| [sticking_mechanism.json](results/sticking_mechanism.json) | Recorded 12,406-rejection case, candidates, replay checks and limits |

Corresponding figures are [generation steps](figures/generation_steps.png), [training budget](figures/training_budget.png), and [sticking mechanism](figures/sticking_mechanism.png). PDF copies are supplied where the original analysis generated them.

Historical context: [seven-temperature Wolff configurations](figures/reference_multitemperature.png), [Wolff statistics](figures/reference_statistics.png), and [the earlier FM/Wolff comparison](figures/legacy_fm_wolff.png). These L=32 reference/baseline results are distinct from the L=4 SPS study.

The [English result guide](../docs/public/results.md) explains axes, representative numbers, costs and limitations. Figure copies retain their original labels.

## Integrity and reproduction

[manifest.json](manifest.json) records original relative filenames, SHA-256 and byte counts of copied artifacts, plus hashes of historical code/protocol files. Copies are byte-identical to source artifacts. Packaging does not recompute physical measurements or claim a fresh audit of omitted raw data.

From the repository root:

```bash
python scripts/build_publication.py --check
```

To rebuild these copies, restore the local research archive and run `python scripts/build_publication.py`. A clean clone intentionally lacks checkpoints, raw NPZ records and Wolff tensor datasets, so it cannot rebuild saved plots or replay historical chains by itself. Source filenames and hashes identify files; they do not replace them.

New experiments can be run using the [root README](../README.md). Historical audits require the omitted archive and compatible environment. Intermediate paths were not recorded, so even the original archive cannot independently recompute every full path density.
