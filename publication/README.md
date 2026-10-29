# Published study artifacts

This directory contains selected figures and compact result exports from the project's existing analyses. Updated 2026-10-29 to include work completed through the 2026-10-21 saved-MH analysis. Original experiments, checkpoints and binary records have not been moved or overwritten.

## Recent results

| Files | Coverage |
|---|---|
| [path_steps_2000.csv](results/path_steps_2000.csv), [figure](figures/path_steps_2000.png) | All 21 fixed-model generation-step settings through N=2000, including three seeds and costs; the local summary-path column is removed |
| [three_temperatures.csv](results/three_temperatures.csv), [figure](figures/three_temperatures.png) | Nine separately trained single-temperature models; all seeds retained, local summary paths removed |
| [shared_nelson.csv](results/shared_nelson.csv), [JSON](results/shared_nelson.json), [efficiency](figures/shared_nelson_efficiency.png), [physics](figures/shared_nelson_physics.png) | Shared free/Nelson seed11 pair evaluated at three training temperatures; includes recorded costs and limitations |
| [multitemperature_efficiency.csv](results/multitemperature_efficiency.csv), [figure](figures/multitemperature_efficiency.png) | Both arms at 12 finite temperatures, 4096 candidates per arm, common 1035-step full mesh |
| [mh_observables.csv](results/mh_observables.csv), [JSON](results/mh_observables.json), [main figure](figures/mh_observables.png) | Saved MH-chain energy, vortex density and radial susceptibility, both arms, finite-chain warnings and block-length sensitivity |
| [raw/MH comparison](figures/raw_mh_observables.png), [low-temperature detail](figures/mh_low_temperature.png) | Supplementary generator diagnostics; raw bias remains visible |

The [current progress guide](../docs/public/progress.md) explains the different models, sample budgets, normalization, uncertainty and the remaining single-seed scope of the Nelson comparison. Matching MH-corrected observables and higher candidate-weight ESS/n do not establish general mixing, Wolff speedup or BKT scaling.

## Earlier baseline artifacts

| Files | Coverage |
|---|---|
| [training_budget.csv](results/training_budget.csv), [JSON](results/training_budget.json) | All 27 seed/budget/generation-step settings, including failed screening and unfavorable outcomes; JSON also contains observables, checks and limitations |
| [generation_steps.csv](results/generation_steps.csv), [JSON](results/generation_steps.json) | Three 5,000-update models evaluated at 32/64/128 steps |
| [sticking_mechanism.json](results/sticking_mechanism.json) | Recorded 12,406-rejection case, candidates, replay checks and limits |

Corresponding figures are [generation steps](figures/generation_steps.png), [training budget](figures/training_budget.png), and [sticking mechanism](figures/sticking_mechanism.png). PDF copies are supplied where the original analysis generated them.

Historical context: [seven-temperature Wolff configurations](figures/reference_multitemperature.png), [Wolff statistics](figures/reference_statistics.png), and [the earlier FM/Wolff comparison](figures/legacy_fm_wolff.png). These L=32 reference/baseline results are distinct from the L=4 SPS study.

The [historical result guide](../docs/public/results.md) explains the baseline axes, representative numbers, costs and limitations. Figure copies retain their original labels; the single-temperature studies are distinct from the new shared-model comparison.

## Integrity and reproduction

[manifest.json](manifest.json) records repository-relative source filenames, output fingerprints and a reviewed implementation-file list. The old artifact copies remain unchanged. New figures and unchanged tables are byte-identical copies; path-column removal and JSON field selection are explicitly identified as **derived public exports**, with both source and output fingerprints. These exports retain numerical results and adverse diagnostics while omitting private execution paths and unrelated fields. Packaging does not recompute physical measurements or claim a fresh audit of omitted raw data.

From the repository root:

```bash
python scripts/build_publication.py --check
```

To rebuild the selected exports, restore the local research archive and run `python scripts/build_publication.py`. For a local comparison against the available sources, run `python scripts/build_publication.py --check-sources`; this also verifies that projected exports can be regenerated from their pinned source summaries. `--check` alone requires only the published files and implementation, so it works on a clean clone.

A clean clone intentionally lacks checkpoints, raw NPZ records and Wolff tensor datasets, so it cannot redraw original analyses or replay historical chains by itself. Source filenames and hashes identify files; they do not replace them. The exporter uses an explicit implementation allowlist: unrelated local scripts and private configuration files are not silently added to the public manifest.

New experiments can be run using the [root README](../README.md). Historical audits require the omitted archive and compatible environment. Intermediate paths were not recorded, so even the original archive cannot independently recompute every full path density.
