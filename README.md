# Generative-XY-Model

**Data-free generative sampling of the two-dimensional XY model.**

An ongoing undergraduate physics research project implementing stochastic path sampling (SPS) on periodic angular variables. The study has progressed from a single-temperature baseline to shared-temperature **free / Nelson** training and a **12-temperature evaluation of frozen models**. Wolff configurations are used for validation, not as SPS training data.

**Updated 2026-10-29.** The latest measurements and saved-data analysis were completed by 2026-10-21; this update publishes selected existing results, not new training. All reported SPS experiments use a **4 x 4 lattice**. The earlier single-temperature study has three training seeds; the shared free/Nelson comparison currently has **one paired training seed, 11**. There is no demonstrated solution to BKT critical slowing down or speedup over Wolff.

## Current progress

| Stage | Completed work | Evidence and limits |
|---|---|---|
| Baseline and diagnostics | Data-free angular SPS; three training seeds and 5,000 / 10,000 / 15,000-update comparisons | [Historical results](docs/public/results.md), including unfavorable outcomes and a recorded 12,406-rejection stay |
| Longer generation paths | Fixed three existing central-temperature models, evaluated through `N=2000` without retraining | Acceptance **48.96%, 46.61%, 45.10%**; all remain below the project's 50% stage target |
| Three temperatures, separate training | Three single-temperature models per temperature at `T=0.87, 1/1.12, 0.92` | Nine finite-run physical screens passed; these are not nine shared models or a three-seed Nelson replication |
| Shared-temperature free / Nelson | One model per arm, seed 11, 15,000 updates; both use three prefix losses | Nelson's original three-temperature acceptance **55.03%–56.03%**, candidate-weight ESS/n **51.31%–52.53%** |
| Frozen-model temperature scan | **4,096 paths per arm**, 12 finite temperatures from **8.7 to 0.87**, one common **1,035-step** full mesh | Nelson acceptance **55.87%–74.83%**, candidate-weight ESS/n **52.28%–83.35%**, higher than free at all 12 sampled temperatures in this evaluation |
| Saved MH-chain physics | Energy, vortex density and radial susceptibility from **4 x 895 retained records per temperature**, including rejection repeats | Nelson is compatible with Wolff for these observables within the present finite uncertainty; free has finite-chain warnings |

These stages have different models, meshes and budgets. Their differences cannot all be attributed to Nelson's relation. See [the current progress and evaluation protocol](docs/public/progress.md) for the comparisons, uncertainties, costs and remaining questions.

## Latest figures

### Complete sampler: MH-corrected physics versus Wolff

![Nelson MH-corrected energy, vortex density and radial susceptibility versus Wolff](publication/figures/mh_observables.png)

The green curves are the **Nelson + path-space MH** output; black curves are same-size Wolff references. Error bars are pointwise 95% block-bootstrap intervals. The displayed susceptibility is explicitly the magnetization-modulus fluctuation, `chi_radial = beta/V * (mean(|M|^2) - mean(|M|)^2)`, with `V=16`. Agreement of selected observables is finite evidence, not a proof of full equilibration or BKT scaling.

### Acceptance and candidate-path importance weights

![Free versus Nelson acceptance and candidate-path importance ESS/n at 12 temperatures](publication/figures/multitemperature_efficiency.png)

Both arms share the original three-temperature training task. The importance denominator is **4,096 candidate paths**; the acceptance denominator is **3,580 retained attempted transitions**. Neither is the observable-specific MCMC ESS. The 50% line is a project stage target, not a validity theorem. Higher acceptance or importance ESS/n does not by itself establish better effective sampling per second.

[Raw / MH / Wolff comparison](publication/figures/raw_mh_observables.png) and [low-temperature detail](publication/figures/mh_low_temperature.png) are retained as diagnostics. Raw proposals still show physical bias; that is a generator diagnostic rather than evidence that the complete SPS/MH sampler is invalid. **All 12 temperatures share path prefixes, not 12 independent training replications.**

## Read the study

| Resource | Contents |
|---|---|
| [Current progress](docs/public/progress.md) | Longer-path, separate-temperature, shared-Nelson and 12-temperature results; present scope |
| [Method](docs/public/method.md) | XY target, angular kernels, prefix training, Nelson convention and path-space MH |
| [Historical baseline](docs/public/results.md) | Training-budget comparisons, effective information rates and a long-stay diagnosis |
| [Published artifacts](publication/README.md) | Selected figures, complete compact tables, projected diagnostic summaries and hashes |
| [Historical Wolff / flow matching](docs/public/legacy.md) | Seven-temperature Wolff visualizations and the earlier supervised DiT pipeline |
| [References](literature/README.md) | Foundational papers, attribution and BibTeX |

The current research phase is consolidation of these results and definition of the next replication and mechanism checks. Additional paired training seeds, larger lattices and a matched Wolff cost comparison remain future work, not completed results.

## Install and verify

Use a Python environment with a compatible PyTorch installation. From the project root:

```bash
python -m pip install -r requirements-sps.txt
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/verify_sps_a.py
python scripts/build_publication.py --check
```

The published tests use synthetic fixtures and tiny CPU operations, without historical checkpoints or production sampling. They check software behavior, not scientific validity. Original production runs used a CUDA GPU; dependency lists specify minimum versions, not an exact environment lock. Notebook tools additionally require `requirements.txt`.

`verify_sps_a.py` checks seven original core files against their pinned hashes. The publication checker verifies published artifacts and an explicit implementation-file list, without requiring private data. Source/configuration bytes and line endings remain preserved where historical tools pin SHA-256.

## Run a new baseline experiment

The portable CLI below is the **single-temperature baseline workflow**. It does not reproduce the shared-model temperature scan or recover the omitted historical models:

```bash
python scripts/sps_experiment.py train --config configs/sps_a_l4.json --out results/new_L4_seed11/train --device cuda --seed 11
python scripts/sps_experiment.py reference --L 4 --out results/new_L4_seed11/reference
python scripts/sps_experiment.py evaluate --checkpoint results/new_L4_seed11/train/checkpoint.pt --reference results/new_L4_seed11/reference --out results/new_L4_seed11/eval --device cuda --steps 32 64 128 --chains 8 --draws 16384 --burn 4096 --batch-size 128 --seed 22000
```

Training does not read the reference. Use fresh output directories; the CLI refuses to overwrite completed outputs unless training is explicitly resumed. To use Slurm, adjust the generic `scripts/slurm/sps_a.slurm` resources and ensure the project and environment are available on the allocated node:

```bash
export SPS_PYTHON="$(command -v python)"
MODE=pilot L=4 SEED=11 sbatch scripts/slurm/sps_a.slurm
# After checking the pilot outputs:
MODE=full L=4 SEED=11 sbatch scripts/slurm/sps_a.slurm
```

The template's default evaluation is shorter than the long-chain baseline study. The shared-model implementation is provided separately in [sps_nelson.py](src/sps_nelson.py), with [raw-prefix collection](src/sps_intermediate.py), [path scoring and MH](src/sps_intermediate_efficiency.py), and synthetic tests. Historical cluster-specific launchers and pinned data archives are not part of this portable CLI example.

## Contents and reproducibility

| Directory | Role |
|---|---|
| `src/` | Angular SPS, shared-temperature Nelson/free model, prefix scoring, Wolff and observables; historical DiT/FM code |
| `scripts/` | Portable baseline CLI, diagnostics, selected-publication export and historical exact-run tools |
| `configs/` | Published baseline settings; the `L=8` option is not a reported result |
| `tests/` | Synthetic physics/probability, MH, resume, prefix and export-integrity checks |
| `docs/public/` | Public method, progress, baseline and historical explanations |
| `publication/` | Lightweight figures and result exports from existing analyses |
| `literature/` | Citations and source links |

**Included:** selected source and tests, compact numerical summaries, original project figures and a [publication manifest](publication/manifest.json). Figure and unchanged-table copies retain their original bytes; tables with local-path columns removed and JSON field projections are explicitly recorded as **derived public exports**, with source and output fingerprints. Exporting does not recompute physical measurements.

**Kept local:** checkpoints, raw datasets, NPZ proposal/chain banks, downloaded bundles, full logs, personal meeting/learning notes, supervisor chat screenshots and machine-specific execution records. GitHub is not the full research archive.

A clean clone supports reading the selected evidence, integrity checks, synthetic tests and new baseline experiments. Historical models and raw records are needed to resume or independently audit the original runs. Complete intermediate trajectories were not retained in the published endpoint/weight archives; saved-state replay must not be described as reconstructing every transition density from saved full paths.

The historical DiT's `VorticityExtractor` remains for compatibility. Current SPS uses periodic CNNs without it; no controlled experiment establishes that attention or flow matching itself is unsuitable.

## Attribution

This project adapts the SPS framework of Chen et al. to periodic XY angles and studies a Nelson-tied backward drift. Wolff provides reference validation. It does not claim to originate SPS, Nelson's relation, Wolff, DiT or flow matching. See [method attribution](docs/public/method.md) and [references](literature/README.md).

AI coding assistants supported implementation and explanatory materials. Measurements, finite-run observations and proposed work are identified separately. This repository is an ongoing research snapshot, not a claim of a published paper or established novelty.
