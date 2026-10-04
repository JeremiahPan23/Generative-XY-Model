# Generative-XY-Model

**Data-free generative sampling of the two-dimensional XY model.**

An ongoing undergraduate physics research project investigating stochastic path sampling (SPS). The current study implements a data-free sampler on a **4 x 4 periodic lattice**, compares three training seeds and three training budgets, and explains a concrete Metropolis-Hastings (MH) sticking event. It has **not demonstrated a solution to BKT critical slowing down or a speedup over Wolff sampling**.

## Completed study

- **Data-free SPS:** training uses the XY Hamiltonian and probabilities of model-generated paths. Wolff configurations are used only for independent validation.
- **Generation-step comparison:** increasing path steps from 32 to 64 to 128 increased acceptance in the tested settings, but also increased cost. Higher acceptance did not guarantee higher effective sample size (ESS) per second.
- **Three paired training histories:** seeds 11, 22 and 33 were trained to 5,000 updates and continued to 10,000 and 15,000. Energy ESS/s initially improved for all nine settings; later changes depended on the seed. Unfavorable results are retained.
- **A diagnosed failure mode:** a high relative path weight caused 12,406 consecutive rejected transitions in one retained chain segment. The sampler kept proposing configurations while MH kept the old state. The recorded segment ends before the state is replaced.

These results concern **one small lattice and one temperature**, `beta=1.12` (`T ≈ 0.893` for `J=k_B=1`). Screening checks and finite-chain ESS estimates do not prove full equilibration or control of rare sticking events. This training-budget study stopped at 15,000 updates.

## Read the study

| Resource | Contents |
|---|---|
| [Method](docs/public/method.md) | XY target, angular paths, data-free training, path-space MH and diagnostics |
| [Results and limitations](docs/public/results.md) | Figures, controlled comparisons, costs and interpretation boundaries |
| [Published artifacts](publication/README.md) | Complete summaries for all 27 settings, selected figures and file hashes |
| [Historical Wolff / flow-matching baseline](docs/public/legacy.md) | Seven-temperature visualizations and the earlier supervised DiT experiment |
| [References](literature/README.md) | Foundational papers and BibTeX entries |

![Three paired training budgets](publication/figures/training_budget.png)

The figure displays `N=32` settings. The [CSV](publication/results/training_budget.csv) and [JSON](publication/results/training_budget.json) include all three generation-step counts and all seeds, including failed screening and unfavorable changes. Energy is reported **per bond**, `H/(2L^2)`.

## Install and verify

Use a Python environment with a compatible PyTorch installation. Run commands from the project root:

```bash
python -m pip install -r requirements-sps.txt
python -m unittest discover -s tests -p "test_*.py" -v
python scripts/verify_sps_a.py
python scripts/build_publication.py --check
```

The tests use temporary fixtures and require no historical datasets or checkpoints. Some perform tiny **CPU** updates to check resume behavior. They are software checks, not substitutes for physical validation. The original experiments used a CUDA GPU; dependency lists specify minimum versions, not an exact environment lock. For notebook tools, additionally install `requirements.txt`.

`verify_sps_a.py` checks seven core files against the original experiment bundle. Historical source/configuration files retain their original bytes, including line endings, because exact-run tools check SHA-256 hashes.

## Run a new experiment

The portable entry point is `scripts/sps_experiment.py`. For example, on a machine with a configured CUDA environment:

```bash
python scripts/sps_experiment.py train --config configs/sps_a_l4.json --out results/new_L4_seed11/train --device cuda --seed 11
python scripts/sps_experiment.py reference --L 4 --out results/new_L4_seed11/reference
python scripts/sps_experiment.py evaluate --checkpoint results/new_L4_seed11/train/checkpoint.pt --reference results/new_L4_seed11/reference --out results/new_L4_seed11/eval --device cuda --steps 32 64 128 --chains 8 --draws 16384 --burn 4096 --batch-size 128 --seed 22000
```

Training does not read the reference. Wolff sampling is a separate validation step. Use fresh output directories; the CLI refuses to overwrite completed outputs unless training is explicitly resumed. These commands reproduce the **workflow**, not the exact omitted historical run or its timing.

For Slurm, activate your compute-node environment and submit the generic template from the project root:

```bash
export SPS_PYTHON="$(command -v python)"
MODE=pilot L=4 SEED=11 sbatch scripts/slurm/sps_a.slurm
# After checking the pilot outputs:
MODE=full L=4 SEED=11 sbatch scripts/slurm/sps_a.slurm
```

The template requests `gpu:1` in a `gpu` partition. Adjust resources for your cluster. Code and the environment must be visible on the allocated node; transfer them first on clusters without a shared filesystem. The template's default evaluation is shorter than the published study. To match the long-chain settings, use the explicit `evaluate` command above.

## Contents and reproducibility

| Directory | Role |
|---|---|
| `src/` | Current angular SPS, Wolff and observables; historical DiT/FM code |
| `scripts/` | Portable experiment CLI, diagnostics, publication packaging and historical exact-run tools |
| `configs/` | Training settings and pinned protocols; `L=8` is an unreported option |
| `tests/` | Physics calculations, MH behavior and provenance/resume checks |
| `notebooks/` | Three historical Wolff/FM notebooks with cleared outputs |
| `docs/public/` | Selected research explanations |
| `publication/` | Lightweight results and figures copied from the local archive |
| `literature/` | Citation entries and source links |

**Included:** source, configurations, tests, complete selected summary tables, numerical result JSON, and selected figures made by this project. [The publication manifest](publication/manifest.json) records source filenames and hashes; copies are not new measurements.

**Kept outside Git:** raw datasets, checkpoints, NPZ proposal/chain archives, downloaded bundles, full logs, temporary files, third-party paper PDFs, and personal meeting/learning notes. They remain in the local research archive; GitHub is not its backup.

Historical comparison, continuation and sticking-analysis scripts need the omitted binary archive in its recorded directory layout. A clean clone supports inspecting results, checking published integrity, running tests and generating a **new** experiment. It cannot resume old models, independently replay their full MH chains or redraw every historical plot without those files. Intermediate SPS paths were not retained even in the original endpoint/weight archives.

The historical DiT's `VorticityExtractor` remains because the old model and checkpoint depend on it. Current SPS uses periodic CNNs without this module. There is no controlled DiT-versus-CNN result. The original cluster-specific submission wrapper is historical tooling, not the generic quickstart.

## Attribution

This sampler adapts the SPS framework of Chen et al. to periodic XY angles. Wolff provides reference validation. The project preserves an earlier supervised flow-matching investigation; it does not claim to originate SPS, Wolff, DiT or flow matching. See the [references](literature/README.md) and [method attribution](docs/public/method.md).

AI coding assistants supported implementation and explanatory materials. Published statements distinguish measurements, interpretations and untested claims. This repository is a research snapshot, not a claim of a published paper.
