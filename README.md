# Sampleworks

> This repository is under heavy and active development. Expect breaking changes. Please feel free to reach out to the team if you're interested in using or contributing!

**Sampleworks** is a Python framework for integrating generative biomolecular structure models with experimental data.

## Why sampleworks?

Biomolecular structure prediction and design models are currently trained on single state structures and fail to accurately predict the ensemble of conformations each macromolecule occupies. But there is still hope! Current models show promise in capturing the underlying distribution of realistic macromolecular structures. We want to utilize the prior represented in these models and experimental observations to improve the sampling of the underlying ensemble present in the experiment and use this information to both understand biomolecular function and improve ensemble prediction.

Currently, each structure prediction model has a different implementation, requiring bespoke boilerplate code to plug each model into experimental guidance. Our goal is to resolve this and expand the experimental methods we can provide guidance with. This will open new opportunities for model evaluation directly against experimental data, and help unlock new sources of data for training the next generation of biomolecular structure predictors.

## Installation

**Requirements**: Linux x86-64, CUDA 12, Python ≥ 3.11

### 1. Install Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | sh
```

### 2. Clone and install

```bash
git clone
cd sampleworks
pixi install -a   # install all environments
```

Each generative model has its own Pixi environment. Install only what you need:

```bash
pixi install -e boltz      # Boltz-1 / Boltz-2
pixi install -e protenix   # Protenix
pixi install -e rf3        # RosettaFold3
```

### 3. Download model checkpoints

**Boltz-1 and Boltz-2** (stored in `~/.boltz/`):

```bash
pixi run -e boltz python -c "
from boltz.main import download_boltz1, download_boltz2
import pathlib
cache = pathlib.Path('~/.boltz/').expanduser()
download_boltz1(cache)
download_boltz2(cache)
"
```

**Protenix**: checkpoint is downloaded automatically on first use.

**RosettaFold3** (RF3): see the [RC-Foundry repository](https://github.com/RosettaCommons/foundry) for instructions. Default path: `~/.foundry/checkpoints/rf3_foundry_01_24_latest.ckpt`


## Quick Start

Run Boltz-2 pure guidance on the included 1VME example:

```bash
pixi run -e boltz python scripts/boltz2_pure_guidance.py \
    --model-checkpoint ~/.boltz/boltz2_conf.ckpt \
    --structure tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif \
    --density tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4 \
    --resolution 1.8 \
    --output-dir output/boltz2_pure_guidance \
    --guidance-start 130 \
    --ensemble-size 4 \
    --augmentation \
    --align-to-input
```

Output files appear in `output/boltz2_pure_guidance/`: `refined.cif` (final ensemble), `losses.txt`, `trajectory/`, `run.log`. See [`scripts/README.md`](scripts/README.md) for all scripts and arguments.


## Grid Search

`run_grid_search.py` sweeps a model across scalers, ensemble sizes, and gradient weights:

```bash
pixi run -e boltz python run_grid_search.py \
    --proteins proteins.csv \
    --models boltz2 \                # options: boltz1, boltz2, protenix, rf3 (make sure env aligns!)
    --methods "X-RAY DIFFRACTION" \  # only useful for Boltz-2, ignored otherwise
    --scalers pure_guidance \        # options: pure_guidance, fk_steering, or both as space-separated list
    --ensemble-sizes "1 4" \
    --gradient-weights "0.1 0.2" \
    --output-dir grid_search_results \
    --gradient-normalization \       # normalize guidance update magnitude to diffusion update magnitude (TODO: document further)
    --augmentation \                 # apply random rotations and translations at each step (defaults for inference with AF3-like models)
    --align-to-input                 # align to input structure at each step (required for density guidance to work since it is not rotation/translation invariant)
```

**`proteins.csv` format**

Required columns and format:
```
name,structure,density,resolution
1abc,/data/structures/1abc.cif,/data/maps/1abc.ccp4,2.0
2xyz,/data/structures/2xyz.cif,/data/maps/2xyz.mrc,1.8
```

**Key arguments:**

| Argument | Description | Default |
|---|---|---|
| `--proteins` | CSV with structure/density/resolution columns | required |
| `--models` | Model to run. One of `boltz1`, `boltz2`, `protenix`, `rf3` | required |
| `--scalers` | Guidance method(s) to sweep | `pure_guidance fk_steering` |
| `--ensemble-sizes` | Space-separated values, e.g. `"1 4"` | `"1 2 4 8"` |
| `--gradient-weights` | Space-separated values, e.g. `"0.1 0.2"` | `"0.01 0.1 0.2"` |
| `--methods` | Boltz-2 sampling method (required for boltz2) | `X-RAY DIFFRACTION` |
| `--max-parallel` | Parallel workers (default: number of GPUs) | `auto` |
| `--dry-run` | Print jobs without running them | off |
| `--force-all` | Re-run including already-successful jobs | off |
| `--only-failed` | Re-run only failed jobs | off |
| `--only-missing` | Run only jobs not yet started | off |

Output layout: `grid_search_results/<protein>/<model>[_<method>]/<scaler>/ens<N>_gw<W>/`


## Docker

TODO: Docker container documentation


## Development

We use [Pixi](https://pixi.sh/) to manage development environments and dependencies. Each model has its own environment, e.g. `boltz-dev`, `protenix-dev`, `rf3-dev`. To install dev dependencies and run tests:

```bash
pixi install -e [model]-dev    # add pytest, ruff, ty
pixi run -e [model]-dev all-tests  # run tests
pixi run test-all            # run all tests across all environments
```

**Prek hooks** (various formatting, ruff + ty type checking):

```bash
pixi run prek install
pixi run prek run --all-files
```

See [`tests/README.md`](tests/README.md) for full testing instructions.


## macOS (experimental)

To develop on OS X, ensure you have [homebrew](https://brew.sh/) installed and run the following commands to install dependencies:

1. Install hatch and uv
    ```bash
    brew install hatch uv
    ```
2. Move/copy `pyproject-hatch.toml` to `pyproject.toml`
3. Use `uvx hatch run <command>` to run commands. Note the use of `uvx` instead of `uv`
4. Use `uvx hatch run <env>:<command>` to run commands in a specific environment `<env>`.

There are different (and as yet untested) environments for `boltz`. `protenix` won't currently work on a Mac due to
the strict requirement of `triton` which requires an NVIDIA GPU. You may find similar issues with other environments.
Debug as needed.
