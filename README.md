# Sampleworks

TODO: detailed project description, installation instructions, usage examples, and contribution guidelines.

## Basic instructions (Linux/Pixi):
1. Install [pixi](https://pixi.prefix.dev/latest/).
2. Clone the repository:
   ```bash
   git clone [this repository URL]
   ```
3. Create [pixi](https://pixi.prefix.dev/latest/) workspace and activate it (if necessary, copy `pixi.toml` to `pyproject.toml`):
   ```bash
   cd sampleworks
   pixi install -a # installs all environments
   ```
4. Activate the environment for your desired use case (e.g., Boltz sampling):
   ```bash
   pixi shell -e boltz
   ```
5. You may need to download the model checkpoints, e.g.,
   ```bash
   cd ~/boltz
   pixi run -e boltz-dev python -c "from boltz.main import download_boltz2; import pathlib; download_boltz2(pathlib.Path('~/boltz/'))"
   ```
6. Run the example script for Boltz sampling:
   ```bash
   python scripts/boltz2_pure_guidance.py --model-checkpoint ~/.boltz/boltz2_conf.ckpt --output-dir output/boltz2_pure_guidance --guidance-start 130 --structure tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif --density tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4 --resolution 1.8 --augmentation --align-to-input
   ```
## Development on OS X (WIP)
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
