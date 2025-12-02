# Sampleworks

TODO: detailed project description, installation instructions, usage examples, and contribution guidelines.

## Basic instructions (Linux/Pixi):
0. Install [pixi](https://github.com/pixi-framework/pixi).
1. Clone the repository:
   ```bash
   git clone [this repository URL]
   ```
2. Create [pixi](pixi.sh) workspace and activate it (if necessary, copy `pixi.toml` to `pyproject.toml`):
   ```bash
   cd sampleworks
   pixi install -a # installs all environments
   ```
3. Activate the environment for your desired use case (e.g., Boltz sampling):
   ```bash
   pixi shell -e boltz
   ```
4. Run the example script for Boltz sampling:
   ```bash
   python examples/boltz2_pure_guidance.py --model-checkpoint ~/.boltz/boltz2_conf.ckpt --output-dir output/boltz2_pure_guidance --guidance-start 130 --structure examples/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif --density examples/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4 --resolution 1.8 --augmentation --align-to-input
   ```
## Development on OS X (WIP)
To develop on OS X, ensure you have [homebrew](https://brew.sh/) installed and run the following commands to install dependencies:

1. Install hatch
    ```bash
    brew install hatch
    ```
2. Move/copy `pyproject-hatch.toml` to `pyproject.toml` 
3. Use `hatch run <command>` to run commands.
4. Use `hatch run <env>:<command>` to run commands in a specific environment `<env>`.

There are different (and as yet untested) environments for `boltz`. `protenix` won't currently work on a Mac due to 
the strict requirement of `triton` which requires an NVIDIA GPU. You may find similar issues with other environments.
Debug as needed.