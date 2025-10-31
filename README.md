# Sampleworks

TODO: detailed project description, installation instructions, usage examples, and contribution guidelines.

## Basic install instructions:
1. Clone the repository:
   ```
   git clone [this repository URL]
   ```
2. Create [pixi](pixi.sh) workspace and activate it:
   ```
   cd sampleworks
   pixi install -a # installs all environments
   ```
3. Activate the environment for your desired use case (e.g., Boltz sampling):
   ```
   pixi shell -e boltz
   ```
4. Run the example script for Boltz sampling:
   ```
   python examples/boltz2_pure_guidance.py --model-checkpoint ~/.boltz/boltz2_conf.ckpt --output-dir output/boltz2_pure_guidance --guidance-start 130 --structure examples/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif --density examples/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4 --resolution 1.8 --augmentation --align-to-input
   ```
