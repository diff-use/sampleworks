"""Generate one conformational ensemble: one structure + one density map -> one output dir.

This is the refinement step of a longer workflow, so it expects a model that is already sitting
in the map's frame. Starting from a sequence and a map, the usual route is:

  1. predict   sequence -> coordinates            (Protenix inference, no map involved)
  2. place     put that model in the map's frame  (molecular replacement, e.g. Phaser; not
                                                  part of this repo)
  3. refine    this script

A deposited structure or an existing MR solution is already placed, so it starts at step 3 --
which is what the runs in it_opt_scratch/ have done.

That is also what --structure is for. It carries the sequence and composition (there is no
separate sequence input in this pipeline), the atom identity used for reconciliation, and the
reference frame: the density reward scores coordinates in the map's frame and aligns against
the input structure rather than the map, so a model in an unrelated frame will not score
meaningfully.

Otherwise, read this as a plain input/output tool -- one run per invocation, no state:

  IN    --structure    one .cif / .pdb
        --density      one .ccp4 map
        --resolution   that map's resolution, in Angstrom
        --mode         which guidance arm to run (list below)
        (everything else is a knob with a default; see --help)

  OUT   --output-dir   refined.cif  trajectory/  losses.txt  job_metadata.json  run.log

  EXIT  the pipeline's own exit code -- 0 on success, non-zero on failure, so a caller
        looping over many targets can tell which ones need rerunning.

The generation itself is not implemented here: this builds a GuidanceConfig and hands it to
the shipped `run_guidance()`, which does featurize -> reward -> scaler -> sample -> save.

Modes:
  baseline        unguided sampling      (pure_guidance, no step scaler)
  s_only          IT-opt, which_latent=single
  z_only          IT-opt, which_latent=pair
  s_plus_z        IT-opt, which_latent=both
  coord_guidance  shipped coordinate DPS (pure_guidance + noise-space step scaler)

One run:
  pixi run -e protenix-dev python it_opt_scratch/run_targets_simplified.py \
      --structure /home/dev/test_data/processed/2YL0/2YL0_single_001_density_input.cif \
      --density   density_maps/2YL0_0.5occA_0.5occB_1.00A.ccp4 \
      --resolution 1.0 --mode s_plus_z --ensemble-size 8 --bond-length-weight 5e-5 \
      --output-dir out/2YL0_s_plus_z_ens8 --device cuda:0

A whole CSV, one after another on one GPU (add --skip-existing to resume a stopped sweep):
  tail -n +2 targets.csv | while IFS=, read -r name structure density resolution; do
      pixi run -e protenix-dev python it_opt_scratch/run_targets_simplified.py \
          --structure "$structure" --density "$density" --resolution "$resolution" \
          --mode s_plus_z --ensemble-size 8 --bond-length-weight 5e-5 \
          --output-dir "out/$name" --device cuda:0 || echo "FAILED: $name"
  done

Spreading a CSV across several GPUs is deliberately NOT this script's job -- that belongs to a
separate driver that calls this one.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

from sampleworks.utils.guidance_constants import GuidanceType, StructurePredictor
from sampleworks.utils.guidance_script_arguments import GuidanceConfig
from sampleworks.utils.guidance_script_utils import get_model_and_device, run_guidance

MODES = ("baseline", "coord_guidance", "s_only", "z_only", "s_plus_z")
WHICH_LATENT = {"s_only": "single", "z_only": "pair", "s_plus_z": "both"}


def main():
    """Read the inputs, run one ensemble generation, return the pipeline's exit code."""
    args = parse_args()

    structure = Path(args.structure).expanduser()
    density = Path(args.density).expanduser()
    out_dir = Path(args.output_dir).expanduser()

    if not structure.exists():
        sys.exit(f"structure not found: {structure}")
    if not density.exists():
        sys.exit(f"density not found: {density}")

    if args.skip_existing and (out_dir / "refined.cif").exists():
        print(f"[skip] already done: {out_dir}/refined.cif")
        return 0
    out_dir.mkdir(parents=True, exist_ok=True)

    config = build_config(args, structure, density, out_dir)
    device, model = get_model_and_device(
        args.device, args.checkpoint, StructurePredictor(args.model)
    )
    torch.manual_seed(args.seed)  # fixed seed -> the same start noise across modes

    job = run_guidance(config, config.guidance_type, model, device)
    print(f"[{args.mode}] {job.status} -> {out_dir}/refined.cif")
    return job.exit_code


def build_config(args, structure: Path, density: Path, out_dir: Path) -> GuidanceConfig:
    """Everything this run will do, in one place: the inputs, the knobs, and the mode.

    The mode-specific fields are assigned directly rather than declared on GuidanceConfig,
    which is how the shipped per-guidance-type arg-adders set them too.
    """
    # baseline and coord_guidance run the shipped pure_guidance path; the rest optimize latents.
    if args.mode in ("baseline", "coord_guidance"):
        guidance_type = GuidanceType.PURE_GUIDANCE
    else:
        guidance_type = GuidanceType.LATENT_OPT

    config = GuidanceConfig(
        protein=args.name or structure.stem,
        structure=str(structure),
        density=str(density),
        resolution=args.resolution,
        output_dir=str(out_dir),
        log_path=str(out_dir / "run.log"),
        model_name=args.model,
        guidance_type=guidance_type,
        num_diffusion_steps=args.num_steps,
        align_to_input=True,
    )
    config.ensemble_size = args.ensemble_size

    if args.mode == "baseline":  # unguided: no step scaler at all
        config.step_scaler_type = "none"
    elif args.mode == "coord_guidance":  # gradients applied in coordinate space
        config.step_scaler_type = "noisespace"
        config.step_size = args.step_size
    else:  # s_only / z_only / s_plus_z: optimize the latents instead of the coordinates
        config.which_latent = WHICH_LATENT[args.mode]
        config.learning_rate = args.lr
        config.outer_steps = args.outer_steps
        config.anchor_weight = args.anchor
        config.max_grad_norm = args.max_grad_norm
        config.bond_length_weight = args.bond_length_weight

    return config


def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    inputs = ap.add_argument_group("in / out")
    inputs.add_argument("--structure", required=True, help="input .cif / .pdb")
    inputs.add_argument("--density", required=True, help="input .ccp4 map")
    inputs.add_argument("--resolution", type=float, required=True, help="map resolution, Angstrom")
    inputs.add_argument("--output-dir", dest="output_dir", required=True,
                        help="everything this run produces is written here")
    inputs.add_argument("--mode", required=True, choices=MODES)
    inputs.add_argument("--name", help="label recorded in job_metadata (default: structure stem)")
    inputs.add_argument("--skip-existing", dest="skip_existing", action="store_true",
                        help="succeed without running if refined.cif is already there")

    run = ap.add_argument_group("sampling")
    run.add_argument("--ensemble-size", dest="ensemble_size", type=int, default=4)
    run.add_argument("--num-steps", dest="num_steps", type=int, default=200)
    run.add_argument("--seed", type=int, default=0)
    run.add_argument("--model", default="protenix", choices=[m.value for m in StructurePredictor])
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--checkpoint", default=None)

    it = ap.add_argument_group("IT-opt modes only (s_only / z_only / s_plus_z)")
    it.add_argument("--lr", type=float, default=0.05)
    it.add_argument("--outer-steps", dest="outer_steps", type=int, default=2)
    it.add_argument("--anchor", type=float, default=0.0, help="on-manifold anchor weight")
    it.add_argument("--bond-length-weight", dest="bond_length_weight", type=float, default=0.0,
                    help="coordinate-space bond-geometry penalty; 0 disables")
    it.add_argument("--max-grad-norm", dest="max_grad_norm", type=float, default=1.0)

    ap.add_argument("--step-size", dest="step_size", type=float, default=0.1,
                    help="coord_guidance mode only: DPS step size")
    return ap.parse_args()


if __name__ == "__main__":
    sys.exit(main())
