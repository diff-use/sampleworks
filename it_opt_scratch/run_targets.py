"""Batch conformational-ensemble generation over a list of protein targets.

This drives the sampleworks guidance pipeline through its public entry point (`run_guidance`): the
model is loaded ONCE, a `GuidanceConfig` is built per run, and the pipeline itself does
featurize -> reward -> scaler -> sample -> save. Output for each (target, mode) lands under
<out_dir>/<mode>/ (refined.cif + trajectory + losses). A per-run failure is recorded and the batch
continues; a batch_summary.json is written at the end.

Where the inputs come from (three forms, checked in this order):
  * CSV   --targets x.csv     columns: name,structure,density,resolution
                              (out dir = <output-base>/<name>)
  * JSON  --targets x.json    list of {name, density, structure, resolution, out_dir}
  * IDs   --proteins 2YL0,5I09  /  --proteins-file ids.txt
          paths are built from --structure-dir/--structure-template and
          --density-dir/--density-template, so you only list PDB IDs. Templates take
          {pdb} (as written), {PDB} (upper) and {pdb_low} (lower).
  * none                      the built-in DEFAULT_TARGETS (1vme)

Where the outputs go:
  <output-base>/<name>/<mode>/   refined.cif  trajectory/  losses.txt  job_metadata.json  run.log
  --summary path                 batch_summary.json (per-shard copies alongside it when sharded)

Modes (--modes, comma-separated, or 'all') -> guidance type:
  baseline        unguided sampling      (pure_guidance, no step scaler)
  s_only          IT-opt, which_latent=single
  z_only          IT-opt, which_latent=pair
  s_plus_z        IT-opt, which_latent=both
  coord_guidance  shipped coordinate DPS (pure_guidance + noise-space step scaler)

Run on the pod from the repo root -- one process, explicit CSV:
  pixi run -e protenix-dev python it_opt_scratch/run_targets.py \
      --targets /home/dev/test_data/proteins.csv --output-base it_opt_scratch/targets_out \
      --modes baseline,z_only,coord_guidance --ensemble-size 4 --num-steps 200 --outer-steps 2

Run on 4 GPUs with 2 proteins in flight per GPU (8 worker processes), IDs + directories:
  pixi run -e protenix-dev python -u it_opt_scratch/run_targets.py \
      --proteins-file it_opt_scratch/regen11.txt \
      --structure-dir /home/dev/test_data/processed \
      --density-dir it_opt_scratch/targets_out_11_regen_0.5occA_0.5occB/density_maps \
      --density-template '{PDB}_0.5occA_0.5occB_1.00A.ccp4' \
      --name-template '{PDB}_0.5occA_0.5occB' --resolution 1.0 \
      --output-base it_opt_scratch/targets_out_11_regen_0.5occA_0.5occB \
      --modes s_plus_z --ensemble-size 8 --bond-length-weight 5e-5 \
      --gpus 0,1,2,3 --jobs-per-gpu 2

Add --dry-run to either command to print the resolved plan (paths, existence, shard
assignment) without loading the model.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

import torch

from sampleworks.utils.guidance_constants import GuidanceType, StructurePredictor
from sampleworks.utils.guidance_script_arguments import GuidanceConfig
from sampleworks.utils.guidance_script_utils import get_model_and_device, run_guidance

REPO = Path(__file__).resolve().parents[1]
ALL_MODES = ["baseline", "s_only", "z_only", "s_plus_z", "coord_guidance"]

DEFAULT_TARGETS = [
    {
        "name": "1vme",
        "density": "tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4",
        "structure": "tests/resources/1vme/1vme_final_carved_edited_0.5occA_0.5occB.cif",
        "resolution": 1.8,
        "out_dir": "it_opt_scratch/targets_out/1vme",
    },
]


# ============================ the batch (top-down) ============================

def main() -> None:
    args = parse_args()
    targets = collect_targets(args)
    modes = ALL_MODES if args.modes == "all" else [m for m in args.modes.split(",") if m]

    if args.dry_run:  # resolve and report the plan without touching a GPU
        print_plan(targets, modes, args)
        return

    # A parent process (--gpus given, no --shard-index) only fans out; the children do the work.
    if args.gpus and args.shard_index is None:
        launch_workers(targets, args)
        return

    if args.num_shards > 1:  # this process is one worker: take its slice, round-robin
        targets = targets[args.shard_index :: args.num_shards]

    print(f"targets={[t['name'] for t in targets]}  modes={modes}  model={args.model}\n"
          f"ensemble_size={args.ensemble_size} num_steps={args.num_steps} "
          f"outer_steps={args.outer_steps} lr={args.lr} anchor={args.anchor}")

    device, model = get_model_and_device(args.device, args.checkpoint, StructurePredictor(args.model))

    summary: list[dict] = []
    for target in targets:
        print(f"\n######### TARGET: {target['name']} #########")
        for mode in modes:
            summary.append(generate_one(target, mode, model, device, args))

    write_summary(summary, shard_summary_path(resolve(args.summary), args.shard_index))


def generate_one(target: dict, mode: str, model, device, args) -> dict:
    """Generate + save one ensemble for (target, mode) via run_guidance; return a summary record."""
    name = target["name"]
    out_dir = resolve(target["out_dir"]) / mode
    try:
        guidance_type, extras = guidance_for_mode(mode, args)
        config = build_config(target, guidance_type, out_dir, args)
        for key, value in extras.items():  # mode-specific args the arg-adders would otherwise set
            setattr(config, key, value)
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.manual_seed(args.seed)  # same start noise across modes -> comparable ensembles
        job = run_guidance(config, guidance_type, model, device)
        status = getattr(job, "status", "unknown")
        print(f"[{name}/{mode}] {status} -> {out_dir}/refined.cif")
        return {"target": name, "mode": mode, "status": "OK" if status == "success" else "FAILED",
                "out_dir": str(out_dir)}
    except Exception:
        print(f"[{name}/{mode}] FAILED:")
        traceback.print_exc()
        return {"target": name, "mode": mode, "status": "FAILED"}


# ============================== target inputs ===============================
# Added for the ID-and-directory input mode. The CSV/JSON contract below is unchanged --
# collect_targets() only falls through to the ID mode when --targets is not given.

def collect_targets(args) -> list[dict]:
    """Build the target list from --targets, or from --proteins/--proteins-file, or the default.

    Every target is a dict with name, structure, density, resolution, out_dir -- the same
    shape run_guidance has always consumed. Missing input files abort the run here rather
    than partway through the batch."""
    output_base = resolve(args.output_base)
    if args.targets:
        targets = load_targets(args.targets, output_base)
    elif args.proteins or args.proteins_file:
        targets = targets_from_ids(read_ids(args.proteins, args.proteins_file), output_base, args)
    else:
        targets = DEFAULT_TARGETS

    missing = [
        f"{t['name']}: {role}={t[role]}"
        for t in targets
        for role in ("structure", "density")
        if not resolve(t[role]).exists()
    ]
    if missing and not args.allow_missing:
        raise SystemExit(
            "input files not found (pass --allow-missing to stage a run anyway):\n  "
            + "\n  ".join(missing)
        )
    return targets


def read_ids(inline: str | None, path: str | None) -> list[str]:
    """PDB IDs from a comma-separated --proteins value and/or a --proteins-file (one per
    line, '#' comments and blank lines skipped)."""
    ids = [i.strip() for i in (inline or "").split(",") if i.strip()]
    if path:
        for line in resolve(path).read_text().splitlines():
            line = line.split("#")[0].strip()
            if line:
                ids.append(line)
    return ids


def targets_from_ids(ids: list[str], output_base: Path, args) -> list[dict]:
    """Expand PDB IDs into targets using the structure/density directory templates."""
    return [
        {
            "name": fill(args.name_template, pdb),
            "structure": str(resolve(args.structure_dir) / fill(args.structure_template, pdb)),
            "density": str(resolve(args.density_dir) / fill(args.density_template, pdb)),
            "resolution": args.resolution,
            "out_dir": str(output_base / fill(args.name_template, pdb)),
        }
        for pdb in ids
    ]


def fill(template: str, pdb: str) -> str:
    """Substitute one PDB ID into a path/name template, in whichever case it needs."""
    return template.format(pdb=pdb, PDB=pdb.upper(), pdb_low=pdb.lower())


# ================================= fan-out ==================================
# Added so one command can drive several GPUs: the parent re-runs this same script once per
# worker with --shard-index/--num-shards, then merges the per-shard summaries.

def launch_workers(targets: list[dict], args) -> None:
    """Run one child process per (GPU x --jobs-per-gpu) slot, wait, and merge summaries."""
    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()]
    n_workers = min(len(gpus) * args.jobs_per_gpu, len(targets))  # no empty workers
    log_dir = resolve(args.output_base) / "shards"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_path = resolve(args.summary)

    env = dict(os.environ) | thread_env(n_workers)
    print(f"fan-out: {len(targets)} targets over {n_workers} workers on GPUs {gpus} "
          f"({args.jobs_per_gpu}/GPU), threads/worker={env['OMP_NUM_THREADS']}")

    children = []
    for worker in range(n_workers):
        device = f"cuda:{gpus[worker % len(gpus)]}"
        log_path = log_dir / f"shard_{worker}.log"
        command = [sys.executable, "-u", str(Path(__file__).resolve()), *worker_argv(sys.argv[1:]),
                   "--shard-index", str(worker), "--num-shards", str(n_workers),
                   "--device", device, "--summary", str(summary_path)]
        print(f"  worker {worker} -> {device}, {len(targets[worker::n_workers])} targets, "
              f"log {log_path}")
        with log_path.open("w") as log:
            children.append(
                subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=env)
            )

    codes = [child.wait() for child in children]
    print(f"workers exited with {codes}")
    merge_summaries(summary_path, n_workers)


def worker_argv(argv: list[str]) -> list[str]:
    """This run's flags with the ones the parent sets per worker removed, in both the
    '--flag value' and '--flag=value' spellings, so children never re-fan-out."""
    parent_only = ("--gpus", "--jobs-per-gpu", "--device", "--summary")
    kept, skip_value = [], False
    for token in argv:
        if skip_value:
            skip_value = False
        elif token in parent_only:
            skip_value = True
        elif not any(token.startswith(flag + "=") for flag in parent_only):
            kept.append(token)
    return kept


def thread_env(n_workers: int) -> dict[str, str]:
    """Per-worker BLAS/OMP thread counts, divided from the cgroup CPU limit.

    On a HAMi vGPU profile os.cpu_count() reports the whole node, not the cgroup quota, so
    the thread pools oversubscribe and the run stalls -- that is what hung the first ens8
    attempt. Read /sys/fs/cgroup/cpu.max when it is there and fall back otherwise.

    An OMP_NUM_THREADS already in the environment wins. This function only knows its own
    worker count, so two fan-outs sharing a pod (e.g. one arm on GPUs 0-2, another on GPU 3)
    would each claim the whole quota; setting the variable by hand is how you split it."""
    if os.environ.get("OMP_NUM_THREADS"):
        threads = os.environ["OMP_NUM_THREADS"]
        return {"OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads,
                "OPENBLAS_NUM_THREADS": threads}

    cores = os.cpu_count() or n_workers
    try:
        quota, period = Path("/sys/fs/cgroup/cpu.max").read_text().split()
        if quota != "max":
            cores = float(quota) / float(period)
    except (OSError, ValueError):
        pass
    threads = str(max(1, int(cores // n_workers)))
    return {"OMP_NUM_THREADS": threads, "MKL_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads}


def shard_summary_path(path: Path, shard_index: int | None) -> Path:
    """batch_summary.json -> batch_summary_shard2.json, so workers never overwrite each other."""
    if shard_index is None:
        return path
    return path.with_name(f"{path.stem}_shard{shard_index}{path.suffix}")


def merge_summaries(summary_path: Path, n_workers: int) -> None:
    """Concatenate the per-shard summaries into the single --summary file."""
    merged: list[dict] = []
    for worker in range(n_workers):
        shard_path = shard_summary_path(summary_path, worker)
        if shard_path.exists():
            merged.extend(json.loads(shard_path.read_text()))
        else:
            print(f"WARNING: no summary from worker {worker} ({shard_path}) -- check its log")
    write_summary(merged, summary_path)


def print_plan(targets: list[dict], modes: list[str], args) -> None:
    """--dry-run: show what would run, where it reads from, and where it would be written."""
    n_workers = (
        min(len([g for g in args.gpus.split(",") if g.strip()]) * args.jobs_per_gpu, len(targets))
        if args.gpus else 1
    )
    print(f"PLAN: {len(targets)} targets x {len(modes)} modes = {len(targets) * len(modes)} runs, "
          f"{n_workers} worker(s), model={args.model} ensemble_size={args.ensemble_size}\n"
          f"modes={modes}  summary -> {resolve(args.summary)}")
    for index, target in enumerate(targets):
        print(f"\n[{index}] {target['name']}  worker={index % n_workers}  "
              f"res={target['resolution']}\n"
              f"     structure {mark(target['structure'])} {target['structure']}\n"
              f"     density   {mark(target['density'])} {target['density']}\n"
              f"     out       {resolve(target['out_dir'])}/<mode>/refined.cif")


def mark(path: str) -> str:
    """'ok' / 'MISSING' tag for one input path in the --dry-run plan."""
    return "ok     " if resolve(path).exists() else "MISSING"


# ================================= plumbing =================================

def guidance_for_mode(mode: str, args) -> tuple[GuidanceType, dict]:
    """Map a mode to its GuidanceType and the extra GuidanceConfig attributes it needs."""
    if mode == "baseline":
        return GuidanceType.PURE_GUIDANCE, {"step_scaler_type": "none"}
    if mode == "coord_guidance":
        # The paper's coordinate guidance normalizes the density gradient to the EDM denoising-update
        # magnitude (gradient_normalization) and applies AF3 augmentation + realign each step; only
        # then does step_size act as a fraction of the denoising step. These are mode-scoped so the
        # latent-opt arms are unaffected.
        return GuidanceType.PURE_GUIDANCE, {
            "step_scaler_type": "noisespace",
            "step_size": args.step_size,
            "gradient_normalization": args.gradient_normalization,
            "augmentation": args.augmentation,
        }
    if mode in ("s_only", "z_only", "s_plus_z"):
        which = {"s_only": "single", "z_only": "pair", "s_plus_z": "both"}[mode]
        return GuidanceType.LATENT_OPT, {
            "which_latent": which,
            "learning_rate": args.lr,
            "outer_steps": args.outer_steps,
            "anchor_weight": args.anchor,
            "max_grad_norm": 1.0,
            "bond_length_weight": args.bond_length_weight,
        }
    raise ValueError(f"unknown mode {mode!r}; pick from {ALL_MODES}")


def build_config(target: dict, guidance_type: GuidanceType, out_dir: Path, args) -> GuidanceConfig:
    config = GuidanceConfig(
        protein=target["name"],
        structure=str(resolve(target["structure"])),
        density=str(resolve(target["density"])),
        model_name=args.model,  # GuidanceConfig renamed this field from `model` (merge from main)
        guidance_type=guidance_type,
        log_path=str(out_dir / "run.log"),
        output_dir=str(out_dir),
        resolution=float(target["resolution"]),
        num_diffusion_steps=args.num_steps,
        guidance_start=args.guidance_start,  # -1 -> guide from step 0; e.g. 120 -> last low-noise steps
        align_to_input=True,
    )
    config.ensemble_size = args.ensemble_size  # set dynamically (not a declared GuidanceConfig field)
    return config


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", help=".csv (name,structure,density,resolution) or .json list of dicts")
    ap.add_argument("--output-base", default="it_opt_scratch/targets_out",
                    help="base output dir for CSV targets: <output-base>/<name>/<mode>/")
    ap.add_argument("--modes", default="baseline,z_only,coord_guidance",
                    help=f"comma-separated subset of {ALL_MODES}, or 'all'")

    ids = ap.add_argument_group("target list from PDB IDs (used when --targets is not given)")
    ids.add_argument("--proteins", help="comma-separated PDB IDs, e.g. 2YL0,5I09,5MC8")
    ids.add_argument("--proteins-file", dest="proteins_file",
                     help="file of PDB IDs, one per line ('#' comments allowed)")
    ids.add_argument("--structure-dir", dest="structure_dir",
                     default="/home/dev/test_data/processed",
                     help="root holding the input structures")
    ids.add_argument("--structure-template", dest="structure_template",
                     default="{PDB}/{PDB}_single_001_density_input.cif",
                     help="path under --structure-dir; takes {pdb}, {PDB}, {pdb_low}")
    ids.add_argument("--density-dir", dest="density_dir", default="",
                     help="directory holding the .ccp4 maps")
    ids.add_argument("--density-template", dest="density_template",
                     default="{PDB}_0.5occA_0.5occB_1.00A.ccp4",
                     help="filename under --density-dir; takes {pdb}, {PDB}, {pdb_low}")
    ids.add_argument("--name-template", dest="name_template", default="{PDB}",
                     help="target (and output dir) name, e.g. '{PDB}_0.5occA_0.5occB'")
    ids.add_argument("--resolution", type=float, default=1.0,
                     help="resolution for ID-built targets")
    ids.add_argument("--allow-missing", dest="allow_missing", action="store_true",
                     help="do not abort when a structure/density file is absent (e.g. staging "
                          "a run locally for pod paths)")

    fan = ap.add_argument_group("multi-GPU fan-out")
    fan.add_argument("--gpus", help="comma-separated GPU indices, e.g. 0,1,2,3; one child "
                                    "process per GPU x --jobs-per-gpu, then summaries are merged")
    fan.add_argument("--jobs-per-gpu", dest="jobs_per_gpu", type=int, default=1,
                     help="concurrent runs per GPU; 2 needs ~2x the weights resident, "
                          "so check VRAM")
    fan.add_argument("--shard-index", dest="shard_index", type=int, default=None,
                     help="set by the parent on each worker; targets[shard_index::num_shards]")
    fan.add_argument("--num-shards", dest="num_shards", type=int, default=1,
                     help="set by the parent on each worker")
    ap.add_argument("--dry-run", dest="dry_run", action="store_true",
                    help="print the resolved plan (paths, existence, shards) and exit")

    ap.add_argument("--model", default="protenix", choices=[m.value for m in StructurePredictor])
    ap.add_argument("--ensemble-size", dest="ensemble_size", type=int, default=4)
    ap.add_argument("--num-steps", dest="num_steps", type=int, default=200)
    ap.add_argument("--guidance-start", dest="guidance_start", type=int, default=-1,
                    help="step at which guidance begins (coord DPS and IT-opt both use it as a "
                         "fraction of num_steps); -1 means from step 0. e.g. 120 of 200 guides "
                         "only the last low-noise steps, matching the tuned coordinate-guidance recipe")
    ap.add_argument("--outer-steps", dest="outer_steps", type=int, default=2)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--anchor", type=float, default=0.0, help="on-manifold anchor weight (IT-opt)")
    ap.add_argument("--bond-length-weight", dest="bond_length_weight", type=float, default=0.0,
                    help="coordinate-space bond-geometry penalty weight (IT-opt); 0 disables")
    ap.add_argument("--step-size", dest="step_size", type=float, default=0.1, help="coord-guidance DPS step")
    ap.add_argument("--gradient-normalization", dest="gradient_normalization", action="store_true",
                    help="coord guidance: normalize the density gradient to the denoising-update "
                         "magnitude before scaling by --step-size (the paper's recipe; makes 0.1 mean "
                         "10%% of the denoising step rather than 0.1x the raw gradient)")
    ap.add_argument("--augmentation", action="store_true",
                    help="coord guidance: apply AF3 random augmentation + realign each step (paper recipe)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--summary", default=None,
                    help="batch summary JSON; defaults to <output-base>/batch_summary.json")

    args = ap.parse_args()
    if args.summary is None:  # keep the summary inside the run tree it describes
        args.summary = str(resolve(args.output_base) / "batch_summary.json")
    return args


def resolve(path: str) -> Path:
    """Absolute path as-is; relative path is taken relative to the repo root."""
    p = Path(path)
    return p if p.is_absolute() else REPO / p


def load_targets(path: str, output_base: Path) -> list[dict]:
    """Load targets from a .csv (name,structure,density,resolution) or a .json list of dicts.
    For CSV rows, out_dir defaults to <output_base>/<name>."""
    path = Path(path)
    if path.suffix.lower() == ".csv":
        return [
            {
                "name": row["name"],
                "structure": row["structure"],
                "density": row["density"],
                "resolution": float(row["resolution"]),
                "out_dir": str(output_base / row["name"]),
            }
            for row in csv.DictReader(path.read_text().splitlines())
        ]
    return json.loads(path.read_text())


def write_summary(summary: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(summary, indent=2))
    n_ok = sum(1 for s in summary if s.get("status") == "OK")
    print(f"\n===== DONE: {n_ok}/{len(summary)} runs OK. Summary -> {path} =====")


if __name__ == "__main__":
    main()
