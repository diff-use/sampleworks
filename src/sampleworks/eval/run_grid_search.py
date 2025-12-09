"""Grid search script for running experiments across models, scalers, and parameters."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import shutil
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from queue import Queue
from threading import Lock


logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)


@dataclass
class JobConfig:
    protein: str
    structure_path: str
    density_path: str
    resolution: float
    model: str
    scaler: str
    ensemble_size: int
    gradient_weight: float
    gd_steps: int
    method: str | None
    output_dir: str
    log_path: str


@dataclass
class JobResult:
    protein: str
    model: str
    method: str | None
    scaler: str
    ensemble_size: int
    gradient_weight: float
    gd_steps: int
    status: str
    exit_code: int
    runtime_seconds: float
    started_at: str
    finished_at: str
    log_path: str
    output_dir: str


@dataclass
class GridSearchConfig:
    models: list[str]
    scalers: list[str]
    ensemble_sizes: list[int]
    gradient_weights: list[float]
    gd_steps: list[int]
    methods: list[str]
    proteins_file: str
    output_dir: str


def get_job_status(job: JobConfig) -> str:
    """
    Check the status of a job by inspecting its log file.

    Returns:
        'success': Job completed successfully (has "Final loss:" in log)
        'failed': Job ran but failed (has errors/traceback in log or exit != 0)
        'not_run': Job has not been executed yet (no log file)
    """
    if not os.path.exists(job.log_path):
        return "not_run"

    try:
        with open(job.log_path) as f:
            log_content = f.read()

        has_error = (
            "Traceback" in log_content
            or "AssertionError" in log_content
            or "Error:" in log_content
        )
        if has_error:
            return "failed"

        if "Final loss:" in log_content:
            return "success"

        return "failed"
    except Exception as e:
        log.warning(f"Error reading log file {job.log_path}: {e}")
        return "failed"


def detect_gpus() -> list[str]:
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible:
        return [g.strip() for g in cuda_visible.split(",") if g.strip()]
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            return [g.strip() for g in result.stdout.strip().split("\n") if g.strip()]
    except FileNotFoundError:
        pass
    return ["0"]


def get_pixi_env(model: str) -> str:
    if model in ("boltz1", "boltz2"):
        return "boltz"
    elif model == "protenix":
        return "protenix"
    else:
        raise ValueError(f"Unknown model: {model}")


def get_checkpoint(model: str, args: argparse.Namespace) -> str | None:
    if model == "boltz1":
        return args.boltz1_checkpoint
    elif model == "boltz2":
        return args.boltz2_checkpoint
    elif model == "protenix":
        return args.protenix_checkpoint
    return None


def build_command(job: JobConfig, args: argparse.Namespace) -> list[str]:
    env = get_pixi_env(job.model)
    script_dir = Path(__file__).parent
    script_name = f"{job.model}_{job.scaler}.py"
    script_path = script_dir / "examples" / script_name

    cmd = ["pixi", "run", "-e", env, "python", str(script_path)]

    cmd.extend(["--ensemble-size", str(job.ensemble_size)])
    cmd.extend(["--structure", job.structure_path])
    cmd.extend(["--density", job.density_path])
    cmd.extend(["--resolution", str(job.resolution)])
    cmd.extend(["--output-dir", job.output_dir])
    cmd.extend(["--partial-diffusion-step", str(args.partial_diffusion_step)])
    cmd.extend(["--loss-order", str(args.loss_order)])
    cmd.extend(["--device", "cuda:0"])

    checkpoint = get_checkpoint(job.model, args)
    if checkpoint:
        cmd.extend(["--model-checkpoint", checkpoint])

    if job.model == "boltz2" and job.method:
        cmd.extend(["--method", job.method])

    if job.scaler == "fk_steering":
        cmd.extend(["--guidance-weight", str(job.gradient_weight)])
        cmd.extend(["--num-gd-steps", str(job.gd_steps)])
        cmd.extend(["--num-particles", str(args.num_particles)])
        cmd.extend(["--fk-lambda", str(args.fk_lambda)])
        cmd.extend(["--fk-resampling-interval", str(args.fk_resampling_interval)])
    else:
        cmd.extend(["--step-size", str(job.gradient_weight)])
        if args.use_tweedie:
            cmd.append("--use-tweedie")

    if args.gradient_normalization:
        cmd.append("--gradient-normalization")
    if args.augmentation:
        cmd.append("--augmentation")
    if args.align_to_input:
        cmd.append("--align-to-input")

    return cmd


def run_job(
    job: JobConfig, gpu: str, args: argparse.Namespace, clean_output: bool = False
) -> JobResult:
    started_at = datetime.now().isoformat()
    start_time = time.time()

    if clean_output and os.path.exists(job.output_dir):
        log.info(f"Cleaning existing output directory: {job.output_dir}")
        shutil.rmtree(job.output_dir)

    os.makedirs(os.path.dirname(job.log_path), exist_ok=True)

    cmd = build_command(job, args)
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu

    log.info(f"Starting on GPU {gpu}: {job.log_path}")

    with open(job.log_path, "w") as log_file:
        result = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=env)

    runtime = time.time() - start_time
    finished_at = datetime.now().isoformat()

    status = "success" if result.returncode == 0 else "failed"

    return JobResult(
        protein=job.protein,
        model=job.model,
        method=job.method,
        scaler=job.scaler,
        ensemble_size=job.ensemble_size,
        gradient_weight=job.gradient_weight,
        gd_steps=job.gd_steps,
        status=status,
        exit_code=result.returncode,
        runtime_seconds=round(runtime, 2),
        started_at=started_at,
        finished_at=finished_at,
        log_path=job.log_path,
        output_dir=job.output_dir,
    )


def generate_jobs(args: argparse.Namespace) -> list[JobConfig]:
    jobs = []

    with open(args.proteins, newline="") as f:
        reader = csv.DictReader(f)
        proteins = list(reader)

    models = args.models.split()
    scalers = args.scalers.split()
    ensemble_sizes = [int(x) for x in args.ensemble_sizes.split()]
    gradient_weights = [float(x) for x in args.gradient_weights.split()]
    gd_steps_list = [int(x) for x in args.num_gd_steps.split()]
    methods = [m.strip() for m in args.methods.split(",")]

    for protein in proteins:
        structure = protein["structure"].strip()
        density = protein["density"].strip()
        resolution = float(protein["resolution"].strip())
        protein_name = protein["name"].strip()

        for model in models:
            model_methods = methods if model == "boltz2" else [None]

            for method in model_methods:
                method_suffix = f"_{method.replace(' ', '_')}" if method else ""

                for scaler in scalers:
                    if scaler == "fk_steering":
                        for ens in ensemble_sizes:
                            for gw in gradient_weights:
                                for gd in gd_steps_list:
                                    output_dir = os.path.join(
                                        args.output_dir,
                                        protein_name,
                                        f"{model}{method_suffix}",
                                        scaler,
                                        f"ens{ens}_gw{gw}_gd{gd}",
                                    )
                                    log_path = os.path.join(output_dir, "run.log")
                                    jobs.append(
                                        JobConfig(
                                            protein=protein_name,
                                            structure_path=structure,
                                            density_path=density,
                                            resolution=resolution,
                                            model=model,
                                            scaler=scaler,
                                            ensemble_size=ens,
                                            gradient_weight=gw,
                                            gd_steps=gd,
                                            method=method,
                                            output_dir=output_dir,
                                            log_path=log_path,
                                        )
                                    )
                    else:
                        for ens in ensemble_sizes:
                            for gw in gradient_weights:
                                output_dir = os.path.join(
                                    args.output_dir,
                                    protein_name,
                                    f"{model}{method_suffix}",
                                    scaler,
                                    f"ens{ens}_gw{gw}",
                                )
                                log_path = os.path.join(output_dir, "run.log")
                                jobs.append(
                                    JobConfig(
                                        protein=protein_name,
                                        structure_path=structure,
                                        density_path=density,
                                        resolution=resolution,
                                        model=model,
                                        scaler=scaler,
                                        ensemble_size=ens,
                                        gradient_weight=gw,
                                        gd_steps=1,
                                        method=method,
                                        output_dir=output_dir,
                                        log_path=log_path,
                                    )
                                )

    return jobs


print_lock = Lock()


def worker_wrapper(args_tuple: tuple) -> JobResult:
    job, gpu, args, clean_output = args_tuple
    return run_job(job, gpu, args, clean_output)


def run_grid_search(
    jobs: list[JobConfig],
    gpus: list[str],
    args: argparse.Namespace,
    job_statuses: dict[int, str] | None = None,
) -> list[JobResult]:
    results: list[JobResult] = []
    successful = 0
    failed = 0

    if args.dry_run:
        for job in jobs:
            cmd = build_command(job, args)
            log.info(f"[DRY-RUN] {' '.join(cmd)}")
        return results

    gpu_queue: Queue[str] = Queue()
    for gpu in gpus:
        gpu_queue.put(gpu)

    max_workers = len(gpus)
    log.info(f"Running {len(jobs)} jobs with {max_workers} parallel workers")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        pending_jobs = list(jobs)
        active_futures = set()

        while pending_jobs or active_futures:
            while pending_jobs and not gpu_queue.empty():
                job = pending_jobs.pop(0)
                gpu = gpu_queue.get()

                clean_output = False
                if job_statuses is not None:
                    job_status = job_statuses.get(id(job), "not_run")
                    clean_output = job_status != "not_run"

                future = executor.submit(worker_wrapper, (job, gpu, args, clean_output))
                futures[future] = gpu
                active_futures.add(future)

            if active_futures:
                done_futures = set()
                for future in active_futures:
                    if future.done():
                        done_futures.add(future)

                for future in done_futures:
                    active_futures.remove(future)
                    gpu = futures[future]
                    gpu_queue.put(gpu)

                    try:
                        result = future.result()
                        results.append(result)
                        if result.status == "success":
                            successful += 1
                            log.info(
                                f"SUCCESS (GPU {gpu}, {result.runtime_seconds:.1f}s): "
                                f"{result.log_path}"
                            )
                        else:
                            failed += 1
                            log.info(
                                f"FAILED (GPU {gpu}, exit={result.exit_code}): "
                                f"{result.log_path}"
                            )
                    except Exception as e:
                        failed += 1
                        log.error(f"Job failed with exception: {e}")

                if not done_futures and active_futures:
                    time.sleep(1)

    log.info(f"Completed: {successful} successful, {failed} failed")
    return results


def save_results(
    results: list[JobResult],
    config: GridSearchConfig,
    output_dir: str,
    total_time: float,
):
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.json")

    existing_runs = []
    if os.path.exists(results_path):
        try:
            with open(results_path) as f:
                existing_data = json.load(f)
                existing_runs = existing_data.get("runs", [])
            log.info(f"Loaded {len(existing_runs)} existing results")
        except Exception as e:
            log.warning(f"Could not load existing results: {e}")

    new_run_keys = {
        (
            r.protein,
            r.model,
            r.method,
            r.scaler,
            r.ensemble_size,
            r.gradient_weight,
            r.gd_steps,
        )
        for r in results
    }

    merged_runs = [asdict(r) for r in results]
    for existing_run in existing_runs:
        key = (
            existing_run.get("protein"),
            existing_run.get("model"),
            existing_run.get("method"),
            existing_run.get("scaler"),
            existing_run.get("ensemble_size"),
            existing_run.get("gradient_weight"),
            existing_run.get("gd_steps"),
        )
        if key not in new_run_keys:
            merged_runs.append(existing_run)

    output = {
        "config": asdict(config),
        "runs": merged_runs,
        "summary": {
            "total": len(merged_runs),
            "successful": sum(1 for r in merged_runs if r.get("status") == "success"),
            "failed": sum(1 for r in merged_runs if r.get("status") == "failed"),
            "total_runtime_seconds": round(total_time, 2),
        },
    }

    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    log.info(f"Results saved to {results_path} ({len(merged_runs)} total runs)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run grid search across models, scalers, and parameters."
    )

    parser.add_argument(
        "--proteins",
        required=True,
        help="CSV file with columns: structure,density,resolution,name",
    )

    parser.add_argument(
        "--models", default="boltz2 protenix", help="Space-separated models"
    )
    parser.add_argument(
        "--scalers", default="pure_guidance fk_steering", help="Space-separated scalers"
    )
    parser.add_argument(
        "--ensemble-sizes", default="1 2 4 8", help="Space-separated ensemble sizes"
    )
    parser.add_argument(
        "--gradient-weights",
        default="0.01 0.1 0.2",
        help="Space-separated gradient weights",
    )
    parser.add_argument(
        "--num-gd-steps",
        default="20",
        help="Space-separated GD steps (FK steering only)",
    )
    parser.add_argument(
        "--output-dir", default="./grid_search_results", help="Output directory"
    )

    parser.add_argument(
        "--boltz1-checkpoint",
        default=os.path.expanduser("~/.boltz/boltz1_conf.ckpt"),
        help="Boltz1 checkpoint path",
    )
    parser.add_argument(
        "--boltz2-checkpoint",
        default=os.path.expanduser("~/.boltz/boltz2_conf.ckpt"),
        help="Boltz2 checkpoint path",
    )
    parser.add_argument(
        "--protenix-checkpoint", default="", help="Protenix checkpoint path"
    )
    parser.add_argument(
        "--methods",
        default="MD,X-RAY DIFFRACTION",
        help="Comma-separated methods for Boltz2",
    )

    parser.add_argument(
        "--num-particles", type=int, default=3, help="FK steering: num particles"
    )
    parser.add_argument(
        "--fk-lambda", type=float, default=0.5, help="FK steering: lambda"
    )
    parser.add_argument(
        "--fk-resampling-interval",
        type=int,
        default=1,
        help="FK steering: resampling interval",
    )

    parser.add_argument(
        "--partial-diffusion-step", type=int, default=0, help="Partial diffusion step"
    )
    parser.add_argument(
        "--loss-order", type=int, default=2, help="L1 (1) or L2 (2) loss"
    )
    parser.add_argument(
        "--use-tweedie", action="store_true", help="Use Tweedie (pure guidance)"
    )
    parser.add_argument(
        "--gradient-normalization",
        action="store_true",
        help="Enable gradient normalization",
    )
    parser.add_argument(
        "--augmentation", action="store_true", help="Enable augmentation"
    )
    parser.add_argument(
        "--align-to-input", action="store_true", help="Align to input structure"
    )

    parser.add_argument(
        "--max-parallel",
        default="auto",
        help="Max parallel jobs (default: auto = number of GPUs)",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Print commands without executing"
    )

    parser.add_argument(
        "--force-all",
        action="store_true",
        help="Re-run all jobs, including successful ones (overrides default)",
    )
    parser.add_argument(
        "--only-failed",
        action="store_true",
        help="Run only failed jobs, skip un-run and successful jobs",
    )
    parser.add_argument(
        "--only-missing",
        action="store_true",
        help="Run only un-run jobs, skip failed and successful jobs",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    gpus = detect_gpus()
    if args.max_parallel != "auto":
        gpus = gpus[: int(args.max_parallel)]

    log.info("=" * 50)
    log.info("Starting grid search")
    log.info(f"Models: {args.models}")
    log.info(f"Scalers: {args.scalers}")
    log.info(f"Ensemble sizes: {args.ensemble_sizes}")
    log.info(f"Gradient weights: {args.gradient_weights}")
    log.info(f"GD steps: {args.num_gd_steps}")
    log.info(f"Boltz2 methods: {args.methods}")
    log.info(f"Output directory: {args.output_dir}")
    log.info(f"GPUs: {gpus}")
    log.info(f"Dry run: {args.dry_run}")
    log.info("=" * 50)

    jobs = generate_jobs(args)
    log.info(f"Generated {len(jobs)} total jobs")

    log.info("Checking job statuses...")
    job_statuses = {}
    for job in jobs:
        status = get_job_status(job)
        job_statuses[id(job)] = status

    successful_count = sum(1 for s in job_statuses.values() if s == "success")
    failed_count = sum(1 for s in job_statuses.values() if s == "failed")
    not_run_count = sum(1 for s in job_statuses.values() if s == "not_run")

    log.info(
        f"Status: {successful_count} successful, {failed_count} failed, "
        f"{not_run_count} not run"
    )

    if args.force_all:
        filtered_jobs = jobs
        log.info("Running all jobs (--force-all)")
    elif args.only_failed:
        filtered_jobs = [job for job in jobs if job_statuses[id(job)] == "failed"]
        log.info(f"Running only failed jobs (--only-failed): {len(filtered_jobs)} jobs")
    elif args.only_missing:
        filtered_jobs = [job for job in jobs if job_statuses[id(job)] == "not_run"]
        log.info(
            f"Running only un-run jobs (--only-missing): {len(filtered_jobs)} jobs"
        )
    else:
        filtered_jobs = [
            job for job in jobs if job_statuses[id(job)] in ("failed", "not_run")
        ]
        log.info(f"Running failed and un-run jobs (default): {len(filtered_jobs)} jobs")

    if len(filtered_jobs) == 0:
        log.info("No jobs to run!")
        return

    config = GridSearchConfig(
        models=args.models.split(),
        scalers=args.scalers.split(),
        ensemble_sizes=[int(x) for x in args.ensemble_sizes.split()],
        gradient_weights=[float(x) for x in args.gradient_weights.split()],
        gd_steps=[int(x) for x in args.num_gd_steps.split()],
        methods=[m.strip() for m in args.methods.split(",")],
        proteins_file=args.proteins,
        output_dir=args.output_dir,
    )

    start_time = time.time()
    results = run_grid_search(filtered_jobs, gpus, args, job_statuses)
    total_time = time.time() - start_time

    if not args.dry_run and results:
        save_results(results, config, args.output_dir, total_time)

    log.info("=" * 50)
    log.info("Grid search complete")
    log.info("=" * 50)


if __name__ == "__main__":
    main()
