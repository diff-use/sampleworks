from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sampleworks.utils.guidance_constants import GuidanceType, StructurePredictor


# Baked-in checkpoint paths (Docker image) with legacy fallbacks
_CHECKPOINT_CANDIDATES = {
    "boltz1": ["/checkpoints/boltz1_conf.ckpt", "~/.boltz/boltz1_conf.ckpt"],
    "boltz2": ["/checkpoints/boltz2_conf.ckpt", "~/.boltz/boltz2_conf.ckpt"],
    "rf3": [
        "/checkpoints/rf3_foundry_01_24_latest.ckpt",
        "~/.foundry/checkpoints/rf3_foundry_01_24_latest.ckpt",
    ],
    "protenix": [
        "/checkpoints/protenix_base_default_v0.5.0.pt",
        ".pixi/envs/protenix-dev/lib/python3.12/site-packages/release_data/checkpoint/protenix_base_default_v0.5.0.pt",
    ],
}


def _resolve_checkpoint(model_key: str) -> str:
    """Return the first checkpoint path that exists on disk for *model_key*.

    Tries baked-in Docker paths first (``/checkpoints/``), then falls back to
    legacy development paths.  If none are found the first candidate is returned
    so that downstream validation produces a clear error message.
    """
    candidates = _CHECKPOINT_CANDIDATES.get(model_key, [])
    for candidate in candidates:
        resolved = Path(candidate).expanduser()
        if resolved.exists():
            return str(resolved)
    # Nothing found – return the primary (baked-in) path so the error message
    # points the user to the expected location.
    resolved = candidates[0] if candidates else ""
    if not resolved:
        raise ValueError(
            f"Running guidance requires a model checkpoint for '{model_key}'. "
            f"Provide --model-checkpoint or bake checkpoints into /checkpoints/."
        )
    if not Path(resolved).exists():
        raise ValueError(
            f"Model checkpoint '{resolved}' does not exist. "
            f"Provide a valid path via --model-checkpoint."
        )

    return resolved


def get_checkpoint(args: argparse.Namespace) -> str | None:
    """Resolve a model checkpoint path from an argparse namespace.

    Looks for a ``model_checkpoint`` attribute on *args*.
    Empty strings are treated as missing values.
    """
    value = getattr(args, "model_checkpoint", None)
    if value is not None and str(value).strip() != "":
        return str(value)

    return None


def validate_model_checkpoint(
    model: str | StructurePredictor,
    checkpoint: str | Path | None,
) -> str:
    """Validate and normalize the checkpoint path for ``model``.

    When *checkpoint* is ``None`` (no ``--model-checkpoint`` provided), the
    function auto-resolves by checking baked-in Docker paths first
    (``/checkpoints/``) and then legacy development paths.

    Returns
    -------
    str
        Absolute checkpoint path.

    Raises
    ------
    ValueError
        If checkpoint points to a directory.
    FileNotFoundError
        If checkpoint does not exist on disk.
    """
    # Auto-resolve when no explicit checkpoint was provided
    if checkpoint is None or str(checkpoint).strip() == "":
        model_key = str(model).lower().replace("structurepredictor.", "")
        resolved = _resolve_checkpoint(model_key)
        if not resolved:
            raise ValueError(
                f"Missing checkpoint for model '{model}'. "
                f"Provide --model-checkpoint or bake checkpoints into /checkpoints/."
            )
        checkpoint = resolved

    checkpoint_path = Path(str(checkpoint)).expanduser().resolve()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint for model '{model}' does not exist: {checkpoint_path}. "
            f"Provide a valid path via --model-checkpoint."
        )

    if not checkpoint_path.is_file():
        raise ValueError(f"Checkpoint for model '{model}' must be a file, got: {checkpoint_path}")

    return str(checkpoint_path)


_MODEL_ARG_ADDERS: dict[str, Any] = {
    "boltz1": lambda p: add_boltz1_specific_args(p),
    "boltz2": lambda p: add_boltz2_specific_args(p),
    "protenix": lambda p: add_protenix_specific_args(p),
    "rf3": lambda p: add_rf3_specific_args(p),
}

_GUIDANCE_ARG_ADDERS: dict[str, Any] = {
    "pure_guidance": lambda p: add_pure_guidance_args(p),
    "fk_steering": lambda p: add_fk_steering_args(p),
}

# Attributes set dynamically by add_*_args helpers that should be copied
# from a parsed argparse.Namespace onto a GuidanceConfig instance.
_DYNAMIC_ATTRS = [
    # pure guidance
    "step_size",
    "step_scaler_type",
    # fk steering
    "num_particles",
    "fk_resampling_interval",
    "fk_lambda",
    "num_gd_steps",
    "guidance_weight",
    "guidance_interval",
    # model-specific
    "model_checkpoint",
    "method",
    "msa_path",
    "disable_chiral_features",
    "track_chiral_features",
    # generic (overridable)
    "ensemble_size",
]


@dataclass
class GuidanceConfig:
    """
    Class to hold guidance config arguments, compatible with argparse, but which
    also can do some basic validation.
    """

    # add basic arguments by default.
    protein: str
    structure: Path | str  # actually a path to a structure file
    density: Path | str
    model: str | StructurePredictor
    guidance_type: str | GuidanceType
    log_path: str
    output_dir: str = "output"
    partial_diffusion_step: int = 0
    loss_order: int = 2
    resolution: float | None = None
    device: str = ""
    gradient_normalization: bool = False
    em: bool = False
    guidance_start: int = -1
    augmentation: bool = False
    align_to_input: bool = False
    recycling_steps: int | None = None
    num_diffusion_steps: int = 200

    # DO NOT remove the **kwargs, it is for compatibility with argparse.
    def add_argument(self, name: str, default: Any = None, **kwargs):
        """Add an argument to the guidance config, in a form compatible with argparse"""
        setattr(self, name.lstrip("-").replace("-", "_"), default)

    @classmethod
    def from_cli(
        cls,
        argv: list[str] | None = None,
        model: str | None = None,
        guidance_type: str | None = None,
    ) -> GuidanceConfig:
        """Parse CLI arguments and return a fully populated GuidanceConfig.

        When *model* and *guidance_type* are provided (e.g. from legacy
        scripts), they are used directly and ``--model`` / ``--guidance-type``
        are not required on the command line.  Otherwise they are parsed as
        required CLI arguments.
        """
        model_choices = [m.value for m in StructurePredictor]
        guidance_choices = [g.value for g in GuidanceType]
        model_preset = model is not None
        guidance_preset = guidance_type is not None

        # -- first pass: resolve model & guidance_type if not pre-set --------
        if not model_preset or not guidance_preset:
            pre = argparse.ArgumentParser(add_help=False)
            if not model_preset:
                pre.add_argument(
                    "--model",
                    type=str,
                    required=True,
                    choices=model_choices,
                    help="Structure prediction model",
                )
            if not guidance_preset:
                pre.add_argument(
                    "--guidance-type",
                    type=str,
                    required=True,
                    choices=guidance_choices,
                    help="Guidance method",
                )
            pre_args, _ = pre.parse_known_args(argv)
            model = model or pre_args.model
            guidance_type = guidance_type or pre_args.guidance_type

        # -- full parser -----------------------------------------------------
        parser = argparse.ArgumentParser(
            description=f"Run {guidance_type} guidance with {model}",
        )
        if not model_preset:
            parser.add_argument(
                "--model",
                type=str,
                default=model,
                choices=model_choices,
                help="Structure prediction model",
            )
        if not guidance_preset:
            parser.add_argument(
                "--guidance-type",
                type=str,
                default=guidance_type,
                choices=guidance_choices,
                help="Guidance method",
            )
        parser.add_argument(
            "--protein",
            type=str,
            required=True,
            help="Protein identifier (must match naming used in grid search / evaluation)",
        )
        add_generic_args(parser)
        _MODEL_ARG_ADDERS[model](parser)
        _GUIDANCE_ARG_ADDERS[guidance_type](parser)

        args = parser.parse_args(argv)

        config = cls(
            protein=args.protein,
            structure=args.structure,
            density=args.density,
            model=model,
            guidance_type=guidance_type,
            log_path=getattr(args, "log_path", None) or "",
            output_dir=args.output_dir,
            partial_diffusion_step=args.partial_diffusion_step,
            loss_order=args.loss_order,
            resolution=args.resolution,
            device=getattr(args, "device", "") or "",
            gradient_normalization=args.gradient_normalization,
            em=args.em,
            guidance_start=args.guidance_start,
            augmentation=args.augmentation,
            align_to_input=args.align_to_input,
        )

        # __post_init__ already set defaults for model/guidance-specific
        # attrs; override with any explicit CLI values.
        for attr in _DYNAMIC_ATTRS:
            val = getattr(args, attr, None)
            if val is not None:
                setattr(config, attr, val)

        return config

    def __post_init__(self):
        """Set up guidance config for a given model and guidance type"""
        if self.guidance_type == GuidanceType.PURE_GUIDANCE:
            add_pure_guidance_args(self)
        elif self.guidance_type == GuidanceType.FK_STEERING:
            add_fk_steering_args(self)
        else:
            raise ValueError(f"Unknown guidance type: {self.guidance_type}")

        if self.model == StructurePredictor.BOLTZ_1:
            add_boltz1_specific_args(self)
        elif self.model == StructurePredictor.BOLTZ_2:
            add_boltz2_specific_args(self)
        elif self.model == StructurePredictor.PROTENIX:
            add_protenix_specific_args(self)
        elif self.model == StructurePredictor.RF3:
            add_rf3_specific_args(self)
        else:
            raise ValueError(f"Unknown model type: {self.model}")

    def populate_config_for_guidance_type(self, job: JobConfig, args: argparse.Namespace):
        checkpoint = get_checkpoint(args)
        if checkpoint is not None:
            self.model_checkpoint = checkpoint
        elif not getattr(self, "model_checkpoint", None):
            # Auto-resolve from baked-in /checkpoints/ or legacy fallback paths
            model_key = str(self.model).lower().replace("structurepredictor.", "")
            self.model_checkpoint = _resolve_checkpoint(model_key)

        if job.model == StructurePredictor.BOLTZ_2 and job.method:
            self.method = job.method

        if job.model == StructurePredictor.RF3:
            self.disable_chiral_features = getattr(args, "disable_chiral_features", False)
            self.track_chiral_features = getattr(args, "track_chiral_features", False)

        if job.scaler == GuidanceType.FK_STEERING:
            self.guidance_weight = job.gradient_weight
            self.num_gd_steps = job.gd_steps
            self.num_particles = args.num_particles
            self.fk_lambda = args.fk_lambda
            self.fk_resampling_interval = args.fk_resampling_interval
            self.ensemble_size = job.ensemble_size
        else:
            self.step_size = job.gradient_weight
            self.step_scaler_type = args.step_scaler_type
            self.ensemble_size = job.ensemble_size

    def as_dict(self) -> dict[str, Any]:
        """Return a dictionary representation of the guidance config, converting Path to strings"""
        output = self.__dict__.copy()
        output["density"] = str(self.density)
        output["structure"] = str(self.structure)
        return output


def add_generic_args(parser: argparse.ArgumentParser | GuidanceConfig):
    parser.add_argument("--structure", type=str, required=True, help="Input structure")
    parser.add_argument("--density", type=str, required=True, help="Input density map")
    parser.add_argument("--output-dir", type=str, default="output", help="Output directory")
    parser.add_argument(
        "--log-path", type=str, default=None, help="Log file path (default: output-dir/run.log)"
    )
    parser.add_argument(
        "--partial-diffusion-step",
        type=int,
        default=0,
        help="Diffusion step to start from",
    )
    parser.add_argument("--loss-order", type=int, default=2, choices=[1, 2], help="L1 or L2 loss")
    parser.add_argument(
        "--resolution",
        type=float,
        required=True,
        help="Map resolution in Angstroms (required for CCP4/MRC/MAP)",
    )
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu, auto-detect)")
    parser.add_argument(
        "--gradient-normalization",
        action="store_true",
        help="Enable gradient normalization",
    )
    parser.add_argument("--em", action="store_true", help="Use EM scattering factors")
    parser.add_argument(
        "--guidance-start",
        type=int,
        default=-1,
        help="Step to start guidance (default: -1, starts immediately)",
    )
    parser.add_argument(
        "--augmentation",
        action="store_true",
        help="Enable data augmentation",
    )
    parser.add_argument(
        "--align-to-input",
        action="store_true",
        help="Enable alignment to input",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=4,
        help="Ensemble size to generate (per particle for FK-steering)",
    )


######################
# Guidance type specific arguments
######################
def add_pure_guidance_args(parser: argparse.ArgumentParser | GuidanceConfig):
    parser.add_argument("--step-size", type=float, default=0.1, help="Gradient step")
    parser.add_argument(
        "--step-scaler-type",
        type=str,
        default="noisespace",
        choices=["dataspace", "noisespace", "none"],
        help="Type of step scaler to use: dataspace (DataSpaceDPSScaler), noisespace "
        "(NoiseSpaceDPSScaler), or none (NoScalingScaler)",
    )


def add_fk_steering_args(parser: argparse.ArgumentParser | GuidanceConfig):
    parser.add_argument(
        "--num-particles",
        type=int,
        default=3,
        help="Number of particles for FK steering",
    )
    parser.add_argument(
        "--fk-resampling-interval",
        type=int,
        default=1,
        help="How often to apply resampling",
    )
    parser.add_argument(
        "--fk-lambda",
        type=float,
        default=1.0,
        help="Weighting factor for resampling",
    )
    parser.add_argument(
        "--num-gd-steps",
        type=int,
        default=1,
        help="Number of gradient descent steps on x0",
    )
    parser.add_argument(
        "--guidance-weight",
        type=float,
        default=0.01,
        help="Weight for gradient descent guidance",
    )
    parser.add_argument(
        "--guidance-interval",
        type=int,
        default=1,
        help="How often to apply guidance",
    )


###########
# Model specific arguments
###########
def add_boltz2_specific_args(parser: argparse.ArgumentParser | GuidanceConfig):
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=None,
        help="Path to Boltz2 checkpoint (default: auto-resolved from /checkpoints/ or ~/.boltz/)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="X-RAY DIFFRACTION",
        help="Boltz2 sampling method",
    )


def add_protenix_specific_args(parser: argparse.ArgumentParser | GuidanceConfig):
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=None,
        help="Path to Protenix checkpoint (default: auto-resolved from /checkpoints/ or pixi env)",
    )


def add_boltz1_specific_args(parser: argparse.ArgumentParser | GuidanceConfig):
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=None,
        help="Path to Boltz1 checkpoint (default: auto-resolved from /checkpoints/ or ~/.boltz/)",
    )


def add_rf3_specific_args(parser: argparse.ArgumentParser | GuidanceConfig):
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=None,
        help="Path to RF3 checkpoint (default: auto-resolved from /checkpoints/ or ~/.foundry/)",
    )
    parser.add_argument(
        "--msa-path",
        type=str,
        default=None,
        help="Path to MSA file (dict, JSON, or .a3m format)",
    )
    parser.add_argument(
        "--disable-chiral-features",
        action="store_true",
        help="Disable RF3 chiral gradient feature during guidance",
    )
    parser.add_argument(
        "--track-chiral-features",
        action="store_true",
        help="Log chiral gradient statistics at each denoising step",
    )


@dataclass
class JobConfig:
    protein: str
    structure_path: Path | str
    density_path: Path | str
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
