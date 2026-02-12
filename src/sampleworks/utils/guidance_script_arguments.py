from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

from sampleworks.utils.checkpoint_utils import get_checkpoint
from sampleworks.utils.guidance_constants import (
    GuidanceType,
    StructurePredictor,
)


@dataclass
class GuidanceConfig:
    # TODO add a class method to set this up completely from args and job config.
    """
    Class to hold guidance config arguments, compatible with argparse, but which
    also can do some basic validation.
    """

    # add basic arguments by default.
    protein: str
    structure: str  # actually a path to a structure file
    density: str
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

    # DO NOT remove the **kwargs, it is for compatibility with argparse.
    def add_argument(self, name: str, default: Any = None, **kwargs):
        """Add an argument to the guidance config, in a form compatible with argparse"""
        setattr(self, name.lstrip("-").replace("-", "_"), default)

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
        self.model_checkpoint = get_checkpoint(self.model, args)
        if job.model == StructurePredictor.BOLTZ_2 and job.method:
            self.method = job.method

        if job.scaler == GuidanceType.FK_STEERING:
            self.guidance_weight = job.gradient_weight
            self.num_gd_steps = job.gd_steps
            self.num_particles = args.num_particles
            self.fk_lambda = args.fk_lambda
            self.fk_resampling_interval = args.fk_resampling_interval
            self.ensemble_size = job.ensemble_size
        else:
            self.step_size = job.gradient_weight
            self.use_tweedie = args.use_tweedie
            self.ensemble_size = job.ensemble_size


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
        "--use-tweedie",
        action="store_true",
        help="Use Tweedie's formula for gradient computation (enables augmentation/alignment)",
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
        default="~/.boltz/boltz2_conf.ckpt",
        help="Path to Boltz2 checkpoint",
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
        default=".pixi/envs/protenix-dev/lib/python3.12/site-packages/release_data/checkpoint/protenix_base_default_v0.5.0.pt",
        help="Path to Protenix checkpoint directory",
    )


def add_boltz1_specific_args(parser: argparse.ArgumentParser | GuidanceConfig):
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="~/.boltz/boltz1_conf.ckpt",
        help="Path to Boltz1 checkpoint",
    )


def add_rf3_specific_args(parser: argparse.ArgumentParser | GuidanceConfig):
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="~/.foundry/checkpoints/rf3_foundry_01_24_latest_remapped.ckpt",
        help="Path to RF3 checkpoint",
    )
    parser.add_argument(
        "--msa-path",
        type=str,
        default=None,
        help="Path to MSA file (dict, JSON, or .a3m format)",
    )


##############
#  Use these methods to parse arguments in scripts which load the model themselves.
##############
def parse_boltz2_pure_guidance_args():
    parser = argparse.ArgumentParser(
        description="Pure guidance refinement with Boltz-2 and real-space density"
    )
    add_generic_args(parser)
    add_boltz2_specific_args(parser)
    add_pure_guidance_args(parser)

    return parser.parse_args()


def parse_boltz1_pure_guidance_args():
    parser = argparse.ArgumentParser(
        description="Pure guidance refinement with Boltz-1 and real-space density"
    )
    add_generic_args(parser)
    add_boltz1_specific_args(parser)
    add_pure_guidance_args(parser)

    return parser.parse_args()


def parse_protenix_pure_guidance_args():
    parser = argparse.ArgumentParser(
        description="Pure guidance refinement with Protenix and real-space density"
    )
    add_generic_args(parser)
    add_protenix_specific_args(parser)
    add_pure_guidance_args(parser)

    return parser.parse_args()


def parse_protenix_fk_steering_args():
    parser = argparse.ArgumentParser(
        description="FK steering refinement with Protenix and real-space density"
    )
    add_protenix_specific_args(parser)
    add_generic_args(parser)
    add_fk_steering_args(parser)
    return parser.parse_args()


def parse_boltz2_fk_steering_args():
    parser = argparse.ArgumentParser(
        description="FK steering refinement with Boltz-2 and real-space density"
    )
    add_boltz2_specific_args(parser)
    add_generic_args(parser)
    add_fk_steering_args(parser)
    return parser.parse_args()


def parse_boltz1_fk_steering_args():
    parser = argparse.ArgumentParser(
        description="FK steering refinement with Boltz-1 and real-space density"
    )
    add_boltz1_specific_args(parser)
    add_generic_args(parser)
    add_fk_steering_args(parser)
    return parser.parse_args()


def parse_rf3_pure_guidance_args():
    parser = argparse.ArgumentParser(
        description="Pure guidance refinement with RF3 and real-space density"
    )
    add_generic_args(parser)
    add_rf3_specific_args(parser)
    add_pure_guidance_args(parser)
    return parser.parse_args()


def parse_rf3_fk_steering_args():
    parser = argparse.ArgumentParser(
        description="FK steering refinement with RF3 and real-space density"
    )
    add_generic_args(parser)
    add_rf3_specific_args(parser)
    add_fk_steering_args(parser)
    return parser.parse_args()


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
