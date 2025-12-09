import argparse

def add_generic_args(parser: argparse.ArgumentParser):
    parser.add_argument("--structure", type=str, required=True, help="Input structure")
    parser.add_argument("--density", type=str, required=True, help="Input density map")
    parser.add_argument(
        "--output-dir", type=str, default="output", help="Output directory"
    )
    parser.add_argument(
        "--partial-diffusion-step",
        type=int,
        default=0,
        help="Diffusion step to start from",
    )
    parser.add_argument(
        "--loss-order", type=int, default=2, choices=[1, 2], help="L1 or L2 loss"
    )
    parser.add_argument(
        "--resolution",
        type=float,
        required=True,
        help="Map resolution in Angstroms (required for CCP4/MRC/MAP)",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="Device (cuda/cpu, auto-detect)"
    )
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


######################
# Guidance type specific arguments
######################
def add_pure_guidance_args(parser: argparse.ArgumentParser):
    parser.add_argument("--step-size", type=float, default=0.1, help="Gradient step")
    parser.add_argument(
        "--use-tweedie",
        action="store_true",
        help="Use Tweedie's formula for gradient computation "
             "(enables augmentation/alignment)",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=1,
        help="Number of ensemble members to generate",
    )


def add_fk_steering_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--num-particles",
        type=int,
        default=3,
        help="Number of particles for FK steering",
    )
    parser.add_argument(
        "--ensemble-size",
        type=int,
        default=4,
        help="Ensemble size per particle",
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
def add_boltz2_specific_args(parser: argparse.ArgumentParser):
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

def add_protenix_specific_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default=".pixi/envs/protenix-dev/lib/python3.12/site-packages/release_data/checkpoint/protenix_base_default_v0.5.0.pt",
        help="Path to Protenix checkpoint directory",
    )


def add_boltz1_specific_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--model-checkpoint",
        type=str,
        default="~/.boltz/boltz1_conf.ckpt",
        help="Path to Boltz1 checkpoint",
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
