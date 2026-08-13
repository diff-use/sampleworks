"""Flattened walkthrough of the sampleworks guidance pipeline using Boltz-2.

Adapted from the Protpardelle debugging script (``debugging-ppdl-in-sw.py``) by
Marcus Collins, modified here for use with Boltz-2.

It inlines, in order, what ``run_guidance()`` normally does behind three layers
of call:

    PureGuidance.sample()  ->  AF3EDMSampler.step()  ->  Boltz2Wrapper.step()

Every intermediate (features, prior coords, reconciler, schedule, per-step
context, noisy state, denoised prediction) is left in module scope so the file
can be run top-to-bottom or pasted into an IPython session for inspection.

Run with:

    pixi run -e boltz python debugging-boltz-in-sw.py

Inputs default to the 1VME resources checked into the repo, matching the Boltz-2
example in README.md. Override the device with ``SW_DEVICE=cpu`` etc.
"""

import os
from pathlib import Path

import einx
import torch
from boltz.data.pad import pad_dim

from sampleworks.eval.structure_utils import process_structure_to_trajectory_input
from sampleworks.models.boltz.wrapper import process_structure_for_boltz
from sampleworks.utils.frame_transforms import (
    apply_forward_transform,
    create_random_transform,
)
from sampleworks.utils.framework_utils import match_batch
from sampleworks.utils.guidance_script_arguments import GuidanceConfig
from sampleworks.utils.guidance_script_utils import (
    AF3EDMSampler,
    EDMSamplerConfig,
    get_model_and_device,
    get_reward_function_and_structure,
    NoiseSpaceDPSScaler,
    PureGuidance,
)


REPO_ROOT = Path(__file__).resolve().parent
RESOURCES = REPO_ROOT / "tests" / "resources" / "1vme"

DEVICE_STR = os.environ.get("SW_DEVICE", "cuda:0")
CHECKPOINT = Path("/checkpoints/boltz2_conf.ckpt")
STRUCTURE = RESOURCES / "1vme_final_carved_edited_0.5occA_0.5occB.cif"
DENSITY = RESOURCES / "1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
OUTPUT_DIR = REPO_ROOT / "output" / "boltz2_debug"

ENSEMBLE_SIZE = 4  # Boltz is far heavier per sample than Protpardelle; raise if VRAM allows
NUM_DIFFUSION_STEPS = 200  # Boltz default (Protpardelle script used 500)
PARTIAL_DIFFUSION_STEP = 120  # same 0.6 fraction of the trajectory as the ppdl script's 300/500
INSPECT_STEP = PARTIAL_DIFFUSION_STEP  # which step index to unroll by hand below

# Boltz preprocessing writes a manifest, NPZs and MSAs here as a side effect.
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Config for zero guidance (--step-size 0). Unlike the Protpardelle script the
# config is built *before* the model, because get_model_and_device() reads
# `method` off it for Boltz-2.
config = GuidanceConfig.from_cli(
    # A list, not a split() string: "X-RAY DIFFRACTION" contains a space.
    [
        "--model", "boltz2",
        "--guidance-type", "pure_guidance",
        "--protein", "1VME",
        "--model-checkpoint", str(CHECKPOINT),
        "--structure", str(STRUCTURE),
        "--density", str(DENSITY),
        "--method", "X-RAY DIFFRACTION",
        "--resolution", "1.8",
        "--output-dir", str(OUTPUT_DIR),
        "--step-size", "0",
        "--ensemble-size", str(ENSEMBLE_SIZE),
        "--augmentation",
        "--align-to-input",
        "--num-diffusion-steps", str(NUM_DIFFUSION_STEPS),
        "--partial-diffusion-step", str(PARTIAL_DIFFUSION_STEP),
    ]
)  # fmt: skip
args = config

device, model = get_model_and_device(DEVICE_STR, str(CHECKPOINT), "boltz2", config=config)

# Boltz uses the sampler defaults; the s_max/sigma_data overrides in the
# Protpardelle script are Protpardelle-specific (guidance_script_utils.py:529).
sampler_config = EDMSamplerConfig(
    device=str(device),
    augmentation=args.augmentation,
    align_to_input=args.align_to_input,
    alignment_reverse_diffusion=True,  # _three_state_resolver default for Boltz
)
sampler = AF3EDMSampler(config=sampler_config)

# Create step scaler for gradient-based guidance.
step_size = getattr(args, "step_size", None)
if step_size is None:
    step_size = getattr(args, "guidance_weight", 0.01)

step_scaler = NoiseSpaceDPSScaler(
    step_size=step_size,
    gradient_normalization=args.gradient_normalization,
)

num_steps = args.num_diffusion_steps
guidance_t_start = args.guidance_start / num_steps if args.guidance_start > 0 else 0.0
t_start = args.partial_diffusion_step / num_steps if args.partial_diffusion_step else 0.0

guidance = PureGuidance(
    ensemble_size=args.ensemble_size,
    num_steps=num_steps,
    t_start=t_start,
    guidance_t_start=guidance_t_start,
)

reward_function, structure = get_reward_function_and_structure(
    args.density,  # str/path to a map file.
    device,  # this needs to come from the global context, not the args object.
    args.em,
    args.loss_order,
    args.resolution,
    args.structure,  # path/string to a structure file.
)

# Protpardelle's annotate_structure_for_protpardelle() equivalent. This one also
# runs Boltz's input preprocessing (manifest/NPZ/MSA) into out_dir.
structure = process_structure_for_boltz(
    structure,
    out_dir=args.output_dir,
    recycling_steps=getattr(args, "recycling_steps", None),
)

# --- PureGuidance.sample -----------------------------------------------------

# For Boltz this runs the pairformer trunk and caches its output on the features.
features = model.featurize(structure)

coords = torch.as_tensor(
    model.initialize_from_prior(
        batch_size=guidance.ensemble_size,
        features=features,
    ),
)

processed_structure = process_structure_to_trajectory_input(
    structure=structure,
    coords_from_prior=coords,
    features=features,
    ensemble_size=guidance.ensemble_size,
)

reconciler = processed_structure.reconciler.to(coords.device)
reward_inputs = processed_structure.to_reward_inputs(device=coords.device)

print(f"prior coords: {tuple(coords.shape)}")
print(f"input coords: {tuple(processed_structure.input_coords.shape)}")
print(f"reconciler mismatch: {reconciler.has_mismatch}")

trajectory_denoised: list[torch.Tensor] = []
trajectory_next_step: list[torch.Tensor] = []
losses: list[float | None] = []

schedule = sampler.compute_schedule(num_steps=guidance.num_steps)

if guidance.starting_step > 0:
    starting_context = sampler.get_context_for_step(guidance.starting_step - 1, schedule)
    # coords becomes a noisy version of the input coords at this t
    coords = processed_structure.input_coords + coords * torch.as_tensor(
        starting_context.noise_scale
    )

# --- one iteration of the sampling loop, unrolled ----------------------------

i = INSPECT_STEP

context = sampler.get_context_for_step(i, schedule)
apply_guidance = i >= guidance.guidance_start

if apply_guidance:
    pass  # context = context.with_reward(reward_function, reward_inputs) — no guidance for now

context = context.with_reconciler(
    reconciler=reconciler,
    alignment_reference=processed_structure.input_coords,
)

# --- now in AF3EDMSampler.step() ---------------------------------------------

sampler.check_context(context)
state = coords  # already on the model device (process_structure_to_trajectory_input)
model_wrapper = model
scaler = None

t_hat = context.t_effective
dt = context.dt
eps_scale = context.noise_scale
allow_gradients = True if scaler and getattr(scaler, "requires_gradients", False) else False

centroid = einx.mean("... [n] c", state)
state_centered = einx.subtract("... n c, ... c -> ... n c", state, centroid)

transform = (
    create_random_transform(state_centered, center_before_rotation=False)
    if sampler_config.augmentation
    else None
)
maybe_augmented_state = (
    apply_forward_transform(state_centered, transform, rotation_only=False)
    if transform is not None
    else state_centered
)

eps = torch.randn_like(maybe_augmented_state) * eps_scale
noisy_state = maybe_augmented_state + eps
noisy_state = torch.as_tensor(noisy_state).detach().requires_grad_(allow_gradients)

# --- now in Boltz2Wrapper.step() ---------------------------------------------

x_t = noisy_state.to(device=model.device)
t = t_hat
cond = features.conditioning

t_tensor = torch.tensor([t], device=model.device, dtype=torch.float32)
t_tensor = match_batch(t_tensor, target_batch_size=x_t.shape[0])

feats = cond.feats
atom_mask = feats["atom_pad_mask"]  # shape [1, n_padded]
atom_mask = atom_mask.repeat_interleave(x_t.shape[0], dim=0)  # shape [batch, n_padded]

pad_len = atom_mask.shape[1] - x_t.shape[1]
if pad_len < 0:
    raise ValueError("pad_len is negative, cannot pad x_t")
padded_x_t = pad_dim(x_t, dim=1, pad_len=pad_len)

print(f"x_t {tuple(x_t.shape)} -> padded {tuple(padded_x_t.shape)} (pad_len={pad_len})")
print(f"t_hat={t_hat:.4f} dt={dt:.4f} eps_scale={eps_scale:.4f}")

with torch.set_grad_enabled(allow_gradients):
    padded_atom_coords_denoised = model.model.structure_module.preconditioned_network_forward(
        padded_x_t,
        t_tensor,
        network_condition_kwargs=dict(
            multiplicity=padded_x_t.shape[0],
            s_inputs=cond.s_inputs,
            s_trunk=cond.s,
            feats=feats,
            diffusion_conditioning=cond.diffusion_conditioning,
        ),
    )

x_hat_0_manual = padded_atom_coords_denoised[atom_mask.bool(), :].reshape(x_t.shape[0], -1, 3)

# --- back out to the real call path, as a cross-check ------------------------

with torch.no_grad():
    x_hat_0 = model_wrapper.step(noisy_state, t_hat, features=features)

print(f"x_hat_0 {tuple(x_hat_0.shape)}")
print(f"max|manual - wrapper| = {(x_hat_0_manual - x_hat_0).abs().max().item():.3e}")

# Full sampler step, including alignment and the update rule. Pass
# scaler=step_scaler (and attach the reward above) to exercise guidance.
step_output = sampler.step(
    state=coords,
    model_wrapper=model,
    context=context,
    scaler=None,
    features=features,
)

print(f"next state {tuple(step_output.state.shape)}, loss={step_output.loss}")
