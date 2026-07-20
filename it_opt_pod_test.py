"""Throwaway pod driver for the IT-opt single-structure Protenix test.

Stage A: gradient-flow check — does a gradient reach the injected s_trunk leaf?
Stage B: tiny end-to-end LatentOptimization run — does the loss move without crashing?

Run on the pod:
    pixi run -e protenix-dev python it_opt_pod_test.py
"""

from __future__ import annotations

import traceback

import torch

from sampleworks.core.samplers.edm import AF3EDMSampler, EDMSamplerConfig
from sampleworks.core.scalers.latent_optimization import LatentOptimization
from sampleworks.core.scalers.step_scalers import NoScalingScaler
from sampleworks.models.latent_adapter import AttrLatentIO
from sampleworks.models.protenix.wrapper import annotate_structure_for_protenix
from sampleworks.models.protocol import GenerativeModelInput
from sampleworks.utils.guidance_constants import StructurePredictor
from sampleworks.utils.guidance_script_utils import (
    get_model_and_device,
    get_reward_function_and_structure,
)

RES = "tests/resources/1vme"
DENSITY = f"{RES}/1vme_final_carved_edited_0.5occA_0.5occB_1.80A.ccp4"
STRUCTURE = f"{RES}/1vme_final_carved_edited_0.5occA_0.5occB.cif"
RESOLUTION = 1.8


def build():
    print(f"torch {torch.__version__}  cuda_available={torch.cuda.is_available()}")
    device, model = get_model_and_device("cuda:0", None, StructurePredictor.PROTENIX)
    print(f"device={device}  model={type(model).__name__}")
    reward, structure = get_reward_function_and_structure(
        DENSITY, device, False, 2, RESOLUTION, STRUCTURE
    )
    structure = annotate_structure_for_protenix(structure, ensemble_size=1)
    return device, model, reward, structure


def stage_a(model, structure):
    """One differentiable denoise; assert the s_trunk leaf receives a gradient."""
    print("\n===== STAGE A: gradient-flow check (s_trunk) =====")
    with torch.no_grad():
        feats = model.featurize(structure)
    io = AttrLatentIO(single_attr="s_trunk")
    s0 = io.read_single(feats.conditioning)
    print(f"s_trunk shape={tuple(s0.shape)} requires_grad={s0.requires_grad}")
    s0 = s0.detach()
    leaf = s0.clone().requires_grad_(True)
    cond = io.write_single(feats.conditioning, leaf)
    feats2 = GenerativeModelInput(x_init=feats.x_init, conditioning=cond)

    x = torch.as_tensor(model.initialize_from_prior(1, features=feats2))
    t = torch.tensor([10.0], device=x.device, dtype=x.dtype)
    with torch.enable_grad():
        x0 = model.step(x, t, features=feats2)
        loss = x0.float().sum()
        loss.backward()

    grad_none = leaf.grad is None
    grad_abs = None if grad_none else float(leaf.grad.abs().sum())
    print(f"leaf.grad is None: {grad_none}")
    print(f"leaf.grad abs-sum: {grad_abs}")
    ok = (not grad_none) and grad_abs is not None and grad_abs > 0.0
    print(f"STAGE A: {'PASS (gradient reaches s_trunk)' if ok else 'FAIL (no gradient to s_trunk)'}")
    return ok


def stage_c(device, model, reward, structure):
    """Real-scale efficacy: no-op baseline vs optimized; does density fit improve?"""
    print("\n===== STAGE C: efficacy (baseline vs optimized, num_steps=200) =====")
    NUM = 200
    sampler = AF3EDMSampler(
        EDMSamplerConfig(device=str(device), augmentation=False, align_to_input=True)
    )

    def run(tag, *, outer_steps, lr, opt_start):
        torch.manual_seed(0)  # same prior/churn seed for a fairer comparison
        sc = LatentOptimization(
            ensemble_size=1,
            num_steps=NUM,
            guidance_t_start=opt_start,  # 1.0 => never optimize (no-op baseline)
            outer_steps=outer_steps,
            optimize_single=True,
            optimize_pair=False,
            single_attr="s_trunk",
            anchor_weight_single=1.0,
            learning_rate=lr,
        )
        res = sc.sample(structure, model, sampler, step_scaler=NoScalingScaler(), reward=reward)
        finals = [x for x in (res.losses or []) if x is not None]
        opt = res.metadata.get("optimization_losses", [])
        per_round_last = [round(r[-1], 4) for r in opt if r]
        print(f"[{tag}] final-pass density loss  last={finals[-1]:.4f}  min={min(finals):.4f}")
        if per_round_last:
            print(f"[{tag}] per-round last-step opt loss (should trend DOWN): {per_round_last}")
        return finals[-1], min(finals)

    base_last, base_min = run("baseline (opt OFF)", outer_steps=1, lr=0.0, opt_start=1.0)
    opt_last, opt_min = run("optimized x3", outer_steps=3, lr=0.05, opt_start=0.0)

    print(f"\nSTAGE C RESULT  final-fit(last)  baseline={base_last:.4f}  optimized={opt_last:.4f}  "
          f"delta={opt_last - base_last:+.4f}")
    print(f"STAGE C RESULT  final-fit(min)   baseline={base_min:.4f}  optimized={opt_min:.4f}  "
          f"delta={opt_min - base_min:+.4f}")
    print("STAGE C: negative delta => latent optimization improved the density fit")


def main():
    try:
        device, model, reward, structure = build()
    except Exception:
        print("BUILD FAILED:")
        traceback.print_exc()
        return
    try:
        stage_a(model, structure)
    except Exception:
        print("STAGE A FAILED:")
        traceback.print_exc()
    try:
        stage_c(device, model, reward, structure)
    except Exception:
        print("STAGE C FAILED:")
        traceback.print_exc()


if __name__ == "__main__":
    main()
