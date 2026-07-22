"""Inference-time latent-space optimization (IT-opt).

A Sampleworks port of the reference ``run_it_optimization``
(``it_opt/protenix/it_optimization_manager.py:288``). It treats the frozen
structure model as a differentiable sampler and optimizes its cached post-trunk
latents -- the single representation ``s`` and the pair representation ``z`` --
against an experimental reward evaluated on the *denoised* structure. No model
weights are updated. See ``docs/IT_OPTIMIZATION_PLAN.md`` and, for a line-by-line
mapping to the reference, ``docs/IT_OPT_REFERENCE_COMPARISON.md``.

The algorithm (kept faithful to the reference), with the reference's bugs fixed:

    extract (s, z) from the trunk once  ->  make them optimizable leaves
    for each optimization round (fresh diffusion noise each round):
        build ONE Adam over [s, z]                       # reference rebuilt it per step
        for each diffusion step:
            x0_hat  = differentiable denoise(noisy_t, s, z)   # one forward
            loss    = reward(x0_hat) + anchor(s, z)
            loss.backward();  per-latent grad-clip (s, then z);  adam.step()
            advance the trajectory one step (latents frozen)
    final clean sampling pass with the optimized latents -> saved ensemble

Only Boltz1 and RF3 propagate gradients cleanly to the cached latents out of the
box. Protenix needs the reversible test edit in ``models/protenix/wrapper.py``
(and its diffusion cache disabled for z); Boltz2 needs its diffusion conditioning
recomputed. See ``docs/IT_OPT_TESTING_PROTENIX.md``.

v1 does ONLY latent optimization: coordinate-space guidance is not applied (the
attached scaler produces a zero coordinate direction), so the sole steering comes
from the evolving latents.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from jaxtyping import Float
from loguru import logger
from torch import Tensor
from tqdm import tqdm

from sampleworks.core.rewards.protocol import RewardFunctionProtocol
from sampleworks.core.samplers.protocol import StepParams, TrajectorySampler
from sampleworks.core.scalers.protocol import GuidanceOutput, StepScalerProtocol
from sampleworks.eval.structure_utils import process_structure_to_trajectory_input
from sampleworks.models.latent_adapter import AttrLatentIO
from sampleworks.models.protocol import FlowModelWrapper, GenerativeModelInput


class _GradEnablingScaler:
    """Step scaler that only enables gradients, applying zero coordinate guidance.

    ``AF3EDMSampler`` runs its denoiser forward under autograd only when the
    attached scaler exposes ``requires_gradients=True``; that is what makes the
    returned :attr:`SamplerStepOutput.denoised` carry a graph back to the injected
    latent leaves. Returning a zero direction keeps the diffusion *advance*
    unguided -- realizing "coordinate guidance disabled" without editing the
    sampler. The reward is recomputed by the trajectory scaler on ``denoised``, so
    there is exactly one denoiser forward and one reward evaluation per step (the
    reference's Tier-2 single-forward optimization, for free).
    """

    requires_gradients = True

    def scale(
        self,
        state: Float[Tensor, "*batch atoms 3"],
        context: StepParams,
        *,
        model: FlowModelWrapper | None = None,
    ) -> tuple[Float[Tensor, "*batch atoms 3"], Float[Tensor, " batch"]]:
        return torch.zeros_like(state), torch.zeros(state.shape[0], device=state.device)

    def guidance_strength(self, context: StepParams) -> Float[Tensor, " batch"]:
        return torch.zeros_like(torch.as_tensor(context.t_effective))


class LatentAnchor:
    r"""On-manifold prior penalizing drift of the latents from their trunk baseline.

    ``penalty = Σ_i w_i · mean((latent_i − baseline_i)²)``.

    The reference ``AnchorLossFunction`` uses a per-sample Frobenius **norm**
    ``‖latent − baseline‖`` (see ``anchor_loss_function.py``). We use the *mean
    squared* deviation instead: it is shape-agnostic and normalizes for the very
    different element counts of ``s`` and ``z`` (``z`` is O(tokens) larger), so the
    weights ``w_s`` and ``w_z`` land on a comparable scale -- part of keeping the
    two latent updates harmonized. Retune the weights when switching objectives.
    """

    def __init__(self, weights: Sequence[float]):
        self.weights = list(weights)

    def __call__(self, latents: Sequence[Tensor], baselines: Sequence[Tensor]) -> Tensor:
        terms = [
            w * torch.mean((lat - base) ** 2)
            for w, lat, base in zip(self.weights, latents, baselines)
        ]
        return torch.stack(terms).sum()


class LatentOptimization:
    """Trajectory scaler that optimizes the model's ``s``/``z`` latents (IT-opt).

    Satisfies ``TrajectoryScalerProtocol`` and drops in alongside ``PureGuidance``
    / ``FKSteering``.
    """

    def __init__(
        self,
        ensemble_size: int = 1,
        num_steps: int = 200,
        guidance_t_start: float = 0.0,
        *,
        outer_steps: int = 1,
        learning_rate: float = 0.05,
        max_grad_norm: float = 1.0,
        optimize_single: bool = True,
        optimize_pair: bool = True,
        anchor_weight_single: float = 0.0,
        anchor_weight_pair: float = 0.0,
        single_attr: str = "s",
        pair_attr: str = "z",
    ):
        """Initialize the latent-optimization trajectory scaler.

        Parameters
        ----------
        ensemble_size
            Number of structures sampled in parallel (they share the latents).
        num_steps
            Number of diffusion steps per pass.
        guidance_t_start
            Fraction of ``num_steps`` after which to begin optimizing within each
            round. Before it, steps are plain frozen-latent diffusion. Default 0
            (optimize from the first step, as the reference does).
        outer_steps
            Number of optimization rounds -- full diffusion passes, each with fresh
            prior noise, sharing the persisting latents (the reference's
            ``outer_diffusion_steps``). Default 1 for a cheap first run.
        learning_rate
            Adam learning rate. A *real* persistent Adam is built once per round --
            unlike the reference, which rebuilt it every diffusion step and thereby
            degenerated to signed gradient descent (its headline bug).
        max_grad_norm
            Per-latent gradient-clip threshold: each optimized latent (``s``, ``z``)
            is clipped to this norm **independently** (matching the reference), so
            ``s``'s update is not scaled down by ``z``'s much larger gradient. A single
            joint clip over ``[s, z]`` does the opposite -- ``z`` dominates the shared
            coefficient and starves ``s``.
        optimize_single, optimize_pair
            Which representations to optimize. Both by default.
        anchor_weight_single, anchor_weight_pair
            Weights of the on-manifold L2-to-baseline anchor per latent.
        single_attr, pair_attr
            Conditioning attribute names for the single / pair representation
            (``"s"``/``"z"`` for Boltz, ``"s_trunk"``/``"z_trunk"`` for
            Protenix/RF3).
        """
        logger.info(
            f"Initialized LatentOptimization (IT-opt): outer_steps={outer_steps}, "
            f"num_steps={num_steps}, lr={learning_rate}, "
            f"optimize_single={optimize_single}, optimize_pair={optimize_pair}."
        )
        self.ensemble_size = ensemble_size
        self.num_steps = num_steps
        self.guidance_start = int(guidance_t_start * num_steps)
        self.outer_steps = outer_steps
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.optimize_single = optimize_single
        self.optimize_pair = optimize_pair
        self.anchor_weight_single = anchor_weight_single
        self.anchor_weight_pair = anchor_weight_pair
        self.single_attr = single_attr
        self.pair_attr = pair_attr

    def sample(
        self,
        structure: dict,
        model: FlowModelWrapper,
        sampler: TrajectorySampler,
        step_scaler: StepScalerProtocol,
        reward: RewardFunctionProtocol,
        num_particles: int = 1,
    ) -> GuidanceOutput:
        """Optimize the latents against ``reward``, then sample the final ensemble.

        ``step_scaler`` is ignored: v1 does latent optimization only (no
        coordinate-space guidance). ``reward`` is evaluated on the
        reconciler-aligned denoised prediction.
        """
        io = AttrLatentIO(
            single_attr=self.single_attr,
            pair_attr=self.pair_attr if self.optimize_pair else None,
        )

        # --- extract (s, z) once and make them optimizable leaves ---------------
        # no_grad avoids retaining a graph into the (frozen) trunk; we build leaves
        # from detached copies. (reference: get_msa_features + clone/detach.)
        with torch.no_grad():
            features = model.featurize(structure)
        features, latents, baselines, anchor_weights = self._leaf_latents(features, io)
        anchor = LatentAnchor(anchor_weights)

        # --- shared per-trajectory context (reconciler + reward inputs) ---------
        coords = torch.as_tensor(
            model.initialize_from_prior(batch_size=self.ensemble_size, features=features)
        )
        processed = process_structure_to_trajectory_input(
            structure=structure,
            coords_from_prior=coords,
            features=features,
            ensemble_size=self.ensemble_size,
        )
        reconciler = processed.reconciler.to(coords.device)
        reward_inputs = processed.to_reward_inputs(device=coords.device)
        schedule = sampler.compute_schedule(num_steps=self.num_steps)
        grad_enabler = _GradEnablingScaler()

        # --- optimize the latents (outer resample × inner diffusion steps) ------
        optimization_losses: list[list[float]] = []
        latent_drift: list[list[float]] = []
        for outer in range(self.outer_steps):
            optimizer = torch.optim.Adam(latents, lr=self.learning_rate)  # once per round
            round_losses = self._optimize_one_round(
                model=model,
                sampler=sampler,
                reward=reward,
                features=features,
                latents=latents,
                baselines=baselines,
                anchor=anchor,
                optimizer=optimizer,
                schedule=schedule,
                reconciler=reconciler,
                alignment_reference=processed.input_coords,
                reward_inputs=reward_inputs,
                grad_enabler=grad_enabler,
                round_index=outer,
            )
            optimization_losses.append(round_losses)
            # relative drift of each latent from its trunk baseline -- shows which latent
            # actually moved (s vs z), which the loss trend alone cannot reveal.
            latent_drift.append(
                [
                    float((lat.detach() - base).norm() / (base.norm() + 1e-12))
                    for lat, base in zip(latents, baselines)
                ]
            )

        # --- final clean sampling pass with the optimized latents ---------------
        final_coords, trajectory, losses = self._sample_with_frozen_latents(
            model=model,
            sampler=sampler,
            reward=reward,
            io=io,
            features=features,
            latents=latents,
            schedule=schedule,
            reconciler=reconciler,
            alignment_reference=processed.input_coords,
            reward_inputs=reward_inputs,
        )

        metadata: dict = {
            "optimization_losses": optimization_losses,
            "latent_drift": latent_drift,
        }
        if reconciler.has_mismatch and processed.model_atom_array is not None:
            metadata["model_atom_array"] = processed.model_atom_array

        return GuidanceOutput(
            structure=structure,
            final_state=final_coords,
            trajectory=trajectory,
            losses=losses,
            metadata=metadata,
        )

    def _optimize_one_round(
        self,
        *,
        model,
        sampler,
        reward,
        features,
        latents,
        baselines,
        anchor,
        optimizer,
        schedule,
        reconciler,
        alignment_reference,
        reward_inputs,
        grad_enabler,
        round_index,
    ) -> list[float]:
        """One optimization round: a full diffusion pass that updates the latents.

        Fresh prior noise; the persisting latents are updated once per diffusion
        step against the reward on that step's denoised prediction, then the
        trajectory is advanced with the latents frozen. Returns the per-step data
        losses.
        """
        coords = torch.as_tensor(
            model.initialize_from_prior(batch_size=self.ensemble_size, features=features)
        )
        losses: list[float] = []
        steps = tqdm(range(self.num_steps), f"IT-opt round {round_index}")
        for i in steps:
            optimize = i >= self.guidance_start
            context = sampler.get_context_for_step(i, schedule)
            if optimize:
                context = context.with_reward(reward, reward_inputs)
            context = context.with_reconciler(
                reconciler=reconciler, alignment_reference=alignment_reference
            )

            step_output = sampler.step(
                state=coords,
                model_wrapper=model,
                context=context,
                scaler=grad_enabler if optimize else None,
                features=features,
            )

            if optimize:
                data_loss = self._latent_adam_step(
                    denoised=step_output.denoised,
                    reward=reward,
                    reward_inputs=reward_inputs,
                    optimizer=optimizer,
                    anchor=anchor,
                    latents=latents,
                    baselines=baselines,
                )
                losses.append(data_loss)
                steps.set_postfix(loss=data_loss)

            coords = step_output.state.detach()
        return losses

    def _latent_adam_step(
        self,
        *,
        denoised: Tensor | None,
        reward: RewardFunctionProtocol,
        reward_inputs,
        optimizer: torch.optim.Optimizer,
        anchor: LatentAnchor,
        latents: Sequence[Tensor],
        baselines: Sequence[Tensor],
    ) -> float:
        """Score the denoised structure, backprop to the latents, take one Adam step.

        ``denoised`` is the sampler's aligned prediction and carries a graph back to
        the latent leaves. Each latent is gradient-clipped **separately** so ``s``'s
        step is not scaled down by ``z``'s much larger gradient. Returns the data-only
        reward for logging.
        """
        if denoised is None:
            raise RuntimeError(
                "Sampler returned no denoised prediction; the grad-enabling scaler "
                "should have been attached this step so the latent gradient exists."
            )
        data_loss = reward(
            coordinates=denoised,
            elements=reward_inputs.elements,
            b_factors=reward_inputs.b_factors,
            occupancies=reward_inputs.occupancies,
        )
        loss = data_loss + anchor(latents, baselines)

        optimizer.zero_grad()
        loss.backward()
        # Clip each latent SEPARATELY (matches reference it_optimization_manager.py:394-395), never
        # as one joint [s, z] group. NB: with the density reward the gradients are tiny (grad-norm
        # ~1e-4, far below a max_grad_norm of 1.0), so this clip is effectively inert here -- it
        # bites only for rewards whose gradients exceed the threshold, where separate (not joint)
        # clips keep s's step decoupled from z's much larger gradient scale.
        for latent in latents:
            torch.nn.utils.clip_grad_norm_(latent, self.max_grad_norm)
        optimizer.step()
        return float(data_loss.detach())

    def _sample_with_frozen_latents(
        self,
        *,
        model,
        sampler,
        reward,
        io,
        features,
        latents,
        schedule,
        reconciler,
        alignment_reference,
        reward_inputs,
    ) -> tuple[Tensor, list[Tensor], list[float | None]]:
        """Final clean diffusion pass with the optimized latents held fixed.

        No optimization and no coordinate guidance (``scaler=None``). This produces
        the ensemble that is returned/saved -- the reference's
        ``run_diffusion_process_it_optimized``. The per-step reward (evaluated under
        no-grad on the denoised prediction) is returned for logging.
        """
        # Re-inject detached copies so the final pass is purely a frozen sampler.
        conditioning = features.conditioning
        detached = [lat.detach() for lat in latents]
        if self.optimize_single and io.read_single(conditioning) is not None:
            conditioning = io.write_single(conditioning, detached.pop(0))
        if self.optimize_pair and io.read_pair(conditioning) is not None:
            conditioning = io.write_pair(conditioning, detached.pop(0))
        frozen_features = GenerativeModelInput(x_init=features.x_init, conditioning=conditioning)

        coords = torch.as_tensor(
            model.initialize_from_prior(batch_size=self.ensemble_size, features=frozen_features)
        )
        trajectory: list[Tensor] = []
        losses: list[float | None] = []
        steps = tqdm(range(self.num_steps), "IT-opt final sampling")
        for i in steps:
            context = sampler.get_context_for_step(i, schedule).with_reconciler(
                reconciler=reconciler, alignment_reference=alignment_reference
            )
            step_output = sampler.step(
                state=coords,
                model_wrapper=model,
                context=context,
                scaler=None,
                features=frozen_features,
            )
            coords = step_output.state.detach()
            trajectory.append(coords.clone().cpu())

            if step_output.denoised is not None:
                with torch.no_grad():
                    loss = reward(
                        coordinates=step_output.denoised,
                        elements=reward_inputs.elements,
                        b_factors=reward_inputs.b_factors,
                        occupancies=reward_inputs.occupancies,
                    )
                losses.append(float(loss))
            else:
                losses.append(None)
        return coords, trajectory, losses

    def _leaf_latents(
        self, features: GenerativeModelInput, io: AttrLatentIO
    ) -> tuple[GenerativeModelInput, list[Tensor], list[Tensor], list[float]]:
        """Replace ``s``/``z`` on the conditioning with fresh optimizable leaves.

        Returns the rewritten ``features`` plus parallel lists of leaves, their
        detached baselines (anchor targets), and per-latent anchor weights. Each
        leaf is a detached clone made ``requires_grad=True`` -- a true leaf severed
        from any trunk graph, so Adam updates it directly (leaves persist and are
        updated in place across rounds and steps). Shapes are preserved (whatever
        the wrapper caches), so no assumption is made about a batch dimension.
        """
        conditioning = features.conditioning
        latents: list[Tensor] = []
        baselines: list[Tensor] = []
        anchor_weights: list[float] = []

        specs = (
            (self.optimize_single, io.read_single, io.write_single, self.anchor_weight_single),
            (self.optimize_pair, io.read_pair, io.write_pair, self.anchor_weight_pair),
        )
        for enabled, read, write, anchor_weight in specs:
            if not enabled:
                continue
            baseline = read(conditioning)
            if baseline is None:
                continue
            baseline = baseline.detach()
            leaf = baseline.clone().requires_grad_(True)
            conditioning = write(conditioning, leaf)
            latents.append(leaf)
            baselines.append(baseline)
            anchor_weights.append(anchor_weight)

        if not latents:
            raise ValueError(
                "LatentOptimization found no optimizable latent on the conditioning "
                f"(single_attr={self.single_attr!r}, pair_attr={self.pair_attr!r}). "
                "Check the attribute names for this model."
            )
        features = GenerativeModelInput(x_init=features.x_init, conditioning=conditioning)
        return features, latents, baselines, anchor_weights
