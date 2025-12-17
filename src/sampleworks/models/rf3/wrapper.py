import json
from collections import deque
from collections.abc import Mapping
from logging import getLogger, Logger
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from atomworks.enums import ChainType
from atomworks.ml.samplers import LoadBalancedDistributedSampler
from einx import rearrange
from jaxtyping import ArrayLike, Float
from rf3.inference_engines import RF3InferenceEngine
from rf3.model.RF3 import RF3WithConfidence
from rf3.trainers.rf3 import assert_no_nans, RF3TrainerWithConfidence
from rf3.utils.inference import InferenceInput, InferenceInputDataset
from torch import Tensor
from torch.utils.data import DataLoader

from sampleworks.utils.torch_utils import send_tensors_in_dict_to_device


class RF3Wrapper:
    """Wrapper for RosettaFold 3 (Baker Lab AlphaFold 3 replication)."""

    SIGMA_DATA = 16.0
    S_MIN = 4e-4
    S_MAX = 160.0
    P = 7.0
    GAMMA_0 = 0.8
    GAMMA_MIN = 0.05
    STEP_SCALE = 1.5
    NOISE_SCALE = 1.003

    def __init__(
        self,
        checkpoint_path: str | Path,
    ):
        """
        Parameters
        ----------
        checkpoint_path: str | Path
            Filesystem path to the checkpoint containing trained weights.
        """
        logger: Logger = getLogger(__name__)
        logger.info("Loading RF3 Inference Engine")

        self.checkpoint_path = (
            Path(checkpoint_path).expanduser().resolve()
            if isinstance(checkpoint_path, str)
            else checkpoint_path.expanduser().resolve()
        )

        # TODO: expose num_steps, num_recycles to user
        self.num_steps = 200  # RF3 default number of diffusion steps
        self.num_recycles = 10  # RF3 default number of recycles

        self.inference_engine = RF3InferenceEngine(
            ckpt_path=str(self.checkpoint_path),
            n_recycles=self.num_recycles,
            diffusion_batch_size=1,
            num_steps=self.num_steps,
        )
        self.inference_engine.initialize()

        self.inference_engine.trainer = cast(
            RF3TrainerWithConfidence, self.inference_engine.trainer
        )
        self.model = self.inference_engine.trainer.state["model"]
        self._device = self.inference_engine.trainer.fabric.device

        self.cached_representations: dict[str, Any] = {}
        self.noise_schedule = self._compute_noise_schedule(self.num_steps)

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def _inner_model(self) -> RF3WithConfidence:
        """Access the unwrapped RF3WithConfidence model through EMA wrappers."""
        model = self.model
        if hasattr(model, "shadow"):
            model = model.shadow
        return cast(RF3WithConfidence, model)

    def _compute_noise_schedule(self, num_steps: int) -> dict[str, Float[Tensor, "..."]]:
        """Compute the noise schedule for diffusion sampling.

        Uses RF3's EDM-style schedule formula:
        t_hat = sigma_data * (s_max^(1/p) + t*(s_min^(1/p) - s_max^(1/p)))^p

        Parameters
        ----------
        num_steps: int
            Number of diffusion sampling steps.

        Returns
        -------
        dict[str, Tensor]
            Noise schedule with keys sigma_tm, sigma_t, gamma.
        """
        t_values = torch.linspace(0, 1, num_steps + 1, device=self.device)

        sigmas = (
            self.SIGMA_DATA
            * (
                self.S_MAX ** (1 / self.P)
                + t_values * (self.S_MIN ** (1 / self.P) - self.S_MAX ** (1 / self.P))
            )
            ** self.P
        )

        gammas = torch.where(
            sigmas > self.GAMMA_MIN,
            torch.tensor(self.GAMMA_0, device=self.device),
            torch.tensor(0.0, device=self.device),
        )

        return {
            "sigma_tm": sigmas[:-1],
            "sigma_t": sigmas[1:],
            "gamma": gammas[1:],
        }

    def featurize(
        self, structure: dict, msa_path: str | Path | dict | None = None, **kwargs: dict
    ) -> dict[str, Any]:
        """From an Atomworks structure, calculate RF3 input features.

        Parameters
        ----------
        structure: dict
            Atomworks structure dictionary.
        msa_path: dict | str | Path | None
            MSA specification. Can be:
            - dict: chain_id -> MSA file path mapping
            - str/Path to .json: JSON file with chain_id -> MSA path mapping
            - str/Path to .a3m: Single MSA file applied to all protein chains
            - None: No MSA information is used
        **kwargs: dict, optional
            Additional arguments for feature generation.

        Returns
        -------
        dict[str, Any]
            RF3 input features.
        """

        # If featurize is called again, we should clear cached representations
        # to avoid using stale data
        self.cached_representations.clear()

        if "asym_unit" not in structure:
            raise ValueError("structure must contain 'asym_unit' key")

        atom_array = structure["asym_unit"]
        chain_info = structure.get("chain_info", {})

        # If msa_path is a JSON file, read it to get chain_id -> msa_path mapping
        if isinstance(msa_path, (str, Path)):
            msa_path_obj = Path(msa_path)
            if msa_path_obj.suffix == ".json" and msa_path_obj.exists():
                with open(msa_path_obj) as f:
                    msa_path = json.load(f)

        # InferenceInput expects msa_path in chain_info
        for chain_id in chain_info:
            if chain_info[chain_id]["chain_type"] == ChainType.POLYPEPTIDE_L:
                if isinstance(msa_path, dict):
                    chain_msa_path = msa_path.get(chain_id, None)
                else:
                    chain_msa_path = msa_path

                if chain_msa_path is not None:
                    chain_info[chain_id]["msa_path"] = chain_msa_path

        inference_input = InferenceInput.from_atom_array(atom_array, chain_info=chain_info)

        inference_dataset = InferenceInputDataset([inference_input])
        trainer = cast(RF3TrainerWithConfidence, self.inference_engine.trainer)

        sampler = LoadBalancedDistributedSampler(
            dataset=inference_dataset,
            key_to_balance=inference_dataset.key_to_balance,
            num_replicas=trainer.fabric.world_size,
            rank=trainer.fabric.global_rank,
            drop_last=False,
        )

        loader = DataLoader(
            dataset=inference_dataset,
            sampler=sampler,
            batch_size=1,
            # multiprocessing is disabled since it shouldn't be hard to read
            # InferenceInput objects
            num_workers=0,
            collate_fn=lambda x: x,  # no collation since we're not batching
            pin_memory=True,
            drop_last=False,
        )

        input_batch = next(iter(loader))
        input_spec = cast(
            InferenceInput, input_batch[0]
        )  # since we're not batching, the loader returns a list of length 1

        # (Hydra instantiation of pipeline means it is going to be hard to type check here)
        pipeline_output = self.inference_engine.pipeline(input_spec.to_pipeline_input())  # type: ignore
        pipeline_output = trainer.fabric.to_device(pipeline_output)

        example = pipeline_output[0] if not isinstance(pipeline_output, dict) else pipeline_output

        features = trainer._assemble_network_inputs(example)

        assert_no_nans(
            features,
            msg=f"network_input for example_id: {example['example_id']}",
        )

        features = send_tensors_in_dict_to_device(features, self.device)

        return features

    def step(self, features: dict[str, Any], grad_needed: bool = False, **kwargs) -> dict[str, Any]:
        """Perform a pass through the RF3 Pairformer to obtain representations.

        Parameters
        ----------
        features: dict[str, Any]
            Model features as returned by featurize.
        grad_needed: bool, optional
            Whether gradients are needed for this pass, by default False.
        **kwargs: dict, optional
            Additional arguments.

            - recycling_steps: int
                Number of recycling steps to perform. Defaults to n_recycles.

        Returns
        -------
        dict[str, Any]
            RF3 model outputs including trunk representations (s_inputs, s_trunk,
            z_trunk).
        """
        recycling_steps = kwargs.get("recycling_steps", self.num_recycles)

        with (
            torch.set_grad_enabled(grad_needed),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):  # TODO: this will require new GPUs and new CUDA for now, may want to fix?
            recycling_output_generator = self._inner_model.trunk_forward_with_recycling(
                features["f"], n_recycles=recycling_steps
            )

            # (We use `deque` with maxlen=1 to ensure that we only keep the last output
            #  in memory)
            try:
                recycling_outputs = deque(recycling_output_generator, maxlen=1).pop()
            except IndexError:
                # Handle the case where the generator is empty
                raise RuntimeError("Recycling generator produced no outputs")

        s_inputs = recycling_outputs["S_inputs_I"]
        s_trunk = recycling_outputs["S_I"]
        z_trunk = recycling_outputs["Z_II"]

        return {
            "s_inputs": s_inputs,
            "s_trunk": s_trunk,
            "z_trunk": z_trunk,
            "features": features["f"],
        }

    def denoise_step(
        self,
        features: dict[str, Any],
        noisy_coords: Float[ArrayLike | Tensor, "..."],
        timestep: float | int,
        grad_needed: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Perform denoising at given timestep for RF3 model.

        Parameters
        ----------
        features: dict[str, Any]
            Model features produced by featurize or step.
        noisy_coords: Float[ArrayLike | Tensor, "..."]
            Noisy atom coordinates at the current timestep.
        timestep: float | int
            Current timestep or noise level in reverse time (starts from 0).
        grad_needed: bool, optional
            Whether gradients are needed for this pass, by default False.
        **kwargs: dict, optional
            Additional keyword arguments for RF3 denoising.
            - t_hat: float, optional
                Precomputed t_hat value; computed internally if not provided.
            - eps: Tensor, optional
                Precomputed noise tensor; sampled internally if not provided.
            - overwrite_representations: bool, optional
                Whether to recompute cached representations (default False).
            - recycling_steps: int
                Number of recycling steps to perform.

        Returns
        -------
        dict[str, Tensor]
            Dictionary containing atom_coords_denoised with cleaned coordinates.
        """
        if not self.cached_representations or kwargs.get("overwrite_representations", False):
            self.cached_representations = self.step(
                features,
                grad_needed=False,
                recycling_steps=kwargs.get("recycling_steps", self.num_recycles),
            )

        outputs = self.cached_representations

        if not isinstance(noisy_coords, torch.Tensor):
            noisy_coords = torch.tensor(noisy_coords, device=self.device)

        with (
            torch.set_grad_enabled(grad_needed),
            torch.autocast("cuda", dtype=torch.float32),
        ):  # TODO: this will require new GPUs and new CUDA for now, may want to fix?
            if "t_hat" in kwargs and "eps" in kwargs:
                t_hat = kwargs["t_hat"]
                eps = cast(Tensor, kwargs["eps"])
            else:
                timestep_scaling = self.get_timestep_scaling(timestep)
                eps = timestep_scaling["eps_scale"] * torch.randn(
                    noisy_coords.shape, device=self.device
                )
                t_hat = timestep_scaling["t_hat"]

            noisy_coords_eps = noisy_coords + eps

            batch_size = noisy_coords_eps.shape[0]
            t_tensor = torch.full((batch_size,), t_hat, device=self.device)

            atom_coords_denoised: torch.Tensor = self._inner_model.diffusion_module(
                X_noisy_L=noisy_coords_eps,
                t=t_tensor,
                f=outputs["features"],
                S_inputs_I=outputs["s_inputs"],
                S_trunk_I=outputs["s_trunk"],
                Z_trunk_II=outputs["z_trunk"],
            )

        return {"atom_coords_denoised": atom_coords_denoised.float()}

    def get_noise_schedule(self) -> Mapping[str, Float[ArrayLike | Tensor, "..."]]:
        """Return the full noise schedule with semantic keys.

        Returns
        -------
        Mapping[str, Float[ArrayLike | Tensor, "..."]]
            Noise schedule arrays with keys sigma_tm, sigma_t, and gamma.
        """
        return self.noise_schedule

    def get_timestep_scaling(self, timestep: float | int) -> dict[str, float]:
        """Return scaling constants for RF3 diffusion at given timestep.

        Parameters
        ----------
        timestep: float | int
            Current timestep or noise level starting from 0.

        Returns
        -------
        dict[str, float]
            Dictionary containing t_hat, sigma_t, and eps_scale.
        """
        if timestep < 0 or timestep >= self.num_steps:
            raise ValueError(f"timestep {timestep} is out of bounds for {self.num_steps} steps")

        sigma_tm = self.noise_schedule["sigma_tm"][int(timestep)]
        sigma_t = self.noise_schedule["sigma_t"][int(timestep)]
        gamma = self.noise_schedule["gamma"][int(timestep)]

        t_hat = sigma_tm * (1 + gamma)
        eps_scale = self.NOISE_SCALE * torch.sqrt(t_hat**2 - sigma_tm**2)

        return {
            "t_hat": t_hat.item(),
            "sigma_t": sigma_t.item(),
            "eps_scale": eps_scale.item(),
        }

    def initialize_from_noise(
        self,
        structure: dict,
        noise_level: float | int,
        ensemble_size: int = 1,
        **kwargs,
    ) -> Float[ArrayLike | Tensor, "*batch _num_atoms 3"]:
        """Create a noisy version of structure coordinates at given noise level.

        Parameters
        ----------
        structure: dict
            Atomworks structure dictionary.
        noise_level: float | int
            Timestep or noise level in reverse time starting from 0.
        ensemble_size: int, optional
            Number of noisy samples to generate per input structure (default 1).
        **kwargs: dict, optional
            Additional keyword arguments for initialization.

        Returns
        -------
        Float[ArrayLike | Tensor, "*batch _num_atoms 3"]
            Noisy structure coordinates for atoms with nonzero occupancy.
        """
        if "asym_unit" not in structure:
            raise ValueError("structure must contain 'asym_unit' key to access coordinates")

        if noise_level < 0 or noise_level >= self.num_steps:
            raise ValueError(
                f"noise_level {noise_level} is out of bounds for {self.num_steps} steps"
            )

        asym_unit = structure["asym_unit"]
        occupancy_mask = asym_unit.occupancy > 0
        coord_array = np.asarray(asym_unit.coord)

        if coord_array.ndim == 3:
            nan_mask = ~np.any(np.isnan(coord_array[0]), axis=-1)
            valid_mask = occupancy_mask & nan_mask
            coords = asym_unit.coord[:, valid_mask]
        else:
            nan_mask = ~np.any(np.isnan(coord_array), axis=-1)
            valid_mask = occupancy_mask & nan_mask
            coords = asym_unit.coord[valid_mask]

        if isinstance(coords, np.ndarray):
            coords = torch.tensor(coords, device=self.device, dtype=torch.float32)

        if coords.ndim == 2:
            coords = cast(Tensor, rearrange("n c -> e n c", coords, e=ensemble_size))
        elif coords.ndim == 3 and coords.shape[0] == 1:
            coords = cast(Tensor, rearrange("() n c -> e n c", coords, e=ensemble_size))
        elif coords.ndim == 3 and coords.shape[0] != ensemble_size:
            coords = cast(Tensor, rearrange("n c -> e n c", coords[0], e=ensemble_size))

        valid_shape = coords.ndim == 3 and coords.shape[0] == ensemble_size and coords.shape[2] == 3
        if not valid_shape:
            raise ValueError(
                f"coords shape should be ({ensemble_size}, N_atoms, 3), got {coords.shape}"
            )

        sigma = self.noise_schedule["sigma_tm"][int(noise_level)]
        if noise_level == 0:
            noisy_coords = sigma * torch.randn_like(coords)
        else:
            noisy_coords = coords + sigma * torch.randn_like(coords)

        return noisy_coords
