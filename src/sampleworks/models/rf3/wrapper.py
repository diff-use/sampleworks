import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from atomworks.enums import ChainType
from atomworks.ml.samplers import LoadBalancedDistributedSampler
from biotite.structure import AtomArray, AtomArrayStack
from jaxtyping import Float
from loguru import logger
from rf3.inference_engines import RF3InferenceEngine
from rf3.model.RF3 import RF3WithConfidence
from rf3.trainers.rf3 import assert_no_nans, RF3TrainerWithConfidence
from rf3.utils.inference import InferenceInput, InferenceInputDataset
from torch import Tensor
from torch.utils.data import DataLoader

from sampleworks.models.protocol import GenerativeModelInput
from sampleworks.utils.framework_utils import match_batch
from sampleworks.utils.guidance_constants import StructurePredictor
from sampleworks.utils.msa import MSAManager


@dataclass(frozen=True, slots=True)
class RF3Conditioning:
    """Conditioning tensors from RF3 trunk forward pass.

    Passable to diffusion module forward.

    Attributes
    ----------
    s_inputs : Tensor
        Input embeddings (S_inputs_I).
    s_trunk : Tensor
        Single representation from trunk (S_I).
    z_trunk : Tensor
        Pair representation from trunk (Z_II).
    features : dict[str, Any]
        Raw feature dict (f tensor).
    true_atom_array : AtomArray | None
        The AtomArray of the true structure, used for determining proper atom counts.
    """

    s_inputs: Tensor
    s_trunk: Tensor
    z_trunk: Tensor
    features: dict[str, Any]
    true_atom_array: AtomArray | None = None
    model_atom_array: AtomArray | None = None


@dataclass
class RF3Config:
    """Configuration for RF3 featurization.

    Attributes
    ----------
    msa_path : str | Path | dict | None
        MSA specification. Can be:
        - dict: chain_id -> MSA file path mapping
        - str/Path to .json: JSON file with chain_id -> MSA path mapping
        - str/Path to .a3m: Single MSA file applied to all protein chains
        - None: No MSA information is used
    ensemble_size : int
        Number of samples to generate (batch dimension of x_init).
    recycling_steps : int | None
        Number of recycling steps to perform. If None, uses model default.
    """

    msa_path: str | Path | dict | None = None
    ensemble_size: int = 1
    recycling_steps: int | None = None


def annotate_structure_for_rf3(
    structure: dict,
    *,
    msa_path: str | Path | dict | None = None,
    ensemble_size: int = 1,
    recycling_steps: int | None = None,
) -> dict:
    """Annotate an Atomworks structure with RF3-specific configuration.

    Parameters
    ----------
    structure : dict
        Atomworks structure dictionary.
    msa_path : str | Path | dict | None
        MSA specification for RF3.
    ensemble_size : int
        Number of samples to generate (batch dimension of x_init).
    recycling_steps : int | None
        Number of recycling steps to perform. If None, uses model default.

    Returns
    -------
    dict
        Structure dict with "_rf3_config" key added.
    """
    config = RF3Config(
        msa_path=msa_path,
        ensemble_size=ensemble_size,
        recycling_steps=recycling_steps,
    )
    return {**structure, "_rf3_config": config}


# TODO: This should go in some sort of atomworks utils module
def add_msa_to_chain_info(chain_info: dict, msa_path: str | Path | dict | None) -> dict:
    """Add MSA paths to chain_info dictionary.

    Parameters
    ----------
    chain_info: dict
        Original chain_info dictionary.
    msa_path: dict | str | Path | None
        MSA specification. Can be:
        - dict: chain_id -> MSA file path mapping
        - str/Path to .json: JSON file with chain_id -> MSA path mapping
        - str/Path to .a3m: Single MSA file applied to all protein chains
        - None: No MSA information is used

    Returns
    -------
    dict
        Updated chain_info dictionary with MSA paths.
    """
    updated_chain_info = chain_info.copy()

    if msa_path is None:
        return updated_chain_info

    # If msa_path is a JSON file, read it to get chain_id -> msa_path mapping
    if isinstance(msa_path, (str, Path)):
        msa_path_obj = Path(msa_path)
        if msa_path_obj.suffix == ".json" and msa_path_obj.exists():
            with open(msa_path_obj) as f:
                msa_path = json.load(f)

    # InferenceInput expects msa_path in chain_info
    for chain_id in updated_chain_info:
        if updated_chain_info[chain_id]["chain_type"] == ChainType.POLYPEPTIDE_L:
            if isinstance(msa_path, dict):
                chain_msa_path = msa_path.get(chain_id, None)
            else:
                chain_msa_path = msa_path

            if chain_msa_path is not None:
                updated_chain_info[chain_id]["msa_path"] = chain_msa_path

    return updated_chain_info


class RF3Wrapper:
    """Wrapper for RosettaFold 3 (Baker Lab AlphaFold 3 replication)."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        msa_manager: MSAManager | None = None,
    ):
        """
        Parameters
        ----------
        checkpoint_path: str | Path
            Filesystem path to the checkpoint containing trained weights.
        msa_manager: MSAManager | None
            MSA manager for retrieving MSAs for input structures.
        """
        logger.info("Loading RF3 Inference Engine")

        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        self.msa_manager = msa_manager
        self.msa_pairing_strategy = "greedy"

        # TODO: expose num_steps, num_recycles to user

        self.inference_engine = RF3InferenceEngine(
            ckpt_path=str(self.checkpoint_path),
            diffusion_batch_size=1,
        )
        self.inference_engine.initialize()

        self.inference_engine.trainer = cast(
            RF3TrainerWithConfidence, self.inference_engine.trainer
        )
        self.model = self.inference_engine.trainer.state["model"]
        self._device = self.inference_engine.trainer.fabric.device

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def _inner_model(self) -> RF3WithConfidence:
        """Access the unwrapped RF3WithConfidence model through EMA wrappers."""
        model = self.model
        if hasattr(model, "shadow"):
            # The model is wrapped with an ExponentialMovingAverage (EMA) wrapper
            # To access the EMA weights that we want to use for inference, we need
            # to access the `shadow` attribute (see AF3 Supplement section 5.6, RF3 preprint
            # supplement A.4.2)
            model = model.shadow
        return cast(RF3WithConfidence, model)

    def featurize(self, structure: dict) -> GenerativeModelInput[RF3Conditioning]:
        """From an Atomworks structure, calculate RF3 input features.

        Runs trunk forward pass and initializes x_init from prior distribution.

        Parameters
        ----------
        structure: dict
            Atomworks structure dictionary. Can be annotated with RF3 config
            via `annotate_structure_for_rf3()`. Config is read from
            `structure["_rf3_config"]` if present, otherwise default RF3Config
            values are used.

        Returns
        -------
        GenerativeModelInput[RF3Conditioning]
            Model input with x_init and trunk conditioning.
        """
        config = structure.get("_rf3_config", RF3Config())
        if isinstance(config, dict):
            config = RF3Config(**config)

        msa_path = config.msa_path
        ensemble_size = config.ensemble_size
        recycling_steps = config.recycling_steps

        if "asym_unit" not in structure:
            raise ValueError("structure must contain 'asym_unit' key")

        atom_array = structure["asym_unit"]
        chain_info = structure.get("chain_info", {})

        # if we have an MSAManager, then use it to get msa_paths unless they've been overridden
        # I'm using one of two possible sequences, which has
        #  non-canonicals filtered out.
        if self.msa_manager is not None and msa_path is None and chain_info:
            polypeptides = {
                chain_id: item["processed_entity_canonical_sequence"]
                for chain_id, item in chain_info.items()
                if item["chain_type"] == ChainType.POLYPEPTIDE_L
            }
            msa_path = self.msa_manager.get_msa(
                polypeptides, self.msa_pairing_strategy, structure_predictor=StructurePredictor.RF3
            )

            # These are debugging assertions.
            assert all(isinstance(pp, str) for pp in polypeptides.values())

        logger.info(f"Using MSA paths: {msa_path}")

        chain_info = add_msa_to_chain_info(chain_info, msa_path)

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

        features = trainer._assemble_network_inputs(pipeline_output)

        assert_no_nans(
            features,
            msg=f"network_input for example_id: {pipeline_output['example_id']}",
        )

        pairformer_out = self._pairformer_pass(
            features, grad_needed=False, recycling_steps=recycling_steps or 10
        )

        true_atom_array: AtomArray = (
            cast(AtomArray, atom_array[0]) if isinstance(atom_array, AtomArrayStack) else atom_array
        )

        num_atoms = len(pairformer_out["features"]["atom_to_token_map"])

        # Build model atom array from non-hydrogen InferenceInput atoms
        model_aa = cast(
            AtomArray, inference_input.atom_array[inference_input.atom_array.element != "H"]
        )
        # RF3 feature assembly preserves inference_input atom order after hydrogen filtering.
        # Any excess atoms are trailing entries not represented in atom_to_token_map.
        if len(model_aa) > num_atoms:
            model_aa = cast(AtomArray, model_aa[:num_atoms])
        if not hasattr(model_aa, "occupancy") or model_aa.occupancy is None:
            model_aa.set_annotation("occupancy", np.ones(len(model_aa), dtype=np.float32))
        if not hasattr(model_aa, "b_factor") or model_aa.b_factor is None:
            model_aa.set_annotation("b_factor", np.full(len(model_aa), 20.0, dtype=np.float32))

        conditioning = RF3Conditioning(
            s_inputs=pairformer_out["s_inputs"],
            s_trunk=pairformer_out["s_trunk"],
            z_trunk=pairformer_out["z_trunk"],
            features=pairformer_out["features"],
            true_atom_array=true_atom_array,
            model_atom_array=model_aa,
        )

        # x_init should be the reference coordinates for alignment purposes.
        if true_atom_array is not None and len(true_atom_array) == num_atoms:
            x_init = torch.tensor(true_atom_array.coord, device=self.device, dtype=torch.float32)
            x_init = match_batch(x_init.unsqueeze(0), target_batch_size=ensemble_size).clone()
        else:
            logger.warning(
                "True structure not available or atom count mismatch; initializing "
                "x_init from prior. This means align_to_input will not work properly,"
                " and reward functions dependent on this won't be accurate."
            )
            x_init = self.initialize_from_prior(batch_size=ensemble_size, shape=(num_atoms, 3))

        return GenerativeModelInput(x_init=x_init, conditioning=conditioning)

    def _pairformer_pass(
        self, features: dict[str, Any], grad_needed: bool = False, recycling_steps: int = 10
    ) -> dict[str, Any]:
        """Perform a pass through the RF3 trunk to obtain representations.

        Internal method that computes trunk representations.

        Parameters
        ----------
        features: dict[str, Any]
            Model features dict that is computed internally in featurize
            (raw features, not GenerativeModelInput).
        grad_needed: bool, optional
            Whether gradients are needed for this pass, by default False.
        **kwargs: dict, optional
            Additional arguments.

            - recycling_steps: int
                Number of recycling steps to perform. Defaults to n_recycles.

        Returns
        -------
        dict[str, Any]
            Trunk outputs (s_inputs, s_trunk, z_trunk, features).
        """

        with (
            torch.set_grad_enabled(grad_needed),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):  # TODO: bfloat16 will require newer GPU generations and new CUDA
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

    def step(
        self,
        x_t: Float[Tensor, "batch atoms 3"],
        t: Float[Tensor, "*batch"] | float,
        *,
        features: GenerativeModelInput[RF3Conditioning] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        r"""Perform denoising at given timestep/noise level.

        Returns predicted clean sample :math:`\hat{x}_\theta`.

        Parameters
        ----------
        x_t : Float[Tensor, "batch atoms 3"]
            Noisy structure at timestep :math:`t`.
        t : Float[Tensor, "*batch"] | float
            Current timestep/noise level (:math:`\hat{t}` from noise schedule).
        features : GenerativeModelInput[RF3Conditioning] | None
            Model features as returned by ``featurize``.

        Returns
        -------
        Float[Tensor, "batch atoms 3"]
            Predicted clean sample coordinates.
        """
        if features is None or features.conditioning is None:
            raise ValueError("features with conditioning required for step()")

        cond = features.conditioning
        if not isinstance(x_t, torch.Tensor):
            x_t = torch.tensor(x_t, device=self.device, dtype=torch.float32)

        if isinstance(t, (int, float)):
            t_tensor = torch.tensor([t], device=self.device, dtype=x_t.dtype)
        else:
            t_tensor = t.to(device=self.device, dtype=x_t.dtype)
            if t_tensor.ndim == 0:
                t_tensor = t_tensor.unsqueeze(0)

        t_tensor = match_batch(t_tensor, target_batch_size=x_t.shape[0])

        with torch.autocast(
            "cuda", dtype=torch.float32
        ):  # TODO: bfloat16 will require newer GPU generations and new CUDA
            atom_coords_denoised: Tensor = self._inner_model.diffusion_module(
                X_noisy_L=x_t,
                t=t_tensor,
                f=cond.features,
                S_inputs_I=cond.s_inputs,
                S_trunk_I=cond.s_trunk,
                Z_trunk_II=cond.z_trunk,
            )

        return atom_coords_denoised.float()

    def initialize_from_prior(
        self,
        batch_size: int,
        features: GenerativeModelInput[RF3Conditioning] | None = None,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        """Create initial samples from the prior distribution.

        Parameters
        ----------
        batch_size : int
            Number of samples to generate.
        features : GenerativeModelInput[RF3Conditioning] | None, optional
            Model features as returned by `featurize`. Useful for determining shape.
        shape : tuple[int, ...] | None, optional
            Explicit shape of the generated state (in the form [num_atoms, 3]).
            NOTE: shape will override features if both are provided.

        Returns
        -------
        Float[Tensor, "batch atoms 3"]
            Gaussian initialized coordinates.

        Raises
        ------
        ValueError
            If both features and shape are None, or if shape is invalid.
        """
        if shape is not None:
            if len(shape) != 2 or shape[1] != 3:
                raise ValueError("shape must be of the form (num_atoms, 3)")
            return torch.randn((batch_size, *shape), device=self.device)

        if features is None or features.conditioning is None:
            raise ValueError("Either features or shape must be provided to initialize_from_prior()")

        cond = features.conditioning
        num_atoms = len(cond.features["atom_to_token_map"])

        return torch.randn((batch_size, num_atoms, 3), device=self.device)
