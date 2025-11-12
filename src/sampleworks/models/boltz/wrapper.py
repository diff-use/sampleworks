"""
Wrapper classes for Boltz models.
Follows the protocol in model_wrapper_protocol.py
to allow dependency injection/interchangeable use in sampling pipelines.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch
from atomworks.enums import ChainType
from boltz.data.module.inference import BoltzInferenceDataModule
from boltz.data.module.inferencev2 import Boltz2InferenceDataModule
from boltz.data.pad import pad_dim
from boltz.data.types import Manifest
from boltz.main import (
    Boltz2DiffusionParams,
    BoltzDiffusionParams,
    BoltzProcessedInput,
    BoltzSteeringParams,
    check_inputs,
    MSAModuleArgs,
    PairformerArgs,
    PairformerArgsV2,
    process_inputs,
)
from boltz.model.models.boltz1 import Boltz1
from boltz.model.models.boltz2 import Boltz2
from boltz.model.modules.utils import center_random_augmentation
from einops import einsum
from jaxtyping import ArrayLike, Float
from torch import Tensor


def weighted_rigid_align_differentiable(
    true_coords,
    pred_coords,
    weights,
    mask,
    allow_gradients: bool = True,
):
    """Compute weighted alignment with optional gradient preservation.

    Identical to boltz.model.loss.diffusion.weighted_rigid_align but without
    the detach_() call when allow_gradients=True, enabling gradient flow.

    I preserve the same parameter names as the original function, but note that
    true_coords will be aligned to the pred_coords in both implementations.

    Parameters
    ----------
    true_coords: torch.Tensor
        The ground truth atom coordinates
    pred_coords: torch.Tensor
        The predicted atom coordinates
    weights: torch.Tensor
        The weights for alignment
    mask: torch.Tensor
        The atoms mask
    allow_gradients: bool, optional
        If True, preserve gradients through alignment. If False, detach
        (matches original Boltz behavior). Default: True

    Returns
    -------
    torch.Tensor
        Aligned coordinates

    """
    batch_size, num_points, dim = true_coords.shape
    weights = (mask * weights).unsqueeze(-1)

    # Compute weighted centroids
    true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )

    # Center the coordinates
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    if num_points < (dim + 1):
        print(
            "Warning: The size of one of the point clouds is <= dim+1. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the weighted covariance matrix
    cov_matrix = einsum(
        weights * pred_coords_centered, true_coords_centered, "b n i, b n j -> b i j"
    )

    # Compute the SVD of the covariance matrix, required float32 for svd and determinant
    original_dtype = cov_matrix.dtype
    cov_matrix_32 = cov_matrix.to(dtype=torch.float32)
    U, S, V = torch.linalg.svd(
        cov_matrix_32, driver="gesvd" if cov_matrix_32.is_cuda else None
    )
    V = V.mH

    # Catch ambiguous rotation by checking the magnitude of singular values
    if (S.abs() <= 1e-15).any() and not (num_points < (dim + 1)):
        print(
            "Warning: Excessively low rank of "
            + "cross-correlation between aligned point clouds. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    # Compute the rotation matrix
    rot_matrix = torch.einsum("b i j, b k j -> b i k", U, V).to(dtype=torch.float32)

    # Ensure proper rotation matrix with determinant 1
    F = torch.eye(dim, dtype=cov_matrix_32.dtype, device=cov_matrix.device)[
        None
    ].repeat(batch_size, 1, 1)
    F[:, -1, -1] = torch.det(rot_matrix)
    rot_matrix = einsum(U, F, V, "b i j, b j k, b l k -> b i l")
    rot_matrix = rot_matrix.to(dtype=original_dtype)

    # Apply the rotation and translation
    aligned_coords = (
        einsum(true_coords_centered, rot_matrix, "b n i, b j i -> b n j")
        + pred_centroid
    )

    # Conditionally detach based on allow_gradients
    if not allow_gradients:
        aligned_coords.detach_()

    return aligned_coords


@dataclass
class PredictArgs:
    """Arguments for model prediction."""

    recycling_steps: int = 3  # default in Boltz1
    sampling_steps: int = 200
    diffusion_samples: int = (
        1  # number of samples you want to generate, will be used as multiplicity
    )
    write_confidence_summary: bool = True
    write_full_pae: bool = False
    write_full_pde: bool = False


def create_boltz_input_from_structure(structure: dict, out_dir: str | Path) -> Path:
    """Creates Boltz YAML file from an Atomworks parsed structure file.

    Parameters
    ----------
    structure : dict
        Atomworks parsed structure.
    out_dir : str | Path
        Path to write the YAML in.

    Returns
    -------
    Path
        Path to the written YAML file.
    """
    out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
    out_dir = out_dir.expanduser().resolve()

    polymer_info = {}
    ligand_info = {}

    chain_info = structure.get("chain_info", {})

    # Process atomworks chain_info into boltz YAML info
    # TODO: Handle non-canonical AAs in a polymer
    # TODO: Handle templates
    # TODO: Handle constraints
    for chain in chain_info:
        # Process protein, DNA, RNA chains
        entity_type: ChainType = chain_info[chain]["chain_type"]
        if entity_type.is_polymer():
            polymer_info[chain] = {}
            polymer_info[chain]["entity_type"] = (
                "protein"
                if entity_type.is_protein()
                else "DNA"
                if entity_type.is_nucleic_acid() and "DEOXY" in entity_type.to_string()
                else "RNA"
            )
            polymer_info[chain]["sequence"] = chain_info[chain][
                "processed_entity_canonical_sequence"
            ]  # Get canonical sequence
            if "CYCLIC" in entity_type.to_string():
                polymer_info[chain]["cyclic"] = True
        else:  # Ligand
            ligand_info[chain] = {}
            ligand_info[chain]["entity_type"] = "ligand"
            ligand_info[chain]["ccd"] = chain_info[chain]["res_name"][0]

    boltz_input_path = out_dir / "boltz_input.yaml"
    boltz_input_path.parent.mkdir(parents=True, exist_ok=True)
    with open(boltz_input_path, "w") as f:
        f.write("sequences:\n")
        for chain_id, info in polymer_info.items():
            f.write(f"    - {info['entity_type']}:\n")
            f.write(f"        id: {chain_id}\n")
            f.write(f"        sequence: {info['sequence']}\n")
            if info.get("cyclic", False):
                f.write("        cyclic: true\n")
        if ligand_info:
            for chain_id, info in ligand_info.items():
                f.write(f"    - {info['entity_type']}:\n")
                f.write(f"        id: {chain_id}\n")
                f.write(f"        ccd: {info['ccd']}\n")

    return boltz_input_path


class Boltz2Wrapper:
    """
    Wrapper for Boltz2 model.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        use_msa_server: bool = True,
        predict_args: PredictArgs = PredictArgs(),
        diffusion_args: Boltz2DiffusionParams = Boltz2DiffusionParams(),
        steering_args: BoltzSteeringParams = BoltzSteeringParams(),
        method: str = "MD",
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """
        Parameters
        ----------
        checkpoint_path : str | Path
            Filesystem path to the Boltz2 checkpoint containing trained weights.
        use_msa_server : bool, optional
            If ``True``, fetch MSA features from the ColabFold server; otherwise rely
            on precomputed MSAs.
        predict_args : PredictArgs, optional
            Runtime prediction configuration such as recycling depth and sampling
            steps.
        diffusion_args : Boltz2DiffusionParams, optional
            Diffusion process parameters passed down to the Boltz2 model.
        steering_args : BoltzSteeringParams, optional
            Steering configuration controlling external potentials applied during
            sampling.
        method : str, optional
            Inference method identifier understood by Boltz2 (e.g. ``"MD"``).
        device : torch.device, optional
            Device to run the model on, by default CUDA if available.
        """
        self.checkpoint_path = checkpoint_path
        self.use_msa_server = use_msa_server
        self.predict_args = predict_args
        self.diffusion_args = diffusion_args
        self.steering_args = steering_args
        self.method = method
        self.device = torch.device(device)
        # NOTE: assumes checkpoint and ccd dictionary get downloaded to the same place
        self.cache_path = (
            (
                Path(checkpoint_path)
                if isinstance(checkpoint_path, str)
                else checkpoint_path
            )
            .parent.expanduser()
            .resolve()
        )
        self.cache_path.mkdir(parents=True, exist_ok=True)

        pairformer_args = PairformerArgsV2()

        msa_args = MSAModuleArgs(
            subsample_msa=True,  # Default from boltz repo
            num_subsampled_msa=1024,  # Default from boltz repo
            use_paired_feature=True,  # Required for Boltz2
        )

        self.model = (
            Boltz2.load_from_checkpoint(
                checkpoint_path,
                strict=True,
                predict_args=asdict(predict_args),
                map_location="cpu",
                diffusion_process_args=asdict(diffusion_args),
                ema=False,
                pairformer_args=asdict(pairformer_args),
                msa_args=asdict(msa_args),
                steering_args=asdict(steering_args),
            )
            .to(self.device)
            .eval()
        )

        self.data_module: Boltz2InferenceDataModule
        self.cached_representations: dict[str, Any] = {}

        sigmas = self.model.structure_module.sample_schedule(
            self.predict_args.sampling_steps
        )
        gammas = torch.where(
            sigmas > self.model.structure_module.gamma_min,
            self.model.structure_module.gamma_0,
            0.0,
        )
        self.noise_schedule: dict[str, Float[Tensor, ...]] = {
            "sigma_tm": sigmas[:-1],
            "sigma_t": sigmas[1:],
            "gamma": gammas[1:],
        }

    def _setup_data_module(
        self,
        input_path: str | Path,
        out_dir: str | Path,
        num_workers: int = 8,
    ):
        """Create the Lightning data module used by Boltz to serve data to the model.

        Parameters
        ----------
        input_path : str | Path
            Path to the input Boltz YAML file.
        out_dir : str | Path
            Directory to output processed input.
        num_workers : int, optional
            Number of parallel workers for input data processing, by default 8
        """
        input_path = Path(input_path) if isinstance(input_path, str) else input_path
        out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
        input_path = input_path.expanduser().resolve()
        ccd_path = self.cache_path / "ccd.pkl"
        mol_dir = self.cache_path / "mols"

        data = check_inputs(input_path)

        process_inputs(
            data=data,
            out_dir=out_dir,
            ccd_path=ccd_path,
            mol_dir=mol_dir,
            use_msa_server=self.use_msa_server,
            msa_server_url="https://api.colabfold.com",
            msa_pairing_strategy="greedy",
            boltz2=True,
            preprocessing_threads=1,
        )

        processed_dir = out_dir / "processed"
        processed = BoltzProcessedInput(
            manifest=Manifest.load(processed_dir / "manifest.json"),  # type: ignore (Boltz repo doesn't have the right type hints?)
            targets_dir=processed_dir / "structures",
            msa_dir=processed_dir / "msa",
            constraints_dir=(processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None,
            template_dir=processed_dir / "templates"
            if (processed_dir / "templates").exists()
            else None,
            extra_mols_dir=processed_dir / "mols"
            if (processed_dir / "mols").exists()
            else None,
        )

        self.data_module = Boltz2InferenceDataModule(
            manifest=processed.manifest,
            target_dir=processed.targets_dir,
            msa_dir=processed.msa_dir,
            mol_dir=mol_dir,
            num_workers=num_workers if num_workers is not None else 8,
            constraints_dir=processed.constraints_dir,
            template_dir=processed_dir / "templates"
            if (processed_dir / "templates").exists()
            else None,
            extra_mols_dir=processed_dir / "mols"
            if (processed_dir / "mols").exists()
            else None,
            override_method=self.method,
        )

    def featurize(self, structure: dict, **kwargs) -> dict[str, Any]:
        """From an Atomworks structure, calculate Boltz-2 input features.

        Parameters
        ----------
        structure : dict
            Atomworks structure dictionary. [See Atomworks documentation](https://baker-laboratory.github.io/atomworks-dev/latest/io/parser.html#atomworks.io.parser.parse)
        **kwargs : dict, optional
            Additional keyword arguments for Boltz-2 featurization.

            - out_dir: str | Path
                Output directory for processed Boltz intermediate files. Defaults first
                to the structure ID from metadata, then to "boltz2_output" in the
                current working directory.
            - num_workers: int
                Number of parallel workers for input data processing. Defaults to 8.

        Returns
        -------
        dict[str, Any]
            Boltz-2 input features (from dataloader).
        """
        # Side effect: creates Boltz input YAML file in out_dir
        input_path = create_boltz_input_from_structure(
            structure,
            kwargs.get(
                "out_dir", structure.get("metadata", {}).get("id", "boltz2_output")
            ),
        )

        # Side effect: creates files in the processed directory of out_dir
        self._setup_data_module(
            input_path,
            kwargs.get("out_dir", "boltz2_output"),
            num_workers=kwargs.get("num_workers", 8),
        )

        batch = self.data_module.transfer_batch_to_device(
            next(iter(self.data_module.predict_dataloader())), self.device, 0
        )

        return batch

    def step(
        self, features: dict[str, Any], grad_needed: bool = False, **kwargs
    ) -> dict[str, Any]:
        """
        Perform a pass through the Pairformer module to obtain output, which can then be
        passed into the diffusion module. Pretty much only here to match the protocol
        and be used in featurize, but could be useful for doing exploration in
        the Boltz embedding space.

        Parameters
        ----------
        features : dict[str, Any]
            Model features as returned by `featurize`.
        grad_needed : bool, optional
            Whether gradients are needed for this pass, by default False.
        **kwargs : dict, optional
            Additional keyword arguments for Boltz-2 Pairformer step.

            - recycling_steps: int
                Number of recycling steps to perform. Defaults to the value in
                predict_args.


        Returns
        -------
        dict[str, Any]
            Boltz-2 Pairformer outputs.
        """
        with torch.set_grad_enabled(grad_needed):
            mask: Tensor = features["token_pad_mask"]
            pair_mask = mask[:, :, None] * mask[:, None, :]
            s_inputs = self.model.input_embedder(features)

            s_init = self.model.s_init(s_inputs)
            z_init = (
                self.model.z_init_1(s_inputs)[:, :, None]
                + self.model.z_init_2(s_inputs)[:, None, :]
            )

            relative_position_encoding = self.model.rel_pos(features)

            z_init = z_init + relative_position_encoding
            z_init = z_init + self.model.token_bonds(features["token_bonds"].float())

            if self.model.bond_type_feature:
                z_init = z_init + self.model.token_bonds_type(
                    features["type_bonds"].long()
                )
            z_init = z_init + self.model.contact_conditioning(features)

            s, z = torch.zeros_like(s_init), torch.zeros_like(z_init)

            for _ in range(
                kwargs.get("recycling_steps", self.predict_args.recycling_steps) + 1
            ):  # 3 is Boltz-2 default
                s = s_init + self.model.s_recycle(self.model.s_norm(s))
                z = z_init + self.model.z_recycle(self.model.z_norm(z))

                if self.model.use_templates:
                    if self.model.is_template_compiled:
                        template_module = (
                            self.model.template_module._orig_mod  # type: ignore (compiled torch module has this attribute, type checker doesn't know)
                        )
                    else:
                        template_module = self.model.template_module

                    z = z + template_module(
                        z, features, pair_mask, use_kernels=self.model.use_kernels
                    )  # type: ignore (Object will be callable here)

                if self.model.is_msa_compiled:
                    msa_module = self.model.msa_module._orig_mod  # type: ignore (compiled torch module has this attribute, type checker doesn't know)
                else:
                    msa_module = self.model.msa_module

                z = z + msa_module(
                    z, s_inputs, features, use_kernels=self.model.use_kernels
                )  # type: ignore (Object will be callable here)

                if self.model.is_pairformer_compiled:
                    pairformer_module = self.model.pairformer_module._orig_mod  # type: ignore (compiled torch module has this attribute, type checker doesn't know)
                else:
                    pairformer_module = self.model.pairformer_module

                s, z = pairformer_module(s, z, mask=mask, pair_mask=pair_mask)  # type: ignore (Object will be callable here)

            q, c, to_keys, atom_enc_bias, atom_dec_bias, token_trans_bias = (
                self.model.diffusion_conditioning(
                    s_trunk=s,
                    z_trunk=z,
                    relative_position_encoding=relative_position_encoding,
                    feats=features,
                )
            )

        diffusion_conditioning = {
            "q": q,
            "c": c,
            "to_keys": to_keys,
            "atom_enc_bias": atom_enc_bias,
            "atom_dec_bias": atom_dec_bias,
            "token_trans_bias": token_trans_bias,
        }

        return {
            "s": s,
            "z": z,
            "s_inputs": s_inputs,
            "relative_position_encoding": relative_position_encoding,
            "feats": features,
            "diffusion_conditioning": diffusion_conditioning,
        }

    def denoise_step(
        self,
        features: dict[str, Any],
        noisy_coords: Float[ArrayLike | Tensor, "..."],
        timestep: float | int,
        grad_needed: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Perform denoising at given timestep/noise level.
        Returns predicted clean sample or predicted noise depending on
        model parameterization.

        Parameters
        ----------
        features : dict[str, Any]
            Model features produced by :meth:`featurize` or :meth:`step`.
        noisy_coords : Float[Tensor, "..."]
            Noisy atom coordinates at the current timestep.
        timestep : int
            Current timestep/noise level - for Boltz, this is the reverse timestep.
            This means that it starts from 0 and goes up to (sampling_steps - 1), so it
            is the "reverse" diffusion time.
        grad_needed : bool, optional
            Whether gradients are needed for this pass, by default False.
        **kwargs : dict, optional
            Additional keyword arguments for Boltz-2 denoising.

            t_hat (float, optional)
                Precomputed t_hat value for the current timestep; if not provided, it
                will be computed internally.

            eps (Tensor, optional)
                Precomputed noise tensor for the current timestep; if not provided, it
                will be sampled internally.

            augmentation (bool, optional)
                Apply `center_random_augmentation` when True (default True).

            align_to_input (bool, optional)
                Align denoised coordinates to `input_coords` when True (default True).

            input_coords (Tensor, optional)
                Reference coordinates required if `align_to_input` is True.

            alignment_weights (Tensor, optional)
                Atom weights for alignment; defaults to `atom_mask`.

            multiplicity (int, optional)
                Overrides the multiplicity passed to the diffusion network, which is the
                replicating the model does internally along axis=0; defaults to the
                batch size of `noisy_coords`.

            overwrite_representations (bool, optional)
                Whether to overwrite cached representations (default False).
                Do this if you are running a new input sample through the model.

            recycling_steps (int, optional)
                Number of recycling steps to perform (default 3 from PredictArgs).
                This will only be applied if overwrite_representations is True, as
                it is passed to the pairformer module computation that is ideally
                cached for efficiency.

            allow_alignment_gradients (bool, optional)
                Whether to allow gradients through the alignment step (default True).
                If False, detaches after alignment (matches original Boltz behavior).

        Returns
        -------
        dict[str, Tensor]
            Dictionary containing ``"atom_coords_denoised"`` with the cleaned
            coordinate tensor.
        """

        if not self.cached_representations or kwargs.get(
            "overwrite_representations", False
        ):
            # Side effect: overwrites class attribute
            self.cached_representations = self.step(
                features,
                grad_needed=grad_needed,
                recycling_steps=kwargs.get(
                    "recycling_steps", self.predict_args.recycling_steps
                ),
            )

        features = self.cached_representations

        if not isinstance(noisy_coords, torch.Tensor):
            noisy_coords = torch.tensor(
                noisy_coords, device=self.device, dtype=torch.float32
            )

        s = features.get("s", None)
        z = features.get("z", None)
        s_inputs = features.get("s_inputs", None)
        relative_position_encoding = features.get("relative_position_encoding", None)
        # These are the input features to the conditioning networks
        feats = features.get("feats", None)

        if any(
            x is None for x in [s, z, s_inputs, relative_position_encoding, features]
        ):
            raise ValueError("Missing required features for denoise_step")

        # shape [1, N_padded]
        feats = cast(dict[str, Any], feats)
        atom_mask = feats.get("atom_pad_mask")
        atom_mask = cast(Tensor, atom_mask)
        # shape [batch_size, N_padded]
        atom_mask = atom_mask.repeat_interleave(noisy_coords.shape[0], dim=0)

        with torch.set_grad_enabled(grad_needed):
            pad_len = atom_mask.shape[1] - noisy_coords.shape[1]
            if pad_len >= 0:
                padded_noisy_coords = pad_dim(noisy_coords, dim=1, pad_len=pad_len)
            else:
                raise ValueError("pad_len is negative, cannot pad noisy_coords")

            if "t_hat" in kwargs and "eps" in kwargs:
                t_hat = kwargs["t_hat"]
                eps = cast(Tensor, kwargs["eps"])
                if pad_len > 0 and eps.shape[1] != padded_noisy_coords.shape[1]:
                    eps = pad_dim(eps, dim=1, pad_len=pad_len)
            else:
                timestep_scaling = self.get_timestep_scaling(timestep)
                eps = timestep_scaling["eps_scale"] * torch.randn(
                    padded_noisy_coords.shape, device=self.device
                )
                t_hat = timestep_scaling["t_hat"]

            if kwargs.get("augmentation", True):
                padded_noisy_coords = center_random_augmentation(
                    padded_noisy_coords,
                    atom_mask=atom_mask,
                    augmentation=True,
                )

            padded_noisy_coords_eps = cast(Tensor, padded_noisy_coords) + eps

            padded_atom_coords_denoised = (
                self.model.structure_module.preconditioned_network_forward(
                    padded_noisy_coords_eps,
                    t_hat,
                    network_condition_kwargs=dict(
                        multiplicity=kwargs.get(
                            "multiplicity", padded_noisy_coords_eps.shape[0]
                        ),
                        s_inputs=s_inputs,
                        s_trunk=s,
                        feats=feats,
                        diffusion_conditioning=features["diffusion_conditioning"],
                    ),
                )
            )

            if kwargs.get("align_to_input", True):
                input_coords = kwargs.get("input_coords")
                if input_coords is not None:
                    if (
                        pad_len > 0
                        and input_coords.shape[1]
                        != padded_atom_coords_denoised.shape[1]
                    ):
                        input_coords = pad_dim(input_coords, dim=1, pad_len=pad_len)

                    alignment_weights = kwargs.get("alignment_weights", atom_mask)
                    allow_alignment_gradients = kwargs.get(
                        "allow_alignment_gradients", False
                    )

                    if allow_alignment_gradients:
                        padded_atom_coords_denoised = (
                            weighted_rigid_align_differentiable(
                                padded_atom_coords_denoised.float(),
                                cast(Tensor, input_coords).float(),
                                weights=alignment_weights,
                                mask=atom_mask,
                                allow_gradients=True,
                            )
                        )
                    else:
                        padded_atom_coords_denoised = (
                            weighted_rigid_align_differentiable(
                                padded_atom_coords_denoised.float(),
                                cast(Tensor, input_coords).float(),
                                weights=alignment_weights,
                                mask=atom_mask,
                                allow_gradients=False,
                            )
                        )
                else:
                    raise ValueError(
                        "Input coordinates must be provided when align_to_input "
                        "is True."
                    )

            atom_coords_denoised = padded_atom_coords_denoised[
                atom_mask.bool(), :
            ].reshape(noisy_coords.shape[0], -1, 3)

        return {"atom_coords_denoised": atom_coords_denoised}

    def get_noise_schedule(self) -> Mapping[str, Float[ArrayLike | Tensor, "..."]]:
        """
        Return the full noise schedule with semantic keys.

        Returns
        -------
        Mapping[str, Float[ArrayLike | Tensor, "..."]]
            Noise schedule arrays.
            Sigma at time t-1: "sigma_tm"
            Sigma at time t: "sigma_t"
            Gamma at time t: "gamma"
        """
        return self.noise_schedule

    def get_timestep_scaling(self, timestep: float | int) -> dict[str, float]:
        """
        Return scaling constants for Boltz.

        Parameters
        ----------
        timestep : float | int
            Current timestep/noise level. (starts from 0)

        Returns
        -------
        dict[str, float]
            Scaling constants.
            "t_hat", "sigma_t", "eps_scale"
        """
        if timestep < 0 or timestep >= self.predict_args.sampling_steps:
            raise ValueError(
                f"timestep {timestep} is out of bounds for sampling steps "
                f"{self.predict_args.sampling_steps}"
            )

        sigma_tm = self.noise_schedule["sigma_tm"][int(timestep)]
        sigma_t = self.noise_schedule["sigma_t"][int(timestep)]
        gamma = self.noise_schedule["gamma"][int(timestep)]

        t_hat = sigma_tm * (1 + gamma)
        eps_scale = self.model.structure_module.noise_scale * torch.sqrt(
            t_hat**2 - sigma_tm**2
        )

        return {
            "t_hat": t_hat.item(),
            "sigma_t": sigma_t.item(),
            "eps_scale": eps_scale.item(),
        }

    def initialize_from_noise(
        self, structure: dict, noise_level: float | int, **kwargs
    ) -> Float[ArrayLike | Tensor, "*batch _num_atoms 3"]:
        """Create a noisy version of structure's coordinates at given noise level.

        Parameters
        ----------
        structure : dict
            Atomworks structure dictionary. [See Atomworks documentation](https://baker-laboratory.github.io/atomworks-dev/latest/io/parser.html#atomworks.io.parser.parse)
        noise_level : float | int
            Timestep/noise level - for Boltz, this is the reverse timestep.
            This means that it starts from 0 and goes up to (sampling_steps - 1), so it
            is the "reverse" diffusion time.
        **kwargs : dict, optional
            Additional keyword arguments needed for classes that implement this Protocol

        Returns
        -------
        Float[ArrayLike | Tensor, "*batch _num_atoms 3"]
            Noisy structure coordinates for atoms that have nonzero occupancy.
        """
        if "asym_unit" not in structure:
            raise ValueError(
                "structure must contain 'asym_unit' key to access"
                "the coordinates in the asymmetric unit."
            )

        if noise_level < 0 or noise_level >= self.predict_args.sampling_steps:
            raise ValueError(
                f"noise_level {noise_level} is out of bounds for sampling steps "
                f"{self.predict_args.sampling_steps}"
            )

        # We need to make sure the missing atoms are handled correctly
        residues_with_occupancy = structure["asym_unit"].occupancy > 0
        coords = structure["asym_unit"].coord[:, residues_with_occupancy]

        if isinstance(coords, ArrayLike):
            coords = torch.tensor(coords, device=self.device, dtype=torch.float32)

        if noise_level == 0:
            noisy_coords = self.noise_schedule["sigma_tm"][0] * torch.randn_like(coords)
        else:
            noisy_coords = coords + self.noise_schedule["sigma_tm"][
                int(noise_level)
            ] * torch.randn_like(coords)

        return noisy_coords


class Boltz1Wrapper:
    """Wrapper for Boltz1 model."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        use_msa_server: bool = True,
        predict_args: PredictArgs = PredictArgs(),
        diffusion_args: BoltzDiffusionParams = BoltzDiffusionParams(),
        steering_args: BoltzSteeringParams = BoltzSteeringParams(),
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """
        Parameters
        ----------
        checkpoint_path : str | Path
            Filesystem path to the Boltz1 checkpoint containing trained weights.
        use_msa_server : bool, optional
            If ``True``, fetch MSA features from the ColabFold server; otherwise rely
            on precomputed MSAs.
        predict_args : PredictArgs, optional
            Runtime prediction configuration such as recycling depth and sampling
            steps.
        diffusion_args : BoltzDiffusionParams, optional
            Diffusion process parameters passed down to the Boltz1 model.
        steering_args : BoltzSteeringParams, optional
            Steering configuration controlling external potentials applied during
            sampling.
        device : torch.device, optional
            Device to run the model on, by default CUDA if available.
        """
        self.checkpoint_path = checkpoint_path
        self.use_msa_server = use_msa_server
        self.predict_args = predict_args
        self.diffusion_args = diffusion_args
        self.steering_args = steering_args
        self.device = torch.device(device)
        # NOTE: assumes checkpoint and ccd dictionary get downloaded to the same place
        self.cache_path = (
            (
                Path(checkpoint_path)
                if isinstance(checkpoint_path, str)
                else checkpoint_path
            )
            .parent.expanduser()
            .resolve()
        )
        self.cache_path.mkdir(parents=True, exist_ok=True)

        pairformer_args = PairformerArgs()

        msa_args = MSAModuleArgs(
            subsample_msa=True,
            num_subsampled_msa=1024,
            use_paired_feature=False,
        )

        self.model = (
            Boltz1.load_from_checkpoint(
                checkpoint_path,
                strict=True,
                predict_args=asdict(predict_args),
                map_location="cpu",
                diffusion_process_args=asdict(diffusion_args),
                ema=False,
                use_kernels=True,
                pairformer_args=asdict(pairformer_args),
                msa_args=asdict(msa_args),
                steering_args=asdict(steering_args),
            )
            .to(self.device)
            .eval()
        )

        self.data_module: BoltzInferenceDataModule
        self.cached_representations: dict[str, Any] = {}

        sigmas = self.model.structure_module.sample_schedule(
            self.predict_args.sampling_steps
        )
        gammas = torch.where(
            sigmas > self.model.structure_module.gamma_min,
            self.model.structure_module.gamma_0,
            0.0,
        )
        self.noise_schedule: dict[str, Float[Tensor, ...]] = {
            "sigma_tm": sigmas[:-1],
            "sigma_t": sigmas[1:],
            "gamma": gammas[1:],
        }

    def _setup_data_module(
        self,
        input_path: str | Path,
        out_dir: str | Path,
        num_workers: int = 2,
    ):
        """Create the Lightning data module used by Boltz to serve data to the model.

        Parameters
        ----------
        input_path : str | Path
            Path to the input Boltz YAML file.
        out_dir : str | Path
            Directory to output processed input.
        num_workers : int, optional
            Number of parallel workers for input data processing, by default 2
        """
        input_path = Path(input_path) if isinstance(input_path, str) else input_path
        out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
        input_path = input_path.expanduser().resolve()
        ccd_path = self.cache_path / "ccd.pkl"
        mol_dir = self.cache_path / "mols"

        data = check_inputs(input_path)

        process_inputs(
            data=data,
            out_dir=out_dir,
            ccd_path=ccd_path,
            mol_dir=mol_dir,
            use_msa_server=self.use_msa_server,
            msa_server_url="https://api.colabfold.com",
            msa_pairing_strategy="greedy",
            boltz2=False,
            preprocessing_threads=1,
        )

        processed_dir = out_dir / "processed"
        processed = BoltzProcessedInput(
            manifest=Manifest.load(processed_dir / "manifest.json"),  # type: ignore (Boltz repo doesn't have the right type hints?)
            targets_dir=processed_dir / "structures",
            msa_dir=processed_dir / "msa",
            constraints_dir=(processed_dir / "constraints")
            if (processed_dir / "constraints").exists()
            else None,
            template_dir=processed_dir / "templates"
            if (processed_dir / "templates").exists()
            else None,
            extra_mols_dir=processed_dir / "mols"
            if (processed_dir / "mols").exists()
            else None,
        )

        self.data_module = BoltzInferenceDataModule(
            manifest=processed.manifest,
            target_dir=processed.targets_dir,
            msa_dir=processed.msa_dir,
            num_workers=num_workers if num_workers is not None else 2,
            constraints_dir=processed.constraints_dir,
        )

    def featurize(self, structure: dict, **kwargs) -> dict[str, Any]:
        """From an Atomworks structure, calculate model features.

        Parameters
        ----------
        structure : dict
            Atomworks structure dictionary. [See Atomworks documentation](https://baker-laboratory.github.io/atomworks-dev/latest/io/parser.html#atomworks.io.parser.parse)
        **kwargs : dict, optional
            Additional keyword arguments for Boltz-1 featurization.

            - out_dir: str | Path
                Output directory for processed Boltz intermediate files. Defaults first
                to the structure ID from metadata, then to "boltz1_output" in the
                current working directory.
            - num_workers: int
                Number of parallel workers for input data processing. Defaults to 2.

        Returns
        -------
        dict[str, Any]
            Boltz-1 input features (raw batch from dataloader).
        """
        # Side effect: creates Boltz input YAML file in out_dir
        input_path = create_boltz_input_from_structure(
            structure,
            kwargs.get(
                "out_dir", structure.get("metadata", {}).get("id", "boltz1_output")
            ),
        )

        # Side effect: creates files in the processed directory of out_dir
        self._setup_data_module(
            input_path,
            kwargs.get("out_dir", "boltz1_output"),
            num_workers=kwargs.get("num_workers", 2),
        )

        batch = self.data_module.transfer_batch_to_device(
            next(iter(self.data_module.predict_dataloader())), self.device, 0
        )

        return batch

    def get_noise_schedule(self) -> Mapping[str, Float[ArrayLike | Tensor, "..."]]:
        """
        Return the full noise schedule with semantic keys.

        Returns
        -------
        Mapping[str, Float[ArrayLike | Tensor, "..."]]
            Noise schedule arrays.
            Sigma at time t-1: "sigma_tm"
            Sigma at time t: "sigma_t"
            Gamma at time t: "gamma"
        """
        return self.noise_schedule

    def get_timestep_scaling(self, timestep: float | int) -> dict[str, float]:
        """
        Return scaling constants for Boltz1.

        Parameters
        ----------
        timestep : float | int
            Current timestep/noise level.

        Returns
        -------
        dict[str, float]
            Scaling constants.
            "t_hat", "sigma_t", "eps_scale"
        """

        if timestep < 0 or timestep >= self.predict_args.sampling_steps:
            raise ValueError(
                f"timestep {timestep} is out of bounds for sampling steps "
                f"{self.predict_args.sampling_steps}"
            )

        sigma_tm = self.noise_schedule["sigma_tm"][int(timestep)]
        sigma_t = self.noise_schedule["sigma_t"][int(timestep)]
        gamma = self.noise_schedule["gamma"][int(timestep)]

        t_hat = sigma_tm * (1 + gamma)
        eps_scale = self.model.structure_module.noise_scale * torch.sqrt(
            t_hat**2 - sigma_tm**2
        )

        return {
            "t_hat": t_hat.item(),
            "sigma_t": sigma_t.item(),
            "eps_scale": eps_scale.item(),
        }

    def step(
        self,
        features: dict[str, Any],
        grad_needed: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Perform a pass through the Pairformer module to obtain output, which can then be
        passed into the diffusion module.

        Parameters
        ----------
        features : dict[str, Any]
            Model features as returned by `featurize`.
        grad_needed : bool, optional
            Whether gradients are needed for this pass, by default False.
        **kwargs : dict, optional
            Additional keyword arguments for Boltz-1 Pairformer step.

            - recycling_steps: int
                Number of recycling steps to perform. Defaults to the value in
                predict_args.


        Returns
        -------
        dict[str, Any]
            Boltz-1 Pairformer outputs.
        """
        with torch.set_grad_enabled(grad_needed):
            mask: Tensor = features["token_pad_mask"]
            pair_mask = mask[:, :, None] * mask[:, None, :]
            s_inputs = self.model.input_embedder(features)

            s_init = self.model.s_init(s_inputs)
            z_init = (
                self.model.z_init_1(s_inputs)[:, :, None]
                + self.model.z_init_2(s_inputs)[:, None, :]
            )

            relative_position_encoding = self.model.rel_pos(features)
            z_init = z_init + relative_position_encoding
            z_init = z_init + self.model.token_bonds(features["token_bonds"].float())

            s, z = torch.zeros_like(s_init), torch.zeros_like(z_init)

            for _ in range(
                kwargs.get("recycling_steps", self.predict_args.recycling_steps) + 1
            ):  # 3 is Boltz-1 default
                s = s_init + self.model.s_recycle(self.model.s_norm(s))
                z = z_init + self.model.z_recycle(self.model.z_norm(z))

                if not self.model.no_msa:
                    z = z + self.model.msa_module(
                        z, s_inputs, features, use_kernels=self.model.use_kernels
                    )  # type: ignore (Object will be callable here)

                if self.model.is_pairformer_compiled:
                    pairformer_module = self.model.pairformer_module._orig_mod  # type: ignore (compiled torch module has this attribute, type checker doesn't know)
                else:
                    pairformer_module = self.model.pairformer_module

                s, z = pairformer_module(
                    s,
                    z,
                    mask=mask,
                    pair_mask=pair_mask,
                    use_kernels=self.model.use_kernels,
                )  # type: ignore (Object will be callable here)

        return {
            "s": s,
            "z": z,
            "s_inputs": s_inputs,
            "relative_position_encoding": relative_position_encoding,
            "feats": features,
        }

    def denoise_step(
        self,
        features: dict[str, Any],
        noisy_coords: Float[ArrayLike | Tensor, "..."],
        timestep: float | int,
        grad_needed: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """
        Perform one denoising step at given timestep/noise level.
        Returns predicted clean sample or predicted noise depending on
        model parameterization.

        Parameters
        ----------
        features : dict[str, Any]
            Model features produced by :meth:`featurize`.
        noisy_coords : Float[ArrayLike | Tensor, "..."]
            Noisy atom coordinates at current timestep.
        timestep : float | int
            Current timestep/noise level - for Boltz, this is the reverse timestep.
            This means that it starts from 0 and goes up to (sampling_steps - 1), so it
            is the "reverse" diffusion time.
        grad_needed : bool, optional
            Whether gradients are needed for this pass, by default False.
        **kwargs : dict, optional
            Additional keyword arguments for Boltz-1 denoising.

            - t_hat (float, optional)
                Precomputed t_hat value for the current timestep; if not provided, it
                will be computed internally.

            - eps (Tensor, optional)
                Precomputed noise tensor for the current timestep; if not provided, it
                will be sampled internally.

            - augmentation (bool, optional)
                Apply `center_random_augmentation` when True (default True).

            - align_to_input (bool, optional)
                Align denoised coordinates to `input_coords` when True (default True).

            - input_coords (Tensor, optional)
                Reference coordinates required if `align_to_input` is True.

            - alignment_weights (Tensor, optional)
                Atom weights for alignment; defaults to `atom_mask`.

            - multiplicity (int, optional)
                Overrides the multiplicity passed to the diffusion network, which is the
                replicating the model does internally along axis=0; defaults to the
                batch size of `noisy_coords`.

            - overwrite_representations (bool, optional)
                Whether to overwrite cached representations (default False).
                Do this if you are running a new input sample through the model.

            - recycling_steps (int, optional)
                Number of recycling steps to perform (default 3 from PredictArgs).
                This will only be applied if overwrite_representations is True, as
                it is passed to the pairformer module computation that is ideally
                cached for efficiency.

            - allow_alignment_gradients (bool, optional)
                Whether to allow gradients through the alignment step (default True).
                If False, detaches after alignment (matches original Boltz behavior).

        Returns
        -------
        dict[str, Any]
            Predicted clean sample or predicted noise.
        """
        if not self.cached_representations or kwargs.get(
            "overwrite_representations", False
        ):
            # Side effect: overwrites class attribute
            self.cached_representations = self.step(
                features,
                grad_needed=grad_needed,
                recycling_steps=kwargs.get(
                    "recycling_steps", self.predict_args.recycling_steps
                ),
            )

        features = self.cached_representations
        if not isinstance(noisy_coords, torch.Tensor):
            noisy_coords = torch.tensor(
                noisy_coords, device=self.device, dtype=torch.float32
            )

        s = features.get("s", None)
        z = features.get("z", None)
        s_inputs = features.get("s_inputs", None)
        relative_position_encoding = features.get("relative_position_encoding", None)
        # These are the input features to the conditioning networks
        feats = features.get("feats", None)

        if any(x is None for x in [s, z, s_inputs, relative_position_encoding, feats]):
            raise ValueError("Missing required features for denoise_step")

        feats = cast(dict[str, Any], feats)
        atom_mask = feats.get("atom_pad_mask")
        atom_mask = cast(Tensor, atom_mask)
        atom_mask = atom_mask.repeat_interleave(noisy_coords.shape[0], dim=0)

        with torch.set_grad_enabled(grad_needed):
            pad_len = atom_mask.shape[1] - noisy_coords.shape[1]
            if pad_len >= 0:
                padded_noisy_coords = pad_dim(noisy_coords, dim=1, pad_len=pad_len)
            else:
                raise ValueError("pad_len is negative, cannot pad noisy_coords")

            if "t_hat" in kwargs and "eps" in kwargs:
                t_hat = kwargs["t_hat"]
                eps = cast(Tensor, kwargs["eps"])
                if pad_len > 0 and eps.shape[1] != padded_noisy_coords.shape[1]:
                    eps = pad_dim(eps, dim=1, pad_len=pad_len)
            else:
                timestep_scaling = self.get_timestep_scaling(timestep)
                eps = timestep_scaling["eps_scale"] * torch.randn(
                    padded_noisy_coords.shape, device=self.device
                )
                t_hat = timestep_scaling["t_hat"]

            if kwargs.get("augmentation", True):
                padded_noisy_coords = center_random_augmentation(
                    padded_noisy_coords,
                    atom_mask=atom_mask,
                    augmentation=True,
                )

            padded_noisy_coords_eps = cast(Tensor, padded_noisy_coords) + eps

            padded_atom_coords_denoised, _ = (
                self.model.structure_module.preconditioned_network_forward(
                    padded_noisy_coords_eps,
                    t_hat,
                    training=False,
                    network_condition_kwargs=dict(
                        s_trunk=s,
                        z_trunk=z,
                        s_inputs=s_inputs,
                        feats=feats,
                        relative_position_encoding=relative_position_encoding,
                        multiplicity=kwargs.get(
                            "multiplicity",
                            cast(Tensor, padded_noisy_coords_eps).shape[0],
                        ),
                    ),
                )
            )

            if kwargs.get("align_to_input", True):
                input_coords = kwargs.get("input_coords")
                if input_coords is not None:
                    if (
                        pad_len > 0
                        and input_coords.shape[1]
                        != padded_atom_coords_denoised.shape[1]
                    ):
                        input_coords = pad_dim(input_coords, dim=1, pad_len=pad_len)

                    alignment_weights = kwargs.get("alignment_weights", atom_mask)
                    allow_alignment_gradients = kwargs.get(
                        "allow_alignment_gradients", False
                    )

                    if allow_alignment_gradients:
                        padded_atom_coords_denoised = (
                            weighted_rigid_align_differentiable(
                                padded_atom_coords_denoised.float(),
                                cast(Tensor, input_coords).float(),
                                weights=alignment_weights,
                                mask=atom_mask,
                                allow_gradients=True,
                            )
                        )
                    else:
                        padded_atom_coords_denoised = (
                            weighted_rigid_align_differentiable(
                                padded_atom_coords_denoised.float(),
                                cast(Tensor, input_coords).float(),
                                weights=alignment_weights,
                                mask=atom_mask,
                            )
                        )
                else:
                    raise ValueError(
                        "Input coordinates must be provided when align_to_input "
                        "is True."
                    )

            atom_coords_denoised = padded_atom_coords_denoised[
                atom_mask.bool(), :
            ].reshape(noisy_coords.shape[0], -1, 3)

        return {"atom_coords_denoised": atom_coords_denoised}

    def initialize_from_noise(
        self, structure: dict, noise_level: float | int, **kwargs
    ) -> Float[ArrayLike | Tensor, "*batch _num_atoms 3"]:
        """Create a noisy version of structure's coordinates at given noise level.

        Parameters
        ----------
        structure : dict
            Atomworks structure dictionary. [See Atomworks documentation](https://baker-laboratory.github.io/atomworks-dev/latest/io/parser.html#atomworks.io.parser.parse)
        noise_level : float | int
            Timestep/noise level - for Boltz, this is the reverse timestep.
            This means that it starts from 0 and goes up to (sampling_steps - 1), so it
            is the "reverse" diffusion time.
        **kwargs : dict, optional
            Additional keyword arguments needed for classes that implement this Protocol

        Returns
        -------
        Float[ArrayLike | Tensor, "*batch _num_atoms 3"]
            Noisy structure coordinates for atoms that have nonzero occupancy.
        """
        if "asym_unit" not in structure:
            raise ValueError(
                "structure must contain 'asym_unit' key to access"
                "the coordinates in the asymmetric unit."
            )

        if noise_level < 0 or noise_level >= self.predict_args.sampling_steps:
            raise ValueError(
                f"noise_level {noise_level} is out of bounds for sampling steps "
                f"{self.predict_args.sampling_steps}"
            )

        # We need to make sure the missing atoms are handled correctly
        residues_with_occupancy = structure["asym_unit"].occupancy > 0
        coords = structure["asym_unit"].coord[:, residues_with_occupancy]

        if isinstance(coords, ArrayLike):
            coords = torch.tensor(coords, device=self.device, dtype=torch.float32)

        if noise_level == 0:
            noisy_coords = self.noise_schedule["sigma_tm"][0] * torch.randn_like(coords)
        else:
            noisy_coords = coords + self.noise_schedule["sigma_tm"][
                int(noise_level)
            ] * torch.randn_like(coords)

        return noisy_coords
