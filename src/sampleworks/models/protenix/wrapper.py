from collections.abc import Mapping
from logging import getLogger, Logger
from pathlib import Path
from typing import Any, cast

import torch
from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from einops import einsum
from jaxtyping import ArrayLike, Float
from ml_collections import ConfigDict
from protenix.config import parse_configs
from protenix.data.data_pipeline import DataPipeline
from protenix.data.json_to_feature import SampleDictToFeatures
from protenix.data.msa_featurizer import InferenceMSAFeaturizer
from protenix.data.utils import data_type_transform, make_dummy_feature
from protenix.model.protenix import InferenceNoiseScheduler, Protenix
from protenix.model.utils import centre_random_augmentation
from protenix.utils.torch_utils import autocasting_disable_decorator, dict_to_tensor
from runner.inference import download_infercence_cache as download_inference_cache
from runner.msa_search import update_infer_json
from torch import Tensor

from .structure_processing import create_protenix_input_from_structure


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

    true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )

    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    if num_points < (dim + 1):
        print(
            "Warning: The size of one of the point clouds is <= dim+1. "
            + "`WeightedRigidAlign` cannot return a unique rotation."
        )

    cov_matrix = einsum(
        weights * pred_coords_centered, true_coords_centered, "b n i, b n j -> b i j"
    )

    original_dtype = cov_matrix.dtype
    cov_matrix_32 = cov_matrix.to(dtype=torch.float32)

    U, _, Vh = torch.linalg.svd(cov_matrix_32)

    rotation = torch.matmul(U, Vh)

    det = torch.det(rotation)
    diag = torch.ones(batch_size, dim, device=rotation.device, dtype=torch.float32)
    diag[:, -1] = det

    rotation = torch.matmul(U * diag.unsqueeze(1), Vh)

    rotation = rotation.to(dtype=original_dtype)

    aligned_coords = (
        einsum(true_coords_centered, rotation, "b n i, b i j -> b n j") + pred_centroid
    )

    if not allow_gradients:
        aligned_coords = aligned_coords.detach()

    return aligned_coords


class ProtenixWrapper:
    """
    Wrapper for Protenix (ByteDance AlphaFold3 implementation)
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        args_str: str = "",
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """
        Parameters
        ----------
        checkpoint_path : str | Path
            Filesystem path to the Protenix checkpoint containing trained weights.
        args_str : str, optional
            Command-line style argument string to override default configurations.
        device : torch.device, optional
            Device to run the model on, by default CUDA if available.
        """
        logger: Logger = getLogger(__name__)
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)

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
        self.cached_representations: dict[str, Any] = {}

        args = args_str.split()
        verified_arg_str = ""
        for k, v in zip(args[::2], args[1::2]):
            assert k.startswith("--")
            verified_arg_str += f"{k} {v} "

        configs = {**configs_base, **{"data": data_configs}, **inference_configs}
        self.configs: ConfigDict = parse_configs(
            configs=configs,
            arg_str=verified_arg_str,
            fill_required_with_null=True,
        )

        # Protenix inference logging
        model_name = self.configs.model_name
        _, model_size, model_feature, model_version = cast(str, model_name).split("_")
        logger.info(
            f"Inference by Protenix: model_size: {model_size}, with_feature: "
            f"{model_feature.replace('-', ', ')}, model_version: {model_version}"
        )
        model_specifics_configs = ConfigDict(model_configs[cast(str, model_name)])
        # update model specific configs
        self.configs.update(model_specifics_configs)
        logger.info(
            f"Triangle_multiplicative kernel: {self.configs.triangle_multiplicative}, "
            f"Triangle_attention kernel: {self.configs.triangle_attention}"
        )
        logger.info(
            f"enable_diffusion_shared_vars_cache: "
            f"{self.configs.enable_diffusion_shared_vars_cache}, "
            f"enable_efficient_fusion: {self.configs.enable_efficient_fusion}, "
            f"enable_tf32: {self.configs.enable_tf32}"
        )
        download_inference_cache(self.configs)

        if torch.cuda.is_available():
            torch.cuda.set_device(self.device)

        self.model = Protenix(self.configs).to(self.device)

        checkpoint_path_str = (
            f"{self.configs.load_checkpoint_dir}/{self.configs.model_name}.pt"
        )
        logger.info(f"Loading checkpoint from {checkpoint_path_str}")
        checkpoint = torch.load(checkpoint_path_str, map_location=self.device)

        if any(k.startswith("module.") for k in checkpoint["model"].keys()):
            checkpoint["model"] = {
                k[len("module.") :]: v for k, v in checkpoint["model"].items()
            }

        self.model.load_state_dict(
            state_dict=checkpoint["model"],
            strict=cast(dict, self.configs).get("load_strict", True),
        )
        self.model.eval()
        logger.info("Finished loading checkpoint")

        if cast(dict, self.configs.get("esm", {})).get("enable", False):
            from protenix.data.esm_featurizer import ESMFeaturizer

            esm_config = self.configs.esm
            self.esm_featurizer = ESMFeaturizer(
                embedding_dir=esm_config.embedding_dir,  # type: ignore (their ConfigDict lacks typing)
                sequence_fpath=esm_config.sequence_fpath,  # type: ignore (their ConfigDict lacks typing)
                embedding_dim=esm_config.embedding_dim,  # type: ignore (their ConfigDict lacks typing)
                error_dir="./esm_embeddings/",
            )
        else:
            self.esm_featurizer = None

        sigmas = self._compute_noise_schedule(
            cast(dict, self.configs.sample_diffusion)["N_step"]
        )
        gammas = torch.where(
            sigmas > cast(dict, self.configs.sample_diffusion)["gamma_min"],
            cast(dict, self.configs.sample_diffusion)["gamma0"],
            0.0,
        )
        self.noise_schedule: dict[str, Float[Tensor, ...]] = {
            "sigma_tm": sigmas[:-1],
            "sigma_t": sigmas[1:],
            "gamma": gammas[1:],
        }

    def _compute_noise_schedule(self, num_steps: int) -> Tensor:
        """Compute the noise schedule for diffusion sampling.

        Parameters
        ----------
        num_steps : int
            Number of diffusion sampling steps.

        Returns
        -------
        Tensor
            Noise schedule with shape (num_steps + 1,).
        """
        scheduler = InferenceNoiseScheduler(
            s_max=cast(dict, self.configs.inference_noise_scheduler)["s_max"],
            s_min=cast(dict, self.configs.inference_noise_scheduler)["s_min"],
            rho=cast(dict, self.configs.inference_noise_scheduler)["rho"],
            sigma_data=cast(dict, self.configs.inference_noise_scheduler)["sigma_data"],
        )
        return scheduler(N_step=num_steps, device=self.device)

    def featurize(self, structure: dict, **kwargs) -> dict[str, Any]:
        """From an Atomworks structure, calculate Protenix input features.

        Parameters
        ----------
        structure : dict
            Atomworks structure dictionary.
        **kwargs : dict, optional
            Additional arguments for feature generation.
            - out_dir: Directory for saving intermediate JSON file
            - use_msa: Whether to generate MSA features (default True)

        Returns
        -------
        dict[str, Any]
            Protenix input features.
        """
        out_dir = kwargs.get(
            "out_dir", structure.get("metadata", {}).get("id", "protenix_output")
        )

        json_path, json_dict = create_protenix_input_from_structure(structure, out_dir)

        use_msa = kwargs.get("use_msa", True)
        if use_msa:
            import json

            updated_json_path = update_infer_json(
                json_file=str(json_path),
                out_dir=str(out_dir),
                use_msa=True,
            )
            with open(updated_json_path) as f:
                json_data = json.load(f)
                json_dict = json_data[0]

        sample2feat = SampleDictToFeatures(json_dict)
        features_dict, atom_array_protenix, token_array = sample2feat.get_feature_dict()
        features_dict["distogram_rep_atom_mask"] = torch.Tensor(
            atom_array_protenix.distogram_rep_atom_mask
        ).long()

        entity_to_asym_id = DataPipeline.get_label_entity_id_to_asym_id_int(
            atom_array_protenix
        )
        msa_features = (
            InferenceMSAFeaturizer.make_msa_feature(  # type: ignore (they forgot @staticmethod)
                bioassembly=json_dict["sequences"],
                entity_to_asym_id=entity_to_asym_id,
                token_array=token_array,
                atom_array=atom_array_protenix,
            )
            if use_msa
            else {}
        )

        if self.esm_featurizer is not None:
            x_esm = self.esm_featurizer(
                token_array=token_array,
                atom_array=atom_array_protenix,
                bioassembly_dict=json_dict,
                inference_mode=True,
            )
            features_dict["esm_token_embedding"] = x_esm

        dummy_feats = ["template"]
        if len(msa_features) == 0:
            dummy_feats.append("msa")
        else:
            msa_features = dict_to_tensor(msa_features)
            features_dict.update(msa_features)
        features_dict = make_dummy_feature(
            features_dict=features_dict,
            dummy_feats=dummy_feats,
        )

        feat = cast(
            dict[str, Any], data_type_transform(feat_or_label_dict=features_dict)
        )

        if "constraint_feature" in feat and isinstance(
            feat["constraint_feature"], dict
        ):
            for k, v in feat["constraint_feature"].items():
                feat[f"constraint_feature_{k}"] = v
            del feat["constraint_feature"]

        input_feature_dict = dict_to_tensor(feat)
        for k, v in input_feature_dict.items():
            if k != "sample_name":
                input_feature_dict[k] = v.unsqueeze(0)

        from .structure_processing import ensure_atom_array

        atom_array = ensure_atom_array(structure["asym_unit"])
        residues_with_occupancy = atom_array.occupancy > 0

        if "asym_unit" in structure:
            n_atoms_protenix = len(atom_array_protenix)
            n_atoms_atomworks = len(atom_array[residues_with_occupancy])
            atom_diff = abs(n_atoms_protenix - n_atoms_atomworks)
            assert atom_diff <= 1, (
                f"Coordinate count mismatch: Protenix processed "
                f"{n_atoms_protenix} atoms, Atomworks has {n_atoms_atomworks} "
                f"atoms (difference: {atom_diff}). Expected difference <= 1 "
                f"(for terminal OXT)"
            )

        features = cast(dict[str, Any], input_feature_dict)

        if "asym_unit" in structure:
            true_coords = atom_array.coord[residues_with_occupancy]
            if not isinstance(true_coords, torch.Tensor):
                true_coords = torch.tensor(
                    true_coords, device=self.device, dtype=torch.float32
                )
            features["true_coords"] = true_coords

        return features

    def step(
        self, features: dict[str, Any], grad_needed: bool = False, **kwargs
    ) -> dict[str, Any]:
        """Perform a pass through the Protenix Pairformer to obtain
        representations.

        Parameters
        ----------
        features : dict[str, Any]
            Model features as returned by featurize.
        grad_needed : bool, optional
            Whether gradients are needed for this pass, by default False.
        **kwargs : dict, optional
            Additional arguments.

            - recycling_steps: int
                Number of recycling steps to perform. Defaults to the value in
                predict_args.


        Returns
        -------
        dict[str, Any]
            Protenix model outputs including trunk representations
            (s_inputs, s_trunk, z_trunk).
        """
        with torch.set_grad_enabled(grad_needed):
            s_inputs, s, z = self.model.get_pairformer_output(
                input_feature_dict=features,
                N_cycle=kwargs.get(
                    "recycling_steps", cast(ConfigDict, self.configs.model).N_cycle
                ),
                # inplace_safe=inplace_safe, # Default in Protenix is True
                # chunk_size=chunk_size, # Default in Protenix is 4
            )

        keys_to_delete = []
        for key in features.keys():
            if "template_" in key or key in [
                "msa",
                "has_deletion",
                "deletion_value",
                "profile",
                "deletion_mean",
                "token_bonds",
            ]:
                keys_to_delete.append(key)

        for key in keys_to_delete:
            del features[key]
        torch.cuda.empty_cache()

        outputs: dict[str, Any] = {
            "s_inputs": s_inputs,
            "s_trunk": s,
            "z_trunk": z,
            "features": features,
        }

        if kwargs.get("enable_diffusion_shared_vars_cache", True):
            outputs["pair_z"] = autocasting_disable_decorator(
                self.model.configs.skip_amp.sample_diffusion  # type: ignore (their ConfigDict lacks typing)
            )(
                self.model.diffusion_module.diffusion_conditioning.prepare_cache
            )(features["relp"], z, False)
            outputs["p_lm/c_l"] = autocasting_disable_decorator(
                self.model.configs.skip_amp.sample_diffusion  # type: ignore (their ConfigDict lacks typing)
            )(self.model.diffusion_module.atom_attention_encoder.prepare_cache)(
                features["ref_pos"],
                features["ref_charge"],
                features["ref_mask"],
                features["ref_element"],
                features["ref_atom_name_chars"],
                features["atom_to_token_idx"],
                features["d_lm"],
                features["v_lm"],
                features["pad_info"],
                "",
                outputs["pair_z"],
                False,
            )
        else:
            outputs["pair_z"] = None
            outputs["p_lm/c_l"] = [None, None]

        return outputs

    def denoise_step(
        self,
        features: dict[str, Any],
        noisy_coords: Float[Tensor, "..."],
        timestep: float,
        grad_needed: bool = False,
        **kwargs,
    ) -> dict[str, Tensor]:
        """Perform denoising at given timestep for Protenix model.

        Parameters
        ----------
        features : dict[str, Any]
            Model features produced by featurize or step.
        noisy_coords : Float[Tensor, "..."]
            Noisy atom coordinates at the current timestep.
        timestep : float
            Current timestep or noise level in reverse time (starts from 0).
        grad_needed : bool, optional
            Whether gradients are needed for this pass, by default False.
        **kwargs : dict, optional
            Additional keyword arguments for Protenix denoising.
            - t_hat: float, optional
                Precomputed t_hat value; computed internally if not provided.
            - delta_noise_level: Tensor, optional
                Precomputed noise tensor; sampled internally if not provided.
            - augmentation: bool, optional
                Apply coordinate augmentation when True (default True).
            - align_to_input: bool, optional
                Align denoised coordinates to input_coords when True (default True).
            - input_coords: Tensor, optional
                Reference coordinates for alignment.
            - alignment_weights: Tensor, optional
                Weights for alignment operation.
            - overwrite_representations: bool, optional
                Whether to recompute cached representations (default False).
            - allow_alignment_gradients: bool, optional
                Preserve gradients through alignment (default False).
            - enable_diffusion_shared_vars_cache: bool, optional
                Enable caching of shared variables in diffusion module
                (default True).

        Returns
        -------
        dict[str, Tensor]
            Dictionary containing atom_coords_denoised with cleaned coordinates.
        """
        if not self.cached_representations or kwargs.get(
            "overwrite_representations", False
        ):
            self.cached_representations = self.step(features, grad_needed=grad_needed)

        outputs = self.cached_representations

        with torch.set_grad_enabled(grad_needed):
            if "t_hat" in kwargs and "eps" in kwargs:
                t_hat = kwargs["t_hat"]
                delta_noise_level = cast(Tensor, kwargs["delta_noise_level"])
            else:
                timestep_scaling = self.get_timestep_scaling(timestep)
                delta_noise_level = timestep_scaling[
                    "delta_noise_level"
                ] * torch.randn_like(noisy_coords)
                t_hat = timestep_scaling["t_hat"]

            if kwargs.get("augmentation", True):
                noisy_coords = (
                    centre_random_augmentation(x_input_coords=noisy_coords, N_sample=1)
                    .squeeze(dim=-3)
                    .to(noisy_coords.dtype)
                )

            noisy_coords_eps = noisy_coords + delta_noise_level

            t_hat_tensor = torch.tensor(
                [t_hat], device=noisy_coords.device, dtype=noisy_coords.dtype
            )
            if noisy_coords_eps.dim() == 2:
                noisy_coords_eps = noisy_coords_eps.unsqueeze(0)

            atom_coords_denoised = self.model.diffusion_module.forward(
                x_noisy=noisy_coords_eps,
                t_hat_noise_level=t_hat_tensor,
                input_feature_dict=features,
                s_inputs=cast(Tensor, outputs.get("s_inputs")),
                s_trunk=cast(Tensor, outputs.get("s_trunk")),
                z_trunk=cast(Tensor, outputs.get("z_trunk")),
                pair_z=cast(Tensor, outputs["pair_z"]),
                p_lm=cast(Tensor, outputs["p_lm/c_l"][0]),
                c_l=cast(Tensor, outputs["p_lm/c_l"][1]),
            )

            if atom_coords_denoised.dim() == 3 and noisy_coords.dim() == 2:
                atom_coords_denoised = atom_coords_denoised.squeeze(0)

            if kwargs.get("align_to_input", True):
                input_coords = kwargs.get("input_coords")
                if input_coords is not None:
                    alignment_weights = kwargs.get(
                        "alignment_weights",
                        torch.ones_like(noisy_coords[..., 0]),
                    )
                    allow_alignment_gradients = kwargs.get(
                        "allow_alignment_gradients", False
                    )

                    if atom_coords_denoised.dim() == 2:
                        atom_coords_denoised = atom_coords_denoised.unsqueeze(0)
                        input_coords_batch = cast(Tensor, input_coords).unsqueeze(0)
                        alignment_weights_batch = alignment_weights.unsqueeze(0)
                    else:
                        input_coords_batch = cast(Tensor, input_coords)
                        alignment_weights_batch = alignment_weights

                    atom_coords_denoised = weighted_rigid_align_differentiable(
                        atom_coords_denoised.float(),
                        input_coords_batch.float(),
                        weights=alignment_weights_batch,
                        mask=torch.ones_like(alignment_weights_batch),
                        allow_gradients=allow_alignment_gradients,
                    )

                    if noisy_coords.dim() == 2:
                        atom_coords_denoised = atom_coords_denoised.squeeze(0)
                else:
                    raise ValueError(
                        "Input coordinates must be provided when align_to_input is "
                        "True."
                    )

        return {"atom_coords_denoised": atom_coords_denoised}

    def get_noise_schedule(self) -> Mapping[str, Float[ArrayLike | Tensor, "..."]]:
        """Return the full noise schedule with semantic keys.

        Returns
        -------
        Mapping[str, Float[ArrayLike | Tensor, "..."]]
            Noise schedule arrays with keys sigma_tm, sigma_t, and gamma.
        """
        return self.noise_schedule

    def get_timestep_scaling(self, timestep: float) -> dict[str, float]:
        """Return scaling constants for Protenix diffusion at given timestep.

        Parameters
        ----------
        timestep : float
            Current timestep or noise level starting from 0.

        Returns
        -------
        dict[str, float]
            Dictionary containing t_hat, sigma_t, and eps_scale.
        """
        sigma_tm = self.noise_schedule["sigma_tm"][int(timestep)]
        sigma_t = self.noise_schedule["sigma_t"][int(timestep)]
        gamma = self.noise_schedule["gamma"][int(timestep)]

        t_hat = sigma_tm * (1 + gamma)
        eps_scale = cast(dict, self.configs.sample_diffusion)[
            "noise_scale_lambda"
        ] * torch.sqrt(t_hat**2 - sigma_tm**2)

        return {
            "t_hat": t_hat.item(),
            "sigma_t": sigma_t.item(),
            "eps_scale": eps_scale.item(),
        }

    def initialize_from_noise(
        self, structure: dict, noise_level: float, **kwargs
    ) -> Float[Tensor, "*batch _num_atoms 3"]:
        """Create a noisy version of structure coordinates at given noise level.

        Parameters
        ----------
        structure : dict
            Atomworks structure dictionary.
        noise_level : float
            Timestep or noise level in reverse time starting from 0.
        **kwargs : dict, optional
            Additional keyword arguments for initialization.

        Returns
        -------
        Float[Tensor, "*batch _num_atoms 3"]
            Noisy structure coordinates for atoms with nonzero occupancy.
        """
        if "asym_unit" not in structure:
            raise ValueError(
                "structure must contain asym_unit key to access coordinates."
            )

        residues_with_occupancy = structure["asym_unit"].occupancy > 0
        coords = structure["asym_unit"].coord[:, residues_with_occupancy]

        if isinstance(coords, ArrayLike):
            coords = torch.tensor(coords, device=self.device, dtype=torch.float32)

        coords = coords - coords.mean(dim=-2, keepdim=True)

        sigma = self.noise_schedule["sigma_tm"][int(noise_level)]
        noise = torch.randn(coords.shape, device=self.device, dtype=coords.dtype)
        noisy_coords = coords + sigma * noise

        return noisy_coords
