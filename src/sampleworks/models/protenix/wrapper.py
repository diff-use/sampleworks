from collections.abc import Mapping
from logging import getLogger, Logger
from pathlib import Path
from typing import Any, cast

import torch
from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from einx import rearrange
from jaxtyping import Array, ArrayLike, Float
from ml_collections import ConfigDict
from protenix.config import parse_configs
from protenix.data.data_pipeline import DataPipeline
from protenix.data.json_to_feature import SampleDictToFeatures
from protenix.data.msa_featurizer import InferenceMSAFeaturizer
from protenix.data.utils import data_type_transform, make_dummy_feature
from protenix.model.protenix import (
    Protenix,
    update_input_feature_dict,
)
from protenix.utils.torch_utils import autocasting_disable_decorator, dict_to_tensor
from runner.inference import download_infercence_cache as download_inference_cache
from runner.msa_search import update_infer_json
from torch import Tensor

from sampleworks.utils.torch_utils import send_tensors_in_dict_to_device

from .structure_processing import (
    add_terminal_oxt_atoms,
    create_protenix_input_from_structure,
    ensure_atom_array,
    filter_zero_occupancy,
    reconcile_atom_arrays,
)


class ProtenixWrapper:
    """
    Wrapper for Protenix (ByteDance AlphaFold3 implementation)
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        args_str: str = "",
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        model: Protenix | None = None
    ):
        """
        Parameters
        ----------
        checkpoint_path: str | Path
            Filesystem path to the Protenix checkpoint containing trained weights.
        args_str: str, optional
            Command-line style argument string to override default configurations.
        device: torch.device, optional
            Device to run the model on, by default CUDA if available.
        """
        logger: Logger = getLogger(__name__)
        self.checkpoint_path = checkpoint_path
        self.device = torch.device(device)

        self.cache_path = (
            (Path(checkpoint_path) if isinstance(checkpoint_path, str) else checkpoint_path)
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
        self.configs.model_name = str(self.checkpoint_path).split("/")[-1].replace(".pt", "")
        self.configs.load_checkpoint_dir = str(self.cache_path)
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

        if model is None:
            self.model = Protenix(self.configs).to(self.device)
        else:
            self.model = model.to(self.device)

        checkpoint_path_str = f"{self.configs.load_checkpoint_dir}/{self.configs.model_name}.pt"
        logger.info(f"Loading checkpoint from {checkpoint_path_str}")
        checkpoint = torch.load(checkpoint_path_str, map_location=self.device)

        if any(k.startswith("module.") for k in checkpoint["model"].keys()):
            checkpoint["model"] = {k[len("module.") :]: v for k, v in checkpoint["model"].items()}

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

        sigmas = self._compute_noise_schedule(cast(dict, self.configs.sample_diffusion)["N_step"])
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
        num_steps: int
            Number of diffusion sampling steps.

        Returns
        -------
        Tensor
            Noise schedule with shape (num_steps + 1,).
        """
        return self.model.inference_noise_scheduler(N_step=num_steps, device=self.device)

    def featurize(self, structure: dict, **kwargs) -> dict[str, Any]:
        """From an Atomworks structure, calculate Protenix input features.

        Parameters
        ----------
        structure: dict
            Atomworks structure dictionary.
        **kwargs: dict, optional
            Additional arguments for feature generation.
            - out_dir: Directory for saving intermediate JSON file
            - use_msa: Whether to generate MSA features (default True)

        Returns
        -------
        dict[str, Any]
            Protenix input features.
        """

        # If featurize is called again, we should clear cached representations
        # to avoid using stale data
        self.cached_representations.clear()

        out_dir = kwargs.get("out_dir", structure.get("metadata", {}).get("id", "protenix_output"))

        json_path, json_dict = create_protenix_input_from_structure(structure, out_dir)

        use_msa = kwargs.get("use_msa", True)
        if use_msa:
            import json

            updated_json_path = json_path.with_name(f"{json_path.stem}-add-msa.json")
            if not updated_json_path.exists():
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

        entity_to_asym_id = DataPipeline.get_label_entity_id_to_asym_id_int(atom_array_protenix)
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

        feat = cast(dict[str, Any], data_type_transform(feat_or_label_dict=features_dict))

        if "constraint_feature" in feat and isinstance(feat["constraint_feature"], dict):
            for k, v in feat["constraint_feature"].items():
                feat[f"constraint_feature_{k}"] = v
            del feat["constraint_feature"]

        input_feature_dict = dict_to_tensor(feat)

        atom_array = ensure_atom_array(structure["asym_unit"])
        atom_array = filter_zero_occupancy(atom_array)
        atom_array = add_terminal_oxt_atoms(
            atom_array=atom_array, chain_info=structure.get("chain_info", {})
        )
        atom_array = reconcile_atom_arrays(atom_array, atom_array_protenix)

        if "asym_unit" in structure:
            n_atoms_protenix = len(atom_array_protenix)
            n_atoms_atomworks = len(atom_array)
            assert n_atoms_protenix == n_atoms_atomworks, (
                f"Atom count mismatch after reconciliation: Protenix has "
                f"{n_atoms_protenix} atoms, Atomworks has {n_atoms_atomworks} atoms."
            )

        features = cast(dict[str, Any], input_feature_dict)

        if "asym_unit" in structure:
            true_coords = cast(Array, atom_array.coord)
            if not isinstance(true_coords, torch.Tensor):
                true_coords = torch.tensor(true_coords, device=self.device, dtype=torch.float32)
            features["true_coords"] = true_coords
            features["true_atom_array"] = atom_array

        features = self.model.relative_position_encoding.generate_relp(features)
        features = update_input_feature_dict(features)

        features = send_tensors_in_dict_to_device(features, self.device, inplace=False)

        return features

    def step(self, features: dict[str, Any], grad_needed: bool = False, **kwargs) -> dict[str, Any]:
        """Perform a pass through the Protenix Pairformer to obtain
        representations.

        Parameters
        ----------
        features: dict[str, Any]
            Model features as returned by featurize.
        grad_needed: bool, optional
            Whether gradients are needed for this pass, by default False.
        **kwargs: dict, optional
            Additional arguments.

            - recycling_steps: int
                Number of recycling steps to perform. Defaults to the value in
                the Protenix model config.

            - enable_diffusion_shared_vars_cache: bool, optional
                Enable caching of shared variables in diffusion module
                (default True).


        Returns
        -------
        dict[str, Any]
            Protenix model outputs including trunk representations
            (s_inputs, s_trunk, z_trunk).
        """
        inplace_safe = not grad_needed
        chunk_size = (
            cast(ConfigDict, self.configs.infer_setting).chunk_size if inplace_safe else None
        )
        with torch.set_grad_enabled(grad_needed):
            s_inputs, s, z = self.model.get_pairformer_output(
                input_feature_dict=features,
                N_cycle=kwargs.get("recycling_steps", cast(ConfigDict, self.configs.model).N_cycle),
                inplace_safe=inplace_safe,  # Default in Protenix is True
                chunk_size=cast(int, chunk_size),  # Default in Protenix is 4
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

        if kwargs.get(
            "enable_diffusion_shared_vars_cache",
            self.configs.enable_diffusion_shared_vars_cache,
        ):
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
        noisy_coords: Float[ArrayLike | Tensor, "..."],
        timestep: float | int,
        grad_needed: bool = False,
        **kwargs,
    ) -> dict[str, Any]:
        """Perform denoising at given timestep for Protenix model.

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
            Additional keyword arguments for Protenix denoising.
            - t_hat: float, optional
                Precomputed t_hat value; computed internally if not provided.
            - eps: Tensor, optional
                Precomputed noise tensor; sampled internally if not provided.
            - overwrite_representations: bool, optional
                Whether to recompute cached representations (default False).
            - recycling_steps: int
                Number of recycling steps to perform. Defaults to the value in
                the Protenix model config.
            - enable_diffusion_shared_vars_cache: bool, optional
                Enable caching of shared variables in diffusion module
                (default True).

        Returns
        -------
        dict[str, Tensor]
            Dictionary containing atom_coords_denoised with cleaned coordinates.
        """
        if not self.cached_representations or kwargs.get("overwrite_representations", False):
            step_kwargs = {
                "recycling_steps": kwargs.get(
                    "recycling_steps",
                    cast(ConfigDict, self.configs.model).N_cycle,
                ),
                "enable_diffusion_shared_vars_cache": kwargs.get(
                    "enable_diffusion_shared_vars_cache",
                    self.configs.enable_diffusion_shared_vars_cache,
                ),
            }
            self.cached_representations = self.step(features, grad_needed=False, **step_kwargs)

        outputs = self.cached_representations
        if grad_needed:
            outputs = {
                k: (
                    v.detach()
                    if isinstance(v, torch.Tensor)
                    else (
                        tuple(x.detach() if isinstance(x, torch.Tensor) else x for x in v)
                        if isinstance(v, tuple)
                        else v
                    )
                )
                for k, v in outputs.items()
            }
        if not isinstance(noisy_coords, torch.Tensor):
            noisy_coords = torch.tensor(noisy_coords, device=self.device, dtype=torch.float32)

        with torch.set_grad_enabled(grad_needed):
            if "t_hat" in kwargs and "eps" in kwargs:
                t_hat = kwargs["t_hat"]
                eps = cast(Tensor, kwargs["eps"])
            else:
                timestep_scaling = self.get_timestep_scaling(timestep)
                eps = timestep_scaling["eps_scale"] * torch.randn_like(noisy_coords)
                t_hat = timestep_scaling["t_hat"]

            noisy_coords_eps = noisy_coords + eps

            t_hat_tensor = torch.tensor(
                [t_hat] * noisy_coords_eps.shape[0],
                device=noisy_coords.device,
                dtype=noisy_coords.dtype,
            )

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

            # TODO: is there a way to handle this more cleanly?
            # remove protenix ensemble dim, shape (N_ensemble, 1, N_atoms, 3)
            # -> (N_ensemble, N_atoms, 3)
            atom_coords_denoised = atom_coords_denoised.squeeze(1)

        return {"atom_coords_denoised": atom_coords_denoised}

    def get_noise_schedule(self) -> Mapping[str, Float[ArrayLike | Tensor, "..."]]:
        """Return the full noise schedule with semantic keys.

        Returns
        -------
        Mapping[str, Float[ArrayLike | Tensor, "..."]]
            Noise schedule arrays with keys sigma_tm, sigma_t, and gamma.
        """
        return self.noise_schedule

    def get_timestep_scaling(self, timestep: float | int) -> dict[str, float]:
        """Return scaling constants for Protenix diffusion at given timestep.

        Parameters
        ----------
        timestep: float | int
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
        eps_scale = cast(dict, self.configs.sample_diffusion)["noise_scale_lambda"] * torch.sqrt(
            t_hat**2 - sigma_tm**2
        )

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
            raise ValueError("structure must contain asym_unit key to access coordinates.")

        atom_array = ensure_atom_array(structure["asym_unit"])
        atom_array = filter_zero_occupancy(atom_array)
        atom_array = add_terminal_oxt_atoms(
            atom_array=atom_array,
            chain_info=structure.get("chain_info", {}),
        )
        coords = cast(Array, atom_array.coord)

        if isinstance(coords, ArrayLike):
            coords = torch.tensor(coords, device=self.device, dtype=torch.float32)

        if coords.ndim == 2:  # single structure
            coords = cast(Tensor, rearrange("n c -> e n c", coords, e=ensemble_size))
        elif coords.ndim == 3 and coords.shape[0] == 1:  # single structure
            coords = cast(Tensor, rearrange("() n c -> e n c", coords, e=ensemble_size))
        elif coords.ndim == 3 and coords.shape[0] != ensemble_size:
            coords = cast(
                Tensor,
                rearrange(
                    "n c -> e n c", coords[0], e=ensemble_size
                ),  # grab only the first structure
            )

        # validate coords shape
        if coords.ndim != 3 or coords.shape[0] != ensemble_size or coords.shape[2] != 3:
            raise ValueError(
                f"coords shape should be (ensemble_size, N_atoms, 3), but is {coords.shape}"
            )

        sigma = self.noise_schedule["sigma_tm"][int(noise_level)]

        if noise_level == 0:
            noisy_coords = sigma * torch.randn_like(coords)
        else:
            noise = torch.randn_like(coords)
            noisy_coords = coords + sigma * noise

        return noisy_coords
