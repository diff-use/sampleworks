from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import torch
from einops import einsum
from jaxtyping import ArrayLike, Float
from protenix.model.protenix import Protenix
from torch import Tensor


def weighted_rigid_align_differentiable(
    true_coords,
    pred_coords,
    weights,
    mask,
    allow_gradients: bool = True,
):
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


@dataclass
class ProteinixPredictArgs:
    """Arguments for Proteinix model prediction."""

    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 1
    num_ensemble: int = 1


@dataclass
class ProteinixDiffusionParams:
    """Diffusion process parameters for Proteinix."""

    sigma_min: float = 0.0004
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    rho: float = 7.0
    P_mean: float = -1.2
    P_std: float = 1.5
    gamma_0: float = 0.8
    gamma_min: float = 1.0
    noise_scale: float = 1.003
    step_scale: float = 1.5


class ProteinixWrapper:
    """
    Wrapper for Protenix (ByteDance AlphaFold3 implementation)
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        predict_args: ProteinixPredictArgs = ProteinixPredictArgs(),
        diffusion_args: ProteinixDiffusionParams = ProteinixDiffusionParams(),
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        """
        Parameters
        ----------
        checkpoint_path : str | Path
            Filesystem path to the Protenix checkpoint containing trained weights.
        predict_args : ProteinixPredictArgs, optional
            Runtime prediction configuration such as recycling depth and sampling steps.
        diffusion_args : ProteinixDiffusionParams, optional
            Diffusion process parameters controlling the noise schedule and sampling.
        device : torch.device, optional
            Device to run the model on, by default CUDA if available.
        """
        self.checkpoint_path = checkpoint_path
        self.predict_args = predict_args
        self.diffusion_args = diffusion_args
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

        self.model = self._load_model(checkpoint_path)
        self.cached_representations: dict[str, Any] = {}

        sigmas = self._compute_noise_schedule(self.predict_args.sampling_steps)
        gammas = torch.where(
            sigmas > self.diffusion_args.gamma_min,
            self.diffusion_args.gamma_0,
            0.0,
        )
        self.noise_schedule: dict[str, Float[Tensor, ...]] = {
            "sigma_tm": sigmas[:-1],
            "sigma_t": sigmas[1:],
            "gamma": gammas[1:],
        }

    def _load_model(self, checkpoint_path: str | Path) -> Protenix:
        """Load the Protenix model from checkpoint.

        Parameters
        ----------
        checkpoint_path : str | Path
            Path to model checkpoint file.

        Returns
        -------
        torch.nn.Module
            Loaded Protenix model in evaluation mode.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        state_dict = checkpoint["model"]
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        configs = checkpoint.get("configs", None)
        if configs is None:
            raise ValueError(
                "Checkpoint must contain 'configs' key with model configuration"
            )

        model = Protenix(configs)
        model.load_state_dict(state_dict)
        model = model.to(self.device).eval()

        return model

    def _compute_noise_schedule(self, num_steps: int) -> Tensor:
        """Compute the noise schedule for diffusion sampling.

        Follows Algorithm 18 from AlphaFold3, matching Protenix's InferenceNoiseScheduler.

        Parameters
        ----------
        num_steps : int
            Number of diffusion sampling steps.

        Returns
        -------
        Tensor
            Noise schedule with shape (num_steps + 1,).
        """
        inv_rho = 1 / self.diffusion_args.rho

        steps = torch.arange(num_steps, device=self.device, dtype=torch.float32)
        sigmas = (
            self.diffusion_args.sigma_max**inv_rho
            + steps
            / (num_steps - 1)
            * (
                self.diffusion_args.sigma_min**inv_rho
                - self.diffusion_args.sigma_max**inv_rho
            )
        ) ** self.diffusion_args.rho

        sigmas = torch.nn.functional.pad(sigmas, (0, 1), value=0.0)

        return sigmas

    def _structure_to_features(self, structure: dict) -> dict[str, Any]:
        """Convert Atomworks structure to Protenix input features.

        Parameters
        ----------
        structure : dict
            Atomworks structure dictionary.

        Returns
        -------
        dict[str, Any]
            Protenix-compatible feature dictionary.
        """
        from protenix.data.json_to_feature import SampleDictToFeatures

        chain_info = structure.get("chain_info", {})
        entities = []

        for chain_id, info in chain_info.items():
            entity = {
                "name": chain_id,
                "id": chain_id,
                "type": info["chain_type"].name.lower(),
            }

            if info["chain_type"].is_protein():
                entity["sequence"] = info["processed_entity_canonical_sequence"]
            elif info["chain_type"].is_nucleic_acid():
                entity["sequence"] = info.get("processed_entity_canonical_sequence", "")
            elif info["chain_type"] == "LIGAND":
                entity["ccd"] = info.get("ccd_code", "UNK")

            entities.append(entity)

        sample_dict = {
            "name": structure.get("metadata", {}).get("name", "sample"),
            "entities": entities,
            "bonds": [],
        }

        featurizer = SampleDictToFeatures(sample_dict)
        features, _, _ = featurizer.get_feature_dict()

        for key in features:
            if isinstance(features[key], torch.Tensor):
                features[key] = features[key].to(self.device)
            elif isinstance(features[key], dict):
                for subkey in features[key]:
                    if isinstance(features[key][subkey], torch.Tensor):
                        features[key][subkey] = features[key][subkey].to(self.device)

        return features

    def featurize(self, structure: dict, **kwargs) -> dict[str, Any]:
        """From an Atomworks structure, calculate Protenix input features.

        Parameters
        ----------
        structure : dict
            Atomworks structure dictionary.
        **kwargs : dict, optional
            Additional arguments for feature generation.

        Returns
        -------
        dict[str, Any]
            Protenix input features.
        """
        features = self._structure_to_features(structure)

        if "asym_unit" in structure:
            atom_array = structure["asym_unit"]
            residues_with_occupancy = atom_array.occupancy > 0
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
        """Perform a pass through the Protenix structure module to obtain representations.

        Parameters
        ----------
        features : dict[str, Any]
            Model features as returned by featurize.
        grad_needed : bool, optional
            Whether gradients are needed for this pass, by default False.
        **kwargs : dict, optional
            Additional arguments.

        Returns
        -------
        dict[str, Any]
            Protenix model outputs including trunk representations (s_trunk, z_trunk).
        """
        with torch.set_grad_enabled(grad_needed):
            outputs = self.model.forward(
                input_feature_dict=features,
                label_full_dict={},
                label_dict={},
                mode="inference",
            )

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
            - eps: Tensor, optional
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
                eps = cast(Tensor, kwargs["eps"])
            else:
                timestep_scaling = self.get_timestep_scaling(timestep)
                eps = timestep_scaling["eps_scale"] * torch.randn_like(noisy_coords)
                t_hat = timestep_scaling["t_hat"]

            if kwargs.get("augmentation", True):
                noisy_coords = noisy_coords - noisy_coords.mean(dim=-2, keepdim=True)

            noisy_coords_eps = noisy_coords + eps

            t_hat_tensor = torch.tensor(
                [t_hat], device=noisy_coords.device, dtype=noisy_coords.dtype
            )
            if noisy_coords_eps.dim() == 2:
                noisy_coords_eps = noisy_coords_eps.unsqueeze(0)

            atom_coords_denoised = self.model.diffusion_module.forward(
                x_noisy=noisy_coords_eps,
                t_hat_noise_level=t_hat_tensor,
                input_feature_dict=features,
                s_inputs=outputs.get("s_inputs"),
                s_trunk=outputs.get("s_trunk"),
                z_trunk=outputs.get("z_trunk"),
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
                        "Input coordinates must be provided when align_to_input is True."
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
        """Return scaling constants for Proteinix diffusion at given timestep.

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
        eps_scale = self.diffusion_args.noise_scale * torch.sqrt(t_hat**2 - sigma_tm**2)

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
