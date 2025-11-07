from collections.abc import Mapping
from dataclasses import dataclass
from logging import getLogger, Logger
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from atomworks.enums import ChainType
from biotite.structure import get_chain_starts, get_residue_starts
from configs.configs_base import configs as configs_base
from configs.configs_data import data_configs
from configs.configs_inference import inference_configs
from configs.configs_model_type import model_configs
from einops import einsum
from jaxtyping import ArrayLike, Float
from ml_collections import ConfigDict
from protenix.config import parse_configs
from protenix.data.constants import STD_RESIDUES
from protenix.data.infer_data_pipeline import get_inference_dataloader
from protenix.data.utils import (
    get_lig_lig_bonds,
    get_ligand_polymer_bond_mask,
    get_polymer_polymer_bond,
)
from protenix.model.protenix import InferenceNoiseScheduler
from protenix.model.utils import centre_random_augmentation
from protenix.utils.torch_utils import autocasting_disable_decorator
from runner.inference import (
    download_infercence_cache as download_inference_cache,
    InferenceRunner,
)
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


def create_protenix_input_from_structure(
    structure: dict, out_dir: str | Path, wrapper: "ProtenixWrapper"
) -> tuple[Path, dict]:
    """Create Protenix input JSON from Atomworks structure.

    Parameters
    ----------
    structure : dict
        Atomworks structure dictionary.
    out_dir : str | Path
        Output directory for saving JSON file.
    wrapper : ProtenixWrapper
        Wrapper instance to access _structure_to_protenix_json method.

    Returns
    -------
    tuple[Path, dict]
        Path to saved JSON file and JSON dictionary.
    """
    import json

    out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
    out_dir = out_dir.expanduser().resolve()

    json_dict = wrapper._structure_to_protenix_json(structure)

    protenix_input_path = out_dir / "protenix_input.json"
    protenix_input_path.parent.mkdir(parents=True, exist_ok=True)

    with open(protenix_input_path, "w") as f:
        json.dump([json_dict], f, indent=4)

    return protenix_input_path, json_dict


@dataclass
class ProtenixPredictArgs:
    """Arguments for Protenix model prediction."""

    recycling_steps: int = 3
    sampling_steps: int = 200
    diffusion_samples: int = 1
    num_ensemble: int = 1


@dataclass
class ProtenixDiffusionParams:
    """Diffusion process parameters for Protenix."""

    sigma_min: float = 1e-4
    sigma_max: float = 160.0
    sigma_data: float = 16.0
    rho: float = 7.0
    gamma0: float = 0.8
    gamma_min: float = 1.0
    noise_scale_lambda: float = 1.003
    step_scale_eta: float = 1.5


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

        # NOTE: weird things might happen here due to the InferenceRunner loading things
        # onto a different device initially than what we might want. The DIST_WRAPPER
        # may interact weirdly as well with our code, something to look out for
        # especially when scaling up and running multiple guidance runs in parallel.
        protenix_runner = InferenceRunner(self.configs)
        protenix_runner.device = self.device
        torch.cuda.set_device(self.device)

        self.model = protenix_runner.model.to(self.device)

        self.dataloader = get_inference_dataloader(self.configs)

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
            sigma_data=cast(dict, self.configs.sample_diffusion)["sigma_data"],
        )
        return scheduler(N_step=num_steps, device=self.device)

    @staticmethod
    def _add_unique_chain_and_copy_ids(atom_array):
        """Add unique chain_id and copy_id annotations to AtomArray.

        Parameters
        ----------
        atom_array : AtomArray
            Biotite AtomArray to annotate.

        Returns
        -------
        AtomArray
            Annotated AtomArray with chain_id and copy_id fields.
        """
        chain_starts = get_chain_starts(atom_array, add_exclusive_stop=False)
        chain_starts_atom_array = atom_array[chain_starts]

        unique_label_entity_id = np.unique(atom_array.label_entity_id)
        chain_id_to_copy_id_dict = {}

        for label_entity_id in unique_label_entity_id:
            chain_ids_in_entity = chain_starts_atom_array.chain_id[
                chain_starts_atom_array.label_entity_id == label_entity_id
            ]
            for chain_count, chain_id in enumerate(chain_ids_in_entity):
                chain_id_to_copy_id_dict[chain_id] = chain_count + 1

        copy_id = np.vectorize(chain_id_to_copy_id_dict.get)(atom_array.chain_id)
        atom_array.set_annotation("copy_id", copy_id)

        return atom_array

    @staticmethod
    def _get_sequences(atom_array, chain_info):
        """Extract entity sequences from AtomArray.

        Parameters
        ----------
        atom_array : AtomArray
            Biotite AtomArray containing structure.
        chain_info : dict[str, Any]
            Atomworks chain information dictionary.

        Returns
        -------
        dict[str, str]
            Mapping from label_entity_id to sequence string.
        """
        entity_seq = {}
        for label_entity_id in np.unique(atom_array.label_entity_id):
            for chain_id, info in chain_info.items():
                chain_atom = atom_array[atom_array.chain_id == chain_id]
                if len(chain_atom) > 0:
                    if chain_atom[0].label_entity_id == label_entity_id:
                        chain_type = info["chain_type"]
                        if chain_type.is_polymer():
                            entity_seq[label_entity_id] = info.get(
                                "processed_entity_canonical_sequence", ""
                            )
                        break
        return entity_seq

    @staticmethod
    def _get_poly_res_names(atom_array, chain_info):
        """Get residue names for polymer entities.

        Parameters
        ----------
        atom_array : AtomArray
            Biotite AtomArray containing structure.
        chain_info : dict[str, Any]
            Atomworks chain information dictionary.

        Returns
        -------
        dict[str, list[str]]
            Mapping from label_entity_id to list of residue names.
        """
        poly_res_names = {}
        for label_entity_id in np.unique(atom_array.label_entity_id):
            for chain_id, info in chain_info.items():
                chain_atom = atom_array[atom_array.chain_id == chain_id]
                if len(chain_atom) > 0:
                    if chain_atom[0].label_entity_id == label_entity_id:
                        chain_type = info["chain_type"]
                        if chain_type.is_polymer():
                            entity_array = atom_array[
                                atom_array.label_entity_id == label_entity_id
                            ]
                            starts = get_residue_starts(
                                entity_array, add_exclusive_stop=True
                            )
                            res_names = entity_array.res_name[starts[:-1]].tolist()
                            poly_res_names[label_entity_id] = res_names
                        break
        return poly_res_names

    @staticmethod
    def _detect_modifications(atom_array, chain_info):
        """Detect polymer modifications (non-standard residues).

        Parameters
        ----------
        atom_array : AtomArray
            Biotite AtomArray containing structure.
        chain_info : dict[str, Any]
            Atomworks chain information dictionary.

        Returns
        -------
        dict[str, list[tuple[int, str]]]
            Mapping from label_entity_id to list of (position, mod_ccd_code).
        """
        entity_id_to_mod_list = {}
        poly_res_names = ProtenixWrapper._get_poly_res_names(atom_array, chain_info)

        for entity_id, res_names in poly_res_names.items():
            modifications_list = []
            for idx, res_name in enumerate(res_names):
                if res_name not in STD_RESIDUES:
                    position = idx + 1
                    modifications_list.append((position, f"CCD_{res_name}"))
            if modifications_list:
                entity_id_to_mod_list[entity_id] = modifications_list

        return entity_id_to_mod_list

    @staticmethod
    def _merge_covalent_bonds(covalent_bonds, all_entity_counts):
        """Merge covalent bonds with same entity and position.

        Parameters
        ----------
        covalent_bonds : list[dict]
            List of covalent bond dictionaries.
        all_entity_counts : dict[str, int]
            Mapping of entity_id to chain count.

        Returns
        -------
        list[dict]
            List of merged covalent bond dictionaries.
        """
        from collections import defaultdict

        bonds_recorder = defaultdict(list)
        bonds_entity_counts = {}

        for bond_dict in covalent_bonds:
            bond_unique_string = []
            entity_counts = (
                all_entity_counts[str(bond_dict["entity1"])],
                all_entity_counts[str(bond_dict["entity2"])],
            )
            for i in range(2):
                for j in ["entity", "position", "atom"]:
                    k = f"{j}{i + 1}"
                    bond_unique_string.append(str(bond_dict[k]))
            bond_unique_string = "_".join(bond_unique_string)
            bonds_recorder[bond_unique_string].append(bond_dict)
            bonds_entity_counts[bond_unique_string] = entity_counts

        merged_covalent_bonds = []
        for k, v in bonds_recorder.items():
            counts1 = bonds_entity_counts[k][0]
            counts2 = bonds_entity_counts[k][1]
            if counts1 == counts2 == len(v):
                import copy

                bond_dict_copy = copy.deepcopy(v[0])
                del bond_dict_copy["copy1"]
                del bond_dict_copy["copy2"]
                merged_covalent_bonds.append(bond_dict_copy)
            else:
                merged_covalent_bonds.extend(v)

        return merged_covalent_bonds

    def _structure_to_protenix_json(self, structure: dict) -> dict[str, Any]:
        """Convert Atomworks structure to Protenix input JSON.

        Parameters
        ----------
        structure : dict
            Atomworks structure dictionary.

        Returns
        -------
        dict[str, Any]
            Protenix-compatible JSON dictionary.
        """
        if "asym_unit" not in structure:
            raise ValueError("structure must contain asym_unit key")

        atom_array = structure["asym_unit"]
        chain_info = structure.get("chain_info", {})

        entity_seq = self._get_sequences(atom_array, chain_info)
        atom_array = self._add_unique_chain_and_copy_ids(atom_array)

        label_entity_id_to_sequences = {}
        lig_chain_ids = []

        for label_entity_id in np.unique(atom_array.label_entity_id):
            entity_chain_type = None
            for chain_id, info in chain_info.items():
                chain_atom = atom_array[atom_array.chain_id == chain_id]
                if len(chain_atom) > 0:
                    if chain_atom[0].label_entity_id == label_entity_id:
                        entity_chain_type = info["chain_type"]
                        break

            if entity_chain_type and not entity_chain_type.is_polymer():
                current_lig_chain_ids = np.unique(
                    atom_array.chain_id[atom_array.label_entity_id == label_entity_id]
                ).tolist()
                lig_chain_ids += current_lig_chain_ids

                for chain_id in current_lig_chain_ids:
                    lig_atom_array = atom_array[atom_array.chain_id == chain_id]
                    starts = get_residue_starts(lig_atom_array, add_exclusive_stop=True)
                    seq = lig_atom_array.res_name[starts[:-1]].tolist()
                    label_entity_id_to_sequences[label_entity_id] = seq
                    break

        entity_id_to_mod_list = self._detect_modifications(atom_array, chain_info)

        chain_starts = get_chain_starts(atom_array, add_exclusive_stop=False)
        chain_starts_atom_array = atom_array[chain_starts]

        json_dict = {"sequences": []}

        unique_label_entity_id = np.unique(atom_array.label_entity_id)
        all_entity_counts = {}
        label_entity_id_to_entity_id_in_json = {}
        entity_idx = 0

        for label_entity_id in unique_label_entity_id:
            entity_dict = {}
            asym_chains = chain_starts_atom_array[
                chain_starts_atom_array.label_entity_id == label_entity_id
            ]

            entity_chain_type: ChainType | None = None
            for chain_id, info in chain_info.items():
                chain_atom = atom_array[atom_array.chain_id == chain_id]
                if len(chain_atom) > 0:
                    if chain_atom[0].label_entity_id == label_entity_id:
                        entity_chain_type = info["chain_type"]
                        break

            if not entity_chain_type:
                continue

            if entity_chain_type.is_polymer():
                if entity_chain_type in (
                    ChainType.POLYPEPTIDE_L,
                    ChainType.POLYPEPTIDE_D,
                ):
                    entity_type = "proteinChain"
                elif entity_chain_type == ChainType.DNA:
                    entity_type = "dnaSequence"
                elif entity_chain_type == ChainType.RNA:
                    entity_type = "rnaSequence"
                else:
                    continue

                sequence = entity_seq.get(label_entity_id, "")
                entity_dict["sequence"] = sequence
            else:
                entity_type = "ligand"
                lig_ccd = "_".join(
                    label_entity_id_to_sequences.get(label_entity_id, ["UNK"])
                )
                entity_dict["ligand"] = f"CCD_{lig_ccd}"

            entity_dict["count"] = len(asym_chains)
            entity_idx += 1
            entity_id_in_json = str(entity_idx)
            label_entity_id_to_entity_id_in_json[label_entity_id] = entity_id_in_json
            all_entity_counts[entity_id_in_json] = len(asym_chains)

            if label_entity_id in entity_id_to_mod_list:
                modifications = entity_id_to_mod_list[label_entity_id]
                if entity_type == "proteinChain":
                    entity_dict["modifications"] = [
                        {"ptmPosition": position, "ptmType": mod_ccd_code}
                        for position, mod_ccd_code in modifications
                    ]
                elif entity_type in ("dnaSequence", "rnaSequence"):
                    entity_dict["modifications"] = [
                        {
                            "basePosition": position,
                            "modificationType": mod_ccd_code,
                        }
                        for position, mod_ccd_code in modifications
                    ]

            json_dict["sequences"].append({entity_type: entity_dict})

        atom_array = atom_array[
            np.isin(
                atom_array.label_entity_id,
                list(label_entity_id_to_entity_id_in_json.keys()),
            )
        ]

        entity_poly_type = {}
        for chain_id, info in chain_info.items():
            chain_atom = atom_array[atom_array.chain_id == chain_id]
            if len(chain_atom) > 0:
                label_entity_id = chain_atom[0].label_entity_id
                chain_type = info["chain_type"]
                if chain_type.is_polymer():
                    if chain_type in (
                        ChainType.POLYPEPTIDE_L,
                        ChainType.POLYPEPTIDE_D,
                    ):
                        entity_poly_type[label_entity_id] = "polypeptide(L)"
                    elif chain_type == ChainType.DNA:
                        entity_poly_type[label_entity_id] = "polydeoxyribonucleotide"
                    elif chain_type == ChainType.RNA:
                        entity_poly_type[label_entity_id] = "polyribonucleotide"

        if not hasattr(atom_array, "token_mol_type"):
            token_mol_types = []
            for i in range(len(atom_array)):
                label_ent_id = atom_array.label_entity_id[i]
                if label_ent_id in entity_poly_type:
                    token_mol_types.append("PROTEIN")
                else:
                    token_mol_types.append("LIGAND")
            atom_array.set_annotation("token_mol_type", np.array(token_mol_types))

        has_modifications = len(entity_id_to_mod_list) > 0

        lig_polymer_bonds = get_ligand_polymer_bond_mask(
            atom_array, lig_include_ions=False
        )
        lig_lig_bonds = get_lig_lig_bonds(atom_array, lig_include_ions=False)

        has_ligand_bonds = lig_polymer_bonds.size > 0 or lig_lig_bonds.size > 0

        if has_modifications or has_ligand_bonds:
            token_bonds_list = []

            if has_ligand_bonds:
                ligand_bonds = np.vstack((lig_polymer_bonds, lig_lig_bonds))
                lig_indices = np.where(np.isin(atom_array.chain_id, lig_chain_ids))[0]
                lig_bond_mask = np.any(
                    np.isin(ligand_bonds[:, :2], lig_indices), axis=1
                )
                ligand_bonds = ligand_bonds[lig_bond_mask]
                if ligand_bonds.size > 0:
                    token_bonds_list.append(ligand_bonds)

            if has_modifications:
                polymer_polymer_bond = get_polymer_polymer_bond(
                    atom_array, entity_poly_type
                )
                if polymer_polymer_bond.size > 0:
                    token_bonds_list.append(polymer_polymer_bond)

            if token_bonds_list:
                token_bonds = np.vstack(token_bonds_list)
                covalent_bonds = []
                for atoms in token_bonds[:, :2]:
                    bond_dict = {}
                    for i in range(2):
                        position = atom_array.res_id[atoms[i]]
                        bond_dict[f"entity{i + 1}"] = int(
                            label_entity_id_to_entity_id_in_json[
                                atom_array.label_entity_id[atoms[i]]
                            ]
                        )
                        bond_dict[f"position{i + 1}"] = int(position)
                        bond_dict[f"atom{i + 1}"] = atom_array.atom_name[atoms[i]]
                        bond_dict[f"copy{i + 1}"] = int(atom_array.copy_id[atoms[i]])

                    covalent_bonds.append(bond_dict)

                merged_covalent_bonds = self._merge_covalent_bonds(
                    covalent_bonds, all_entity_counts
                )
                json_dict["covalent_bonds"] = merged_covalent_bonds

        json_dict["name"] = structure.get("metadata", {}).get("name", "sample")

        return json_dict

    def featurize(self, structure: dict, **kwargs) -> dict[str, Any]:
        """From an Atomworks structure, calculate Protenix input features.

        Parameters
        ----------
        structure : dict
            Atomworks structure dictionary.
        **kwargs : dict, optional
            Additional arguments for feature generation.
            - out_dir: Directory for saving intermediate JSON file

        Returns
        -------
        dict[str, Any]
            Protenix input features.
        """
        out_dir = kwargs.get(
            "out_dir", structure.get("metadata", {}).get("id", "protenix_output")
        )

        from protenix.data.infer_data_pipeline import InferenceDataset

        _, json_dict = create_protenix_input_from_structure(structure, out_dir, self)

        dataset = cast(InferenceDataset, self.dataloader.dataset)
        data, atom_array_protenix, _ = dataset.process_one(json_dict)

        atom_array = structure["asym_unit"]
        residues_with_occupancy = atom_array.occupancy > 0

        if "asym_unit" in structure:
            n_atoms_protenix = len(atom_array_protenix)
            n_atoms_atomworks = len(structure["asym_unit"][0][residues_with_occupancy])
            assert n_atoms_protenix == n_atoms_atomworks, (
                f"Coordinate count mismatch: Protenix processed {n_atoms_protenix} "
                f"atoms, Atomworks has {n_atoms_atomworks} atoms"
            )

        features = cast(dict[str, Any], data["input_feature_dict"])

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
                self.model.configs.skip_amp.sample_diffusion
            )(self.model.diffusion_module.diffusion_conditioning.prepare_cache)(
                features["relp"], z, False
            )
            outputs["p_lm/c_l"] = autocasting_disable_decorator(
                self.model.configs.skip_amp.sample_diffusion
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
