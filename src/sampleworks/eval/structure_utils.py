import re
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import cast

import numpy as np
import torch
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from atomworks.io.utils.io_utils import load_any
from biotite.structure import AtomArray, AtomArrayStack, from_template
from loguru import logger
from sampleworks.core.rewards.protocol import RewardInputs
from sampleworks.eval.eval_dataclasses import ProteinConfig
from sampleworks.models.protocol import GenerativeModelInput
from sampleworks.utils.atom_space import AtomReconciler
from sampleworks.utils.framework_utils import match_batch


try:
    from sampleworks.models.protenix.structure_processing import (
        add_terminal_oxt_atoms as _add_terminal_oxt_atoms,
        ensure_atom_array as _ensure_atom_array,
        filter_zero_occupancy as _filter_zero_occupancy,
    )

    _HAS_PROTENIX = True
except ImportError:
    _HAS_PROTENIX = False
    _add_terminal_oxt_atoms = None  # ty:ignore[invalid-assignment]
    _ensure_atom_array = None  # ty:ignore[invalid-assignment]
    _filter_zero_occupancy = None  # ty:ignore[invalid-assignment]


# TODO: standardize this more! This needs to have a specified output dataclass that
# we deal with, and define how we couple the data here to the structure we output at the end
# of sampling.
@dataclass(frozen=True, slots=True)
class SampleworksProcessedStructure:
    """Processed structure and associated model and forward model info for sampling."""

    structure: dict
    model_input: GenerativeModelInput
    input_coords: torch.Tensor
    atom_array: AtomArray
    ensemble_size: int
    reconciler: AtomReconciler
    model_atom_array: AtomArray | None = None

    def to_reward_inputs(self, device: torch.device | str = "cpu") -> RewardInputs:
        """Build RewardInputs with model atom count when model atom arrays are available.

        Delegates to :meth:`RewardInputs.from_atom_array` so that element
        lookup, masking, and batching logic lives in one place.

        Reward tensors are generated from the model atom array whenever possible.
        Structure-derived B-factors are copied onto common atoms, and the
        reference coordinates come from ``self.input_coords`` (model-space).

        Parameters
        ----------
        device
            PyTorch device to place tensors on.

        Returns
        -------
        RewardInputs
        """
        reconciler = self.reconciler
        atom_array_for_rewards = self.model_atom_array or self.atom_array

        reward_inputs = RewardInputs.from_atom_array(
            atom_array=atom_array_for_rewards,
            ensemble_size=self.ensemble_size,
            device=device,
        )

        updated_b_factors = reward_inputs.b_factors
        if self.model_atom_array is not None and self.atom_array.b_factor is not None:
            struct_b_factors = torch.as_tensor(
                cast(np.ndarray, self.atom_array.b_factor)[
                    reconciler.struct_indices.detach().cpu().numpy()
                ],
                device=reward_inputs.b_factors.device,
                dtype=reward_inputs.b_factors.dtype,
            )
            model_indices = reconciler.model_indices.to(device=reward_inputs.b_factors.device)
            updated_b_factors = reward_inputs.b_factors.clone()
            updated_b_factors[..., model_indices] = struct_b_factors

        # input_coords stored on the processed structure are always full model atom
        # references, apply reward_param_mask for reward calls.
        masked_input_coords = self.input_coords[..., reward_inputs.reward_param_mask, :].to(
            device=reward_inputs.input_coords.device,
            dtype=reward_inputs.input_coords.dtype,
        )
        updated_mask_like = torch.ones_like(masked_input_coords[..., 0])

        return replace(
            reward_inputs,
            b_factors=updated_b_factors,
            input_coords=masked_input_coords,
            mask_like=updated_mask_like,
        )


# TODO: this function could maybe be an atomworks transform/use those?
def process_structure_to_trajectory_input(
    structure: dict,
    coords_from_prior: torch.Tensor,
    features: GenerativeModelInput,
    ensemble_size: int,
) -> SampleworksProcessedStructure:
    """Convert a structure dict and model features into a ready-to-sample bundle.

    CURRENTLY: Resolves the ground-truth ``AtomArray`` from either the conditioning
    attached to *features* or the ``"asym_unit"`` key of *structure*,
    applies Protenix-specific preprocessing when needed (OXT atoms,
    zero-occupancy filtering), masks invalid atoms, and tiles the
    coordinates to *ensemble_size*.

    IDEALLY: This function should not need to be aware of the model, and should
    produce a cleaned and masked AtomArray and coordinates usable for any model.
    The model-specific logic should be handled in the model.

    Parameters
    ----------
    structure : dict
        Structure dictionary; must contain ``"asym_unit"`` if the model
        conditioning does not carry a ``true_atom_array``.  Mutated
        in-place: ``"asym_unit"`` is replaced with the cleaned array.
    coords_from_prior : torch.Tensor
        Prior sample used only to infer dtype and device for the output
        coordinate tensor. (in future this should be useful for more than that)
    features : GenerativeModelInput
        Featurized model input.  If ``features.conditioning`` exposes a
        ``true_atom_array`` attribute it is used as the reference array.
    ensemble_size : int
        Number of ensemble members; the reference coordinates are
        broadcast along a leading ``e`` dimension.

    Returns
    -------
    SampleworksProcessedStructure
        Frozen dataclass bundling the cleaned structure, model input,
        tiled coordinates, atom array, and ensemble size.
    """
    atom_array = None
    needs_protenix_preprocessing = False

    # TODO: still a bit jank, but it is better than before I think, and the transform related issues
    # will help when resolved
    conditioning = features.conditioning
    if hasattr(conditioning, "true_atom_array"):
        atom_array = conditioning.true_atom_array

    if atom_array is None:
        if "asym_unit" not in structure:
            raise ValueError("structure must contain 'asym_unit' key")
        atom_array = ensure_atom_array_stack(structure["asym_unit"])[0]
        if conditioning is not None:
            cond_class = conditioning.__class__.__name__
            needs_protenix_preprocessing = cond_class == "ProtenixConditioning"

    # Add OXT atoms for Protenix to match the model's internal representation.
    # Only needed when falling back to structure's asym_unit (true_atom_array was None).
    if _HAS_PROTENIX and needs_protenix_preprocessing:
        assert _ensure_atom_array is not None
        assert _filter_zero_occupancy is not None
        assert _add_terminal_oxt_atoms is not None
        atom_array = _ensure_atom_array(atom_array)
        atom_array = _filter_zero_occupancy(atom_array)
        atom_array = _add_terminal_oxt_atoms(atom_array, structure.get("chain_info", {}))

    # Mask to valid atoms (nonzero occupancy, no NaN coords) in structure atom space
    reward_param_mask = atom_array.occupancy > 0
    reward_param_mask &= ~np.any(np.isnan(atom_array.coord), axis=-1)
    atom_array = atom_array[reward_param_mask]

    # Build reconciler from model and structure atom arrays.
    model_atom_array = (
        cast(AtomArray | None, conditioning.model_atom_array)
        if hasattr(conditioning, "model_atom_array")
        else None
    )
    if model_atom_array is not None:
        reconciler = AtomReconciler.from_arrays(model_atom_array, atom_array)  # pyright: ignore[reportArgumentType]
        if reconciler.has_mismatch:
            logger.info(
                f"Atom count mismatch: model={reconciler.n_model}, "
                f"structure={reconciler.n_struct}, common={reconciler.n_common}"
            )
    else:
        reconciler = AtomReconciler.identity(len(atom_array))  # pyright: ignore[reportArgumentType]

    # Build model atom reference coordinates used for alignment and partial diffusion.
    struct_coords_np = np.ascontiguousarray(cast(np.ndarray, atom_array.coord))
    struct_coords = torch.from_numpy(struct_coords_np).to(
        dtype=coords_from_prior.dtype,
        device=coords_from_prior.device,
    )
    if model_atom_array is not None:
        model_template_np = np.asarray(cast(np.ndarray, model_atom_array.coord))

        # Replace non-finite coords with the common-atom centroid
        # TODO: use something like the RF3 processing where they put things on the nearest token
        # (this is in atomworks RF3 transform pipeline)
        struct_centroid = struct_coords_np.mean(axis=0)
        if not np.isfinite(model_template_np).all():
            logger.warning(
                "Model atom array contains non-finite coordinates; replacing with "
                "structure centroid for model-only template atoms."
            )
            bad = ~np.isfinite(model_template_np)
            model_template_np = model_template_np.copy()
            model_template_np[bad] = np.broadcast_to(struct_centroid, model_template_np.shape)[bad]
        model_template_np = np.ascontiguousarray(model_template_np)

        model_template = torch.from_numpy(model_template_np).to(
            dtype=coords_from_prior.dtype,
            device=coords_from_prior.device,
        )
        model_reference_coords = reconciler.struct_to_model(struct_coords, model_template)
    else:
        model_reference_coords = struct_coords

    input_coords = torch.as_tensor(match_batch(model_reference_coords.unsqueeze(0), ensemble_size))

    structure["asym_unit"] = atom_array

    return SampleworksProcessedStructure(
        structure=structure,
        model_input=features,
        input_coords=input_coords,
        atom_array=atom_array,
        ensemble_size=ensemble_size,
        reconciler=reconciler,
        model_atom_array=model_atom_array,
    )


def parse_selection_string(selection: str) -> tuple[str | None, int | None, int | None]:
    """Parse a selection string like 'chain A and resi 326-339'.

    Supports both residue ranges ('resi 10-50') and single residues ('resi 10').
    For single residues, resi_start and resi_end will be equal.

    Parameters
    ----------
    selection : str
        Selection string (e.g., 'chain A', 'resi 10', 'chain A and resi 10-50')

    Returns
    -------
    tuple
        (chain_id, resi_start, resi_end) - any component may be None if not specified
    """
    # Parse "chain X and resi N-M" format and generalizations of that.
    chain = re.search(r"chain\s+(\w+)", selection, re.IGNORECASE)
    if chain is not None:
        chain = chain.group(1).upper()

    residues = re.search(r"resi\s+(\d+)(?:-(\d+))?", selection, re.IGNORECASE)
    if residues is not None:
        resi_start = int(residues.group(1))
        resi_end = int(residues.group(2)) if residues.group(2) else resi_start
    else:
        resi_start = resi_end = None

    if chain is None and resi_start is None and resi_end is None:
        logger.warning(
            "Selection string did not match any known patterns (e.g. 'chain A', 'resi 10-50')"
        )

    return chain, resi_start, resi_end


def apply_selection(atom_array: AtomArray, selection: str | None) -> AtomArray:
    """Apply an atom selection string to filter a structure.

    Parameters
    ----------
    atom_array
        Structure to filter
    selection
        Selection string (e.g., 'chain A and resi 10-50'). If None, returns
        the entire structure unchanged.

    Returns
    -------
    AtomArray
        Filtered structure containing only atoms matching the selection

    Raises
    ------
    ValueError
        If the selection string matches no atoms
    """
    if selection is None:
        return atom_array

    chain_id, resi_start, resi_end = parse_selection_string(selection)
    mask = np.ones(len(atom_array), dtype=bool)

    if chain_id is not None:
        mask &= atom_array.chain_id == chain_id

    if resi_start is not None:
        res_ids = cast(np.ndarray, atom_array.res_id)
        if resi_end is not None:
            mask &= (res_ids >= resi_start) & (res_ids <= resi_end)
        else:
            mask &= res_ids == resi_start

    if mask.sum() == 0:
        raise ValueError(f"Selection '{selection}' matched no atoms")

    return cast(AtomArray, atom_array[mask])


def extract_selection_coordinates(
    atom_array: AtomArray | AtomArrayStack, selection: str
) -> np.ndarray:
    """
    Extract coordinates for atoms matching a selection from an atomworks structure.

    Parameters
    ----------
    atom_array : AtomArray | AtomArrayStack Atomworks parsed structure
    selection : str Selection string like 'chain A and resi 326-339'

    Returns
    -------
    np.ndarray Coordinates of selected atoms, shape (n_atoms, 3)

    Raises
    ------
    RuntimeError: If no atoms match the selection or coordinates are invalid
    TypeError: If the "asym_unit" in `structure` is not an AtomArray or AtomArrayStack
    """
    # TODO: we will need to handle other kinds of selections later, like radius around a point.
    #   surely biotite has this capability?
    if isinstance(atom_array, AtomArrayStack):
        working_array = cast(AtomArray, atom_array[0])
    else:
        working_array = atom_array

    chain_id, resi_start, resi_end = parse_selection_string(selection)

    # Create the selection mask, don't rely on len(atom_array) in case it is the ensemble size
    mask = np.ones(len(working_array), dtype=bool)

    if chain_id is not None:
        mask &= working_array.chain_id == chain_id

    if resi_start is not None:
        res_ids = cast(np.ndarray, working_array.res_id)
        if resi_end is not None:
            # Explicitly check for None to satisfy pyright
            start: int = resi_start
            end: int = resi_end
            mask &= (res_ids >= start) & (res_ids <= end)
        else:
            start = resi_start
            mask &= res_ids == start

    selected_coords = cast(np.ndarray, working_array.coord)[mask]

    # VALIDATION
    if len(selected_coords) == 0:
        raise RuntimeError(
            f"No atoms matched selection: '{selection}'. "
            f"Chain ID: {chain_id}, Residue range: {resi_start}-{resi_end}. "
            f"Total atoms in structure: {len(atom_array)}"
        )

    # TODO? if there are missing atoms, what happens to the computed density?
    #  Do we need to be careful here?
    # Filter out atoms with NaN or Inf coordinates (common in alt conf structures)
    finite_mask = np.isfinite(selected_coords).all(axis=1)
    if not finite_mask.all():
        n_invalid = (~finite_mask).sum()
        n_total = len(selected_coords)
        logger.warning(
            f"Filtered {n_invalid} atoms with NaN/Inf coordinates from "
            f"selection '{selection}' ({n_total - n_invalid} valid atoms remaining)"
        )
        selected_coords = selected_coords[finite_mask]

    # Check if we have any valid coordinates left
    if len(selected_coords) == 0:
        raise RuntimeError(
            f"No valid (finite) coordinates after filtering NaN/Inf from selection: '{selection}'"
        )

    return selected_coords


def get_asym_unit_from_structure(
    structure: dict, atom_array_index: int | None = None
) -> AtomArray | AtomArrayStack:
    """
    Extract the AtomArray from a structure dictionary, handling AtomArrayStack if present.
    optionally specify the index of the AtomArray in the stack (e.g. for NMR models).
    """
    atom_array = structure["asym_unit"]
    if atom_array_index and isinstance(atom_array, AtomArrayStack):
        atom_array = atom_array.get_array(atom_array_index)
    if not isinstance(atom_array, (AtomArray, AtomArrayStack)):
        raise TypeError(f"Unexpected atom array type: {type(atom_array)}")
    return atom_array


def get_reference_atomarraystack(
    protein_config: ProteinConfig, occupancy_a: float = 0.5
) -> tuple[Path | str | None, AtomArrayStack | None]:
    ref_path = protein_config.get_reference_structure_path(occupancy_a)  # will warn if not found
    if ref_path is None:
        return None, None
    ref_struct = load_any(ref_path, altloc="all", extra_fields=["occupancy"])
    if ref_struct.coord is None:
        raise ValueError(f"Unable to load coordinates from {ref_path} Please check file")
    if isinstance(ref_struct, AtomArray):
        ref_struct = from_template(ref_struct, ref_struct.coord[None, :, :])
    return ref_path, ref_struct


def get_reference_structure_coords(
    protein_config: ProteinConfig, protein_key: str, occ_list: tuple[float, ...] = (0.0, 1.0)
) -> np.ndarray | None:
    """
    This has a slightly odd function, which is to output an array of all possible coordinates
    of a structure, with altlocs mixed in. It returns NO information about which atom is which
    or whether there are duplicates. It's used for masking density maps.
    """
    protein_ref_coords_list = []
    for occ in occ_list:
        ref_path, ref_struct = get_reference_atomarraystack(protein_config, occ)
        if ref_path and ref_struct:  # if not None, it is already a validated Path object
            try:
                # TODO: enumerate actual exceptions this can raise.
                coords = extract_selection_coordinates(ref_struct, protein_config.selection)
                if not len(coords):
                    logger.warning(
                        f"  No atoms in selection '{protein_config.selection}' for {protein_key}"
                    )
                elif not np.isfinite(coords).all():
                    logger.warning(
                        f"  NaN/Inf coordinates in selection "
                        f"'{protein_config.selection}' for {protein_key}"
                    )
                else:
                    protein_ref_coords_list.append(coords)
                    logger.info(
                        f"  Loaded reference structure for {protein_key}: "
                        f"{len(coords)} atoms in selection '{protein_config.selection}'"
                    )
            except Exception as _e:
                _selection = protein_config.selection if protein_config.selection else "(none)"
                logger.error(
                    f"  ERROR: Failed to load reference structure for {protein_key}: {_e}\n"
                    f"    Path: {ref_path}\n"
                    f"    Selection: {_selection}\n"
                    f"    Traceback: {traceback.format_exc()}"
                )

    if not protein_ref_coords_list:
        logger.error(f"No reference structures found for {protein_key}")
        return None

    return np.vstack(protein_ref_coords_list)
