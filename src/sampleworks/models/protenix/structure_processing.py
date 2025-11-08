import copy
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from atomworks.enums import ChainType
from biotite.structure import get_chain_starts, get_residue_starts
from protenix.data.constants import STD_RESIDUES
from protenix.data.utils import (
    get_lig_lig_bonds,
    get_ligand_polymer_bond_mask,
    get_polymer_polymer_bond,
)


def add_unique_chain_and_copy_ids(atom_array):
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


def get_sequences(atom_array, chain_info):
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


def get_poly_res_names(atom_array, chain_info):
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


def detect_modifications(atom_array, chain_info):
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
    poly_res_names = get_poly_res_names(atom_array, chain_info)

    for entity_id, res_names in poly_res_names.items():
        modifications_list = []
        for idx, res_name in enumerate(res_names):
            if res_name not in STD_RESIDUES:
                position = idx + 1
                modifications_list.append((position, f"CCD_{res_name}"))
        if modifications_list:
            entity_id_to_mod_list[entity_id] = modifications_list

    return entity_id_to_mod_list


def merge_covalent_bonds(covalent_bonds, all_entity_counts):
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
            bond_dict_copy = copy.deepcopy(v[0])
            del bond_dict_copy["copy1"]
            del bond_dict_copy["copy2"]
            merged_covalent_bonds.append(bond_dict_copy)
        else:
            merged_covalent_bonds.extend(v)

    return merged_covalent_bonds


def structure_to_protenix_json(structure: dict) -> dict[str, Any]:
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

    entity_seq = get_sequences(atom_array, chain_info)
    atom_array = add_unique_chain_and_copy_ids(atom_array)

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

    entity_id_to_mod_list = detect_modifications(atom_array, chain_info)

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

    lig_polymer_bonds = get_ligand_polymer_bond_mask(atom_array, lig_include_ions=False)
    lig_lig_bonds = get_lig_lig_bonds(atom_array, lig_include_ions=False)

    has_ligand_bonds = lig_polymer_bonds.size > 0 or lig_lig_bonds.size > 0

    if has_modifications or has_ligand_bonds:
        token_bonds_list = []

        if has_ligand_bonds:
            ligand_bonds = np.vstack((lig_polymer_bonds, lig_lig_bonds))
            lig_indices = np.where(np.isin(atom_array.chain_id, lig_chain_ids))[0]
            lig_bond_mask = np.any(np.isin(ligand_bonds[:, :2], lig_indices), axis=1)
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

            merged_covalent_bonds = merge_covalent_bonds(
                covalent_bonds, all_entity_counts
            )
            json_dict["covalent_bonds"] = merged_covalent_bonds

    json_dict["name"] = structure.get("metadata", {}).get("name", "sample")

    return json_dict


def create_protenix_input_from_structure(
    structure: dict, out_dir: str | Path
) -> tuple[Path, dict]:
    """Create Protenix input JSON from Atomworks structure.

    Parameters
    ----------
    structure : dict
        Atomworks structure dictionary.
    out_dir : str | Path
        Output directory for saving JSON file.

    Returns
    -------
    tuple[Path, dict]
        Path to saved JSON file and JSON dictionary.
    """
    out_dir = Path(out_dir) if isinstance(out_dir, str) else out_dir
    out_dir = out_dir.expanduser().resolve()

    json_dict = structure_to_protenix_json(structure)

    protenix_input_path = out_dir / "protenix_input.json"
    protenix_input_path.parent.mkdir(parents=True, exist_ok=True)

    with open(protenix_input_path, "w") as f:
        json.dump([json_dict], f, indent=4)

    return protenix_input_path, json_dict
