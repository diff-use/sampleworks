"""Tests for the Protpardelle structure-model wrapper.

These tests build a small, randomly-initialized ``ai-allatom`` model (see
``conftest.py``) so they exercise real featurization/sampling logic without
needing downloaded weights.
"""

import atexit
import os
import shutil
import tempfile

import pytest


# Ensure the model-params directory exists before protpardelle is imported, mirroring
# conftest (import order across conftests is not guaranteed). Whichever module runs first
# creates the throwaway dir and registers its cleanup; the other sees the var already set
# and does nothing. Guard explicitly rather than `setdefault`, which would eagerly leak a
# mkdtemp even when the var is present.
if "PROTPARDELLE_MODEL_PARAMS" not in os.environ:
    _model_params_dir = tempfile.mkdtemp(prefix="protpardelle_model_params_")
    os.environ["PROTPARDELLE_MODEL_PARAMS"] = _model_params_dir
    atexit.register(shutil.rmtree, _model_params_dir, ignore_errors=True)

pytest.importorskip(
    "protpardelle.core.models", reason="Protpardelle not installed in this environment"
)

import biotite.structure as struc
import numpy as np
import torch
from atomworks.enums import ChainType
from protpardelle.common import residue_constants
from protpardelle.data.sequence import seq_to_aatype
from sampleworks.models.protocol import GenerativeModelInput, StructureModelWrapper
from sampleworks.models.protpardelle.wrapper import (
    annotate_structure_for_protpardelle,
    extract_protein_sequences,
    NUM_AATYPE_TOKENS,
    ProtpardelleConditioning,
    ProtpardelleConfig,
    ProtpardelleWrapper,
)


SEQ_A = "ACDEFGHIKL"
SEQ_B = "MNPQRSTVWY"


def _build_asym_unit(sequences) -> struc.AtomArray:
    """Build a biotite AtomArray for the given chains.

    Each residue carries exactly the atom37 heavy atoms implied by its type
    (matching ``atom37_mask_from_aatype``), so the atom count agrees with the
    conditioning's ``atom_mask``. Coordinates are arbitrary but distinct.
    """
    chain_ids = "ABCDEFGH"
    atom_mask = np.asarray(residue_constants.restype_atom37_mask)
    atom_names, res_ids, chains = [], [], []
    for chain_idx, seq in enumerate(sequences):
        for res_pos, aa in enumerate(seq):
            restype = residue_constants.restype_order[aa]
            present = [
                residue_constants.atom_types[slot]
                for slot in range(residue_constants.atom_type_num)
                if atom_mask[restype][slot]
            ]
            atom_names.extend(present)
            res_ids.extend([res_pos + 1] * len(present))
            chains.extend([chain_ids[chain_idx]] * len(present))

    arr = struc.AtomArray(len(atom_names))
    arr.atom_name = np.array(atom_names)
    arr.res_id = np.array(res_ids)
    arr.chain_id = np.array(chains)
    arr.coord = np.arange(len(atom_names) * 3, dtype=np.float32).reshape(-1, 3)
    return arr


def _protein_structure(*sequences: str) -> dict:
    """Build a minimal structure dict with protein chain_info and an asym_unit."""
    chain_ids = "ABCDEFGH"
    chain_info = {
        chain_ids[i]: {
            "chain_type": ChainType.POLYPEPTIDE_L,
            "processed_entity_canonical_sequence": seq,
        }
        for i, seq in enumerate(sequences)
    }
    return {
        "chain_info": chain_info,
        "asym_unit": _build_asym_unit(sequences),
        "metadata": {"id": "test"},
    }


# ---------------------------------------------------------------------------
# Pure-logic tests (no model needed)
# ---------------------------------------------------------------------------


class TestExtractProteinSequences:
    def test_returns_sequences_in_chain_order(self):
        structure = _protein_structure(SEQ_A, SEQ_B)
        assert extract_protein_sequences(structure) == [SEQ_A, SEQ_B]

    def test_skips_non_protein_chains(self):
        structure = _protein_structure(SEQ_A)
        structure["chain_info"]["L"] = {
            "chain_type": ChainType.NON_POLYMER,
            "processed_entity_canonical_sequence": "",
        }
        assert extract_protein_sequences(structure) == [SEQ_A]

    def test_raises_without_chain_info(self):
        with pytest.raises(ValueError, match="chain_info"):
            extract_protein_sequences({"metadata": {"id": "x"}})

    def test_raises_without_protein_chains(self):
        structure = {
            "chain_info": {
                "L": {
                    "chain_type": ChainType.NON_POLYMER,
                    "processed_entity_canonical_sequence": "",
                }
            }
        }
        with pytest.raises(ValueError, match="protein"):
            extract_protein_sequences(structure)


class TestAnnotateStructure:
    def test_adds_config(self):
        structure = _protein_structure(SEQ_A)
        annotated = annotate_structure_for_protpardelle(structure, ensemble_size=4)
        config = annotated["_protpardelle_config"]
        assert isinstance(config, ProtpardelleConfig)
        assert config.ensemble_size == 4
        # Original structure is not mutated.
        assert "_protpardelle_config" not in structure

    def test_default_ensemble_size(self):
        config = ProtpardelleConfig()
        assert config.ensemble_size == 8


class TestConstructorValidation:
    def test_requires_checkpoint_and_config_paths(self):
        # checkpoint_path, config_path and device are required positional args.
        with pytest.raises(TypeError, match="required positional argument"):
            ProtpardelleWrapper()  # ty: ignore[missing-argument]


class TestProtocolConformance:
    def test_is_structure_model_wrapper(self, protpardelle_wrapper):
        assert isinstance(protpardelle_wrapper, StructureModelWrapper)


# ---------------------------------------------------------------------------
# Featurization tests (small random model)
# ---------------------------------------------------------------------------


class TestFeaturize:
    def test_returns_generative_model_input(self, protpardelle_wrapper):
        structure = _protein_structure(SEQ_A)
        features = protpardelle_wrapper.featurize(structure)
        assert isinstance(features, GenerativeModelInput)
        assert isinstance(features.conditioning, ProtpardelleConditioning)
        assert features.x_init is not None

    def test_x_init_shape(self, protpardelle_wrapper):
        structure = annotate_structure_for_protpardelle(_protein_structure(SEQ_A), ensemble_size=3)
        features = protpardelle_wrapper.featurize(structure)
        assert features.x_init.ndim == 3
        assert features.x_init.shape[0] == 3
        assert features.x_init.shape[2] == 3

    def test_conditioning_shapes(self, protpardelle_wrapper):
        structure = annotate_structure_for_protpardelle(_protein_structure(SEQ_A), ensemble_size=2)
        cond = protpardelle_wrapper.featurize(structure).conditioning
        length = len(SEQ_A)
        assert cond.aatype.shape == (2, length)
        assert cond.seq_mask.shape == (2, length)
        assert cond.residue_index.shape == (2, length)
        assert cond.atom_mask.shape == (2, length, 37)
        assert cond.sequences == (SEQ_A,)

    def test_aatype_matches_sequence(self, protpardelle_wrapper):
        structure = _protein_structure(SEQ_A)
        cond = protpardelle_wrapper.featurize(structure).conditioning
        expected = seq_to_aatype(SEQ_A, num_tokens=NUM_AATYPE_TOKENS)
        assert torch.equal(cond.aatype[0].cpu(), expected)

    def test_x_init_atom_count_matches_atom_mask(self, protpardelle_wrapper):
        structure = _protein_structure(SEQ_A)
        features = protpardelle_wrapper.featurize(structure)
        cond = features.conditioning
        n_atoms = int(cond.atom_mask[0].sum().item())
        assert features.x_init.shape[1] == n_atoms

    def test_atom37_index_maps_present(self, protpardelle_wrapper):
        structure = _protein_structure(SEQ_A)
        cond = protpardelle_wrapper.featurize(structure).conditioning
        n_atoms = int(cond.atom_mask[0].sum().item())
        # One entry per flat atom, matching the atom37 occupancy count.
        assert cond.atom37_residue_index.shape == (n_atoms,)
        assert cond.atom37_atom_index.shape == (n_atoms,)
        # Slots are valid atom37 indices; residues stay within L.
        assert int(cond.atom37_atom_index.max()) < 37
        assert int(cond.atom37_residue_index.max()) == len(SEQ_A) - 1

    def test_mse_selenium_maps_to_methionine_sd(self, protpardelle_wrapper):
        structure = _protein_structure("M")
        atom_array = structure["asym_unit"]
        selenium_mask = atom_array.atom_name == "SD"
        atom_array.atom_name[selenium_mask] = "SE"

        cond = protpardelle_wrapper.featurize(structure).conditioning
        selenium_index = int(np.flatnonzero(selenium_mask)[0])

        assert cond.atom37_atom_index[selenium_index].item() == residue_constants.atom_order["SD"]

    def test_featurize_requires_asym_unit(self, protpardelle_wrapper):
        structure = _protein_structure(SEQ_A)
        del structure["asym_unit"]
        with pytest.raises(ValueError, match="asym_unit"):
            protpardelle_wrapper.featurize(structure)

    def test_multichain_indices_have_gap(self, protpardelle_wrapper):
        structure = _protein_structure(SEQ_A, SEQ_B)
        cond = protpardelle_wrapper.featurize(structure).conditioning
        # Two distinct chains present.
        chain_ids = torch.unique(cond.chain_index[0])
        assert chain_ids.numel() == 2
        # The residue index jumps by more than 1 at the chain boundary.
        residue_index = cond.residue_index[0]
        boundary_gap = residue_index[len(SEQ_A)] - residue_index[len(SEQ_A) - 1]
        assert boundary_gap > 1


class TestInitializeFromPrior:
    def test_with_shape(self, protpardelle_wrapper):
        out = protpardelle_wrapper.initialize_from_prior(2, shape=(40, 3))
        assert out.shape == (2, 40, 3)

    def test_with_features(self, protpardelle_wrapper):
        features = protpardelle_wrapper.featurize(_protein_structure(SEQ_A))
        out = protpardelle_wrapper.initialize_from_prior(3, features=features)
        assert out.shape[0] == 3
        assert out.shape[-1] == 3

    def test_raises_without_features_or_shape(self, protpardelle_wrapper):
        with pytest.raises(ValueError, match="features|shape"):
            protpardelle_wrapper.initialize_from_prior(batch_size=2)

    def test_invalid_shape_raises(self, protpardelle_wrapper):
        with pytest.raises(ValueError, match="num_atoms"):
            protpardelle_wrapper.initialize_from_prior(1, shape=(40, 4))


# ---------------------------------------------------------------------------
# Step / sampling tests (small random model, short trajectory)
# ---------------------------------------------------------------------------


class TestConvertToAtom37:
    def test_shape_and_placement(self, protpardelle_wrapper):
        features = protpardelle_wrapper.featurize(_protein_structure(SEQ_A))
        cond = features.conditioning
        n_atoms = int(cond.atom_mask[0].sum().item())

        x_t = torch.randn(2, n_atoms, 3)
        out = protpardelle_wrapper._convert_to_atom37(cond, x_t)

        assert out.shape == (2, len(SEQ_A), 37, 3)
        # Every flat atom lands at its (residue, slot) destination.
        res_idx = cond.atom37_residue_index
        atom_idx = cond.atom37_atom_index
        for n in range(n_atoms):
            assert torch.allclose(out[:, res_idx[n], atom_idx[n]], x_t[:, n])
        # Total non-zero atom slots equals the number of placed atoms.
        occupied = (out.abs().sum(-1) > 0).sum().item()
        assert occupied == 2 * n_atoms

    def test_gradients_flow_back_to_x_t(self, protpardelle_wrapper):
        features = protpardelle_wrapper.featurize(_protein_structure(SEQ_A))
        cond = features.conditioning
        n_atoms = int(cond.atom_mask[0].sum().item())

        x_t = torch.randn(1, n_atoms, 3, requires_grad=True)
        out = protpardelle_wrapper._convert_to_atom37(cond, x_t)
        out.sum().backward()
        # Each input coordinate is placed exactly once -> unit gradient.
        assert torch.allclose(x_t.grad, torch.ones_like(x_t))

    def test_atom_count_mismatch_raises(self, protpardelle_wrapper):
        cond = protpardelle_wrapper.featurize(_protein_structure(SEQ_A)).conditioning
        with pytest.raises(ValueError, match="atoms"):
            protpardelle_wrapper._convert_to_atom37(cond, torch.randn(1, 3, 3))


class TestStep:
    def test_step_raises_without_features(self, protpardelle_wrapper):
        with pytest.raises(ValueError, match="features"):
            protpardelle_wrapper.step(torch.zeros(1, 1, 3), 0.0)

    @pytest.mark.slow
    def test_step_returns_coords(self, protpardelle_wrapper):
        short_seq = "ACDEFG"
        structure = annotate_structure_for_protpardelle(
            _protein_structure(short_seq), ensemble_size=1
        )
        features = protpardelle_wrapper.featurize(structure)
        result = protpardelle_wrapper.step(features.x_init, 1.0, features=features)
        assert torch.is_tensor(result)
        assert result.shape == features.x_init.shape
        assert result.shape[-1] == 3
        assert torch.isfinite(result).all()
