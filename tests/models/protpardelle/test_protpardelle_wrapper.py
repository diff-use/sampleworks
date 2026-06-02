"""Tests for the Protpardelle structure-model wrapper.

These tests build a small, randomly-initialized ``ai-allatom`` model (see
``conftest.py``) so they exercise real featurization/sampling logic without
needing downloaded weights.
"""

import os
import tempfile

import pytest


# Ensure the model-params directory exists before protpardelle is imported,
# mirroring conftest (import order across conftests is not guaranteed).
os.environ.setdefault(
    "PROTPARDELLE_MODEL_PARAMS", tempfile.mkdtemp(prefix="protpardelle_model_params_")
)

pytest.importorskip(
    "protpardelle.core.models", reason="Protpardelle not installed in this environment"
)

import torch
from atomworks.enums import ChainType
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


def _protein_structure(*sequences: str) -> dict:
    """Build a minimal structure dict with protein chain_info only."""
    chain_ids = "ABCDEFGH"
    chain_info = {
        chain_ids[i]: {
            "chain_type": ChainType.POLYPEPTIDE_L,
            "processed_entity_canonical_sequence": seq,
        }
        for i, seq in enumerate(sequences)
    }
    return {"chain_info": chain_info, "metadata": {"id": "test"}}


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
        annotated = annotate_structure_for_protpardelle(
            structure, ensemble_size=4, num_steps=10
        )
        config = annotated["_protpardelle_config"]
        assert isinstance(config, ProtpardelleConfig)
        assert config.ensemble_size == 4
        assert config.num_steps == 10
        # Original structure is not mutated.
        assert "_protpardelle_config" not in structure

    def test_defaults_match_ai_allatom_recipe(self):
        config = ProtpardelleConfig()
        assert config.uniform_steps is True
        assert config.jump_steps is False
        assert config.sidechain_mode is False


class TestBuildSamplingKwargs:
    def test_defaults(self):
        kwargs = ProtpardelleWrapper._build_sampling_kwargs(ProtpardelleConfig())
        assert kwargs["uniform_steps"] is True
        assert kwargs["jump_steps"] is False
        assert kwargs["num_steps"] == 500

    def test_extra_overrides_take_precedence(self):
        config = ProtpardelleConfig(
            num_steps=500, extra_sampling_kwargs={"num_steps": 7, "noise_scale": 2.0}
        )
        kwargs = ProtpardelleWrapper._build_sampling_kwargs(config)
        assert kwargs["num_steps"] == 7
        assert kwargs["noise_scale"] == 2.0


class TestConstructorValidation:
    def test_requires_paths_or_model(self):
        with pytest.raises(ValueError, match="config_path and checkpoint_path"):
            ProtpardelleWrapper()


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
        structure = annotate_structure_for_protpardelle(
            _protein_structure(SEQ_A), ensemble_size=3
        )
        features = protpardelle_wrapper.featurize(structure)
        assert features.x_init.ndim == 3
        assert features.x_init.shape[0] == 3
        assert features.x_init.shape[2] == 3

    def test_conditioning_shapes(self, protpardelle_wrapper):
        structure = annotate_structure_for_protpardelle(
            _protein_structure(SEQ_A), ensemble_size=2
        )
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


class TestStep:
    def test_step_raises_without_features(self, protpardelle_wrapper):
        with pytest.raises(ValueError, match="features"):
            protpardelle_wrapper.step(None)

    @pytest.mark.slow
    def test_step_returns_coords(self, protpardelle_wrapper):
        short_seq = "ACDEFG"
        structure = annotate_structure_for_protpardelle(
            _protein_structure(short_seq), ensemble_size=1, num_steps=3, s_churn=0.0
        )
        features = protpardelle_wrapper.featurize(structure)
        result = protpardelle_wrapper.step(features)
        assert torch.is_tensor(result)
        assert result.shape == features.x_init.shape
        assert result.shape[-1] == 3
        assert torch.isfinite(result).all()
