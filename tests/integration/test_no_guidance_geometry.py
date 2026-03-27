"""No guidance geometry validation across all model wrappers and structures.

Verifies that running each model wrapper through the sampleworks sampling
pipeline without guidance produces structures with valid geometry.
"""

import pytest
import torch
from biotite.structure import filter_linear_bond_continuity, filter_peptide_backbone
from sampleworks.core.samplers.edm import AF3EDMSampler, EDMSamplerConfig
from sampleworks.utils.guidance_constants import StructurePredictor

from tests.conftest import (
    annotate_structure_for_wrapper_type,
    get_fixture_name_for_wrapper_type,
    get_slow_wrappers,
    STRUCTURES,
)


NO_GUIDANCE_STRUCTURES: list[str] = [*STRUCTURES, "structure_5i09_density"]


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.parametrize("wrapper_type", get_slow_wrappers(), ids=lambda w: w.value)
@pytest.mark.parametrize(
    "structure_fixture",
    NO_GUIDANCE_STRUCTURES,
    ids=lambda s: s.replace("structure_", ""),
)
class TestNoGuidanceGeometry:
    """Validate that no-guidance sampling produces valid peptide geometry.

    Each test runs a short 20 step trajectory with no guidance and
    checks that the resulting peptide bond lengths are physically reasonable.
    """

    NUM_STEPS: int = 20

    def test_no_guidance_produces_valid_peptide_geometry(
        self,
        wrapper_type: StructurePredictor,
        structure_fixture: str,
        temp_output_dir,
        request,
    ):
        """
        No guidance sampling should produce structures with valid C-N bonds.
        """
        wrapper = request.getfixturevalue(get_fixture_name_for_wrapper_type(wrapper_type))
        structure = request.getfixturevalue(structure_fixture)
        device = wrapper.device if hasattr(wrapper, "device") else torch.device("cpu")

        annotated = annotate_structure_for_wrapper_type(
            wrapper_type, structure, temp_output_dir, ensemble_size=1
        )
        features = wrapper.featurize(annotated)

        sampler = AF3EDMSampler(
            EDMSamplerConfig(device=device, augmentation=False, align_to_input=False)
        )  # NOTE: augmentation and alignment might be useful to parametrize here?
        schedule = sampler.compute_schedule(num_steps=self.NUM_STEPS)

        torch.manual_seed(42)
        state = wrapper.initialize_from_prior(batch_size=1, features=features)

        for i in range(self.NUM_STEPS):
            context = sampler.get_context_for_step(i, schedule)
            step_output = sampler.step(
                state=state,
                model_wrapper=wrapper,
                context=context,
                scaler=None,
                features=features,
            )
            state = step_output.state

        cond = features.conditioning
        model_aa = cond.model_atom_array
        assert model_aa is not None, (
            f"model_atom_array is None for wrapper={wrapper_type.value}, "
            f"structure={structure_fixture}"
        )

        output_aa = model_aa.copy()
        # state shape: (batch=1, atoms, 3)
        output_aa.coord = state[0].detach().cpu().numpy()

        # Filter_peptide_backbone selects N, CA, C atoms, filter_linear_bond_continuity returns a
        # boolean mask where True means the distance to the next atom is within [min_len, max_len]
        bb_mask = filter_peptide_backbone(output_aa)
        bb = output_aa[bb_mask]
        assert len(bb) >= 6, (  # at least 2 residues (N, CA, C each)
            f"Too few backbone atoms ({len(bb)}) for "
            f"wrapper={wrapper_type.value}, structure={structure_fixture}"
        )

        # Generous bounds for a 20-step stochastic sample.
        # Ideal backbone bonds are 1.33–1.52 Å
        # We're a little more generous because of the short trajectory.
        con_mask = filter_linear_bond_continuity(bb, min_len=1.1, max_len=1.7)
        # Last element is always True (no next atom to compare), exclude it.
        n_valid = int(con_mask[:-1].sum())
        n_total = len(con_mask) - 1
        fraction_valid = n_valid / n_total

        assert fraction_valid >= 0.9, (
            f"Only {n_valid}/{n_total} ({fraction_valid:.1%}) backbone bonds "
            "are within [1.1, 1.7] Å for "
            f"wrapper={wrapper_type.value}, structure={structure_fixture}. "
            "This suggests something is wrong (an atom ordering problem?)."
        )
