"""
Common fixtures for metrics tests.
"""

import pytest
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from biotite.structure import AtomArrayStack
from sampleworks.utils.atom_array_utils import (
    select_altloc,
    select_backbone,
)


@pytest.fixture(scope="session")
def altlocA_backbone(structure_6b8x_with_altlocs) -> AtomArrayStack:
    altlocA = select_altloc(structure_6b8x_with_altlocs, "A", return_full_array=True)
    return ensure_atom_array_stack(select_backbone(altlocA))


@pytest.fixture(scope="session")
def altlocB_backbone(structure_6b8x_with_altlocs) -> AtomArrayStack:
    altlocB = select_altloc(structure_6b8x_with_altlocs, "B", return_full_array=True)
    return ensure_atom_array_stack(select_backbone(altlocB))
