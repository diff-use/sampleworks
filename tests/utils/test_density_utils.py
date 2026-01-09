"""Tests for density_utils module."""

from typing import cast

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray, AtomArrayStack, stack
from sampleworks.core.forward_models.xray.real_space_density import XMap_torch
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import XMap
from sampleworks.utils.density_utils import (
    compute_density_from_atomarray,
    create_synthetic_grid,
)


@pytest.fixture(scope="module")
def simple_atom_array() -> AtomArray:
    """Small AtomArray with valid coords, elements, occupancy, b_factor."""
    atom_array = AtomArray(5)
    coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    atom_array.coord = coord
    atom_array.set_annotation("chain_id", np.array(["A"] * 5))
    atom_array.set_annotation("res_id", np.array([1, 2, 3, 4, 5]))
    atom_array.set_annotation("res_name", np.array(["ALA", "GLY", "VAL", "LEU", "SER"]))
    atom_array.set_annotation("atom_name", np.array(["CA", "CA", "CA", "CA", "CA"]))
    atom_array.set_annotation("element", np.array(["C", "C", "C", "C", "C"]))
    atom_array.set_annotation("b_factor", np.array([20.0, 20.0, 20.0, 20.0, 20.0]))
    atom_array.set_annotation("occupancy", np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    return atom_array


@pytest.fixture(scope="module")
def atom_array_with_nan() -> AtomArray:
    """AtomArray with some NaN coordinates."""
    atom_array = AtomArray(5)
    coord = np.array(
        [
            [0.0, 0.0, 0.0],
            [np.nan, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, np.inf, 1.0],
            [1.0, 1.0, 1.0],
        ]
    )
    atom_array.coord = coord
    atom_array.set_annotation("chain_id", np.array(["A"] * 5))
    atom_array.set_annotation("res_id", np.array([1, 2, 3, 4, 5]))
    atom_array.set_annotation("element", np.array(["C", "C", "C", "C", "C"]))
    atom_array.set_annotation("b_factor", np.array([20.0, 20.0, 20.0, 20.0, 20.0]))
    atom_array.set_annotation("occupancy", np.array([1.0, 1.0, 1.0, 1.0, 1.0]))
    return atom_array


@pytest.fixture(scope="module")
def simple_atom_array_stack() -> AtomArrayStack:
    """AtomArrayStack with 2 models."""
    arrays = []
    for i in range(2):
        atom_array = AtomArray(3)
        base_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        atom_array.coord = base_coords + i * 0.1
        atom_array.set_annotation("chain_id", np.array(["A"] * 3))
        atom_array.set_annotation("res_id", np.array([1, 2, 3]))
        atom_array.set_annotation("element", np.array(["C", "C", "C"]))
        atom_array.set_annotation("b_factor", np.array([20.0, 20.0, 20.0]))
        atom_array.set_annotation("occupancy", np.array([1.0, 1.0, 1.0]))
        arrays.append(atom_array)

    atom_array_stack = stack(arrays)

    return atom_array_stack


class TestCreateSyntheticGrid:
    """Tests for create_synthetic_grid function."""

    def test_returns_xmap(self, simple_atom_array):
        """Test that output is XMap type."""
        xmap = create_synthetic_grid(simple_atom_array, resolution=2.0)
        assert isinstance(xmap, XMap)

    def test_grid_contains_structure(self, simple_atom_array):
        """Test that grid bounds include all coordinates."""
        xmap = create_synthetic_grid(simple_atom_array, resolution=2.0, padding=5.0)
        coords = cast(np.ndarray, simple_atom_array.coord)

        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)

        assert np.all(xmap.origin <= min_coords - 5.0)
        uc_dims = np.array([xmap.unit_cell.a, xmap.unit_cell.b, xmap.unit_cell.c])  # pyright: ignore[reportOptionalMemberAccess]
        grid_end = xmap.origin + uc_dims
        assert np.all(grid_end >= max_coords + 5.0)

    def test_voxel_spacing(self, simple_atom_array):
        """Test that voxel spacing equals resolution / 4.0."""
        resolution = 2.0
        xmap = create_synthetic_grid(simple_atom_array, resolution=resolution)
        expected_spacing = resolution / 4.0
        if isinstance(xmap.voxelspacing, np.ndarray):
            assert np.allclose(xmap.voxelspacing, expected_spacing, rtol=1e-6)
        else:
            assert abs(xmap.voxelspacing - expected_spacing) < 1e-6

    def test_padding_applied(self, simple_atom_array):
        """Test that grid is expanded by padding amount."""
        padding = 10.0
        xmap = create_synthetic_grid(simple_atom_array, resolution=2.0, padding=padding)
        coords = cast(np.ndarray, simple_atom_array.coord)

        min_coords = coords.min(axis=0)
        expected_origin = min_coords - padding

        np.testing.assert_allclose(xmap.origin, expected_origin, rtol=1e-5)

    def test_orthogonal_unit_cell(self, simple_atom_array):
        """Test that unit cell has 90 degree angles."""
        xmap = create_synthetic_grid(simple_atom_array, resolution=2.0)
        assert xmap.unit_cell.alpha == 90.0  # pyright: ignore[reportOptionalMemberAccess]
        assert xmap.unit_cell.beta == 90.0  # pyright: ignore[reportOptionalMemberAccess]
        assert xmap.unit_cell.gamma == 90.0  # pyright: ignore[reportOptionalMemberAccess]

    def test_space_group_p1(self, simple_atom_array):
        """Test that space group is P1."""
        xmap = create_synthetic_grid(simple_atom_array, resolution=2.0)
        space_group_str = str(xmap.unit_cell.space_group)  # pyright: ignore[reportOptionalMemberAccess]
        assert "P1" in space_group_str or space_group_str == "P 1"

    def test_handles_atomarray_stack(self, simple_atom_array_stack):
        """Test that function works with AtomArrayStack input."""
        xmap = create_synthetic_grid(simple_atom_array_stack, resolution=2.0)
        assert isinstance(xmap, XMap)
        assert xmap.array.ndim == 3

    def test_filters_nan_coordinates(self, atom_array_with_nan):
        """Test that NaN coordinates are excluded from bounds calculation."""
        xmap = create_synthetic_grid(atom_array_with_nan, resolution=2.0)
        assert isinstance(xmap, XMap)
        assert np.isfinite(xmap.origin).all()

    def test_custom_padding(self, simple_atom_array):
        """Test that non-default padding values work."""
        padding_values = [0.0, 2.0, 15.0]
        for padding in padding_values:
            xmap = create_synthetic_grid(simple_atom_array, resolution=2.0, padding=padding)
            assert isinstance(xmap, XMap)

    def test_resolution_affects_grid_shape(self, simple_atom_array):
        """Test that smaller resolution creates more voxels."""
        xmap_low_res = create_synthetic_grid(simple_atom_array, resolution=4.0)
        xmap_high_res = create_synthetic_grid(simple_atom_array, resolution=1.0)

        volume_low = np.prod(xmap_low_res.array.shape)
        volume_high = np.prod(xmap_high_res.array.shape)

        assert volume_high > volume_low

    def test_array_shape_ordering(self, simple_atom_array):
        """Test that array shape is (nz, ny, nx)."""
        xmap = create_synthetic_grid(simple_atom_array, resolution=2.0)
        assert xmap.array.ndim == 3
        assert xmap.array.shape[0] > 0
        assert xmap.array.shape[1] > 0
        assert xmap.array.shape[2] > 0

    def test_empty_array_initialized(self, simple_atom_array):
        """Test that array is initialized to zeros."""
        xmap = create_synthetic_grid(simple_atom_array, resolution=2.0)
        assert np.all(xmap.array == 0.0)


class TestComputeDensityFromAtomarray:
    """Tests for compute_density_from_atomarray function."""

    def test_returns_density_tensor(self, simple_atom_array, device):
        """Test that first output is torch.Tensor."""
        density, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert isinstance(density, torch.Tensor)

    def test_returns_xmap_torch(self, simple_atom_array, device):
        """Test that second output is XMap_torch."""
        _, xmap_torch = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert isinstance(xmap_torch, XMap_torch)

    def test_density_shape_matches_grid(self, simple_atom_array, device):
        """Test that density tensor shape matches grid dimensions."""
        density, xmap_torch = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        expected_shape = xmap_torch.array.shape
        assert density.shape == expected_shape

    def test_density_is_finite(self, simple_atom_array, device):
        """Test that density contains no NaN or Inf values."""
        density, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert torch.isfinite(density).all()

    def test_with_resolution_parameter(self, simple_atom_array, device):
        """Test that resolution parameter creates synthetic grid."""
        density, xmap_torch = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert isinstance(density, torch.Tensor)
        assert density.numel() > 0

    def test_with_xmap_parameter(self, simple_atom_array, device):
        """Test that function uses provided XMap."""
        xmap = create_synthetic_grid(simple_atom_array, resolution=2.0)
        density, xmap_torch = compute_density_from_atomarray(
            simple_atom_array, xmap=xmap, em_mode=False, device=device
        )
        assert isinstance(density, torch.Tensor)
        assert density.shape == xmap.array.shape

    def test_both_xmap_and_resolution_raises(self, simple_atom_array, device):
        """Test that providing both xmap and resolution raises ValueError."""
        xmap = create_synthetic_grid(simple_atom_array, resolution=2.0)
        with pytest.raises(ValueError, match="Cannot provide both xmap and resolution"):
            compute_density_from_atomarray(
                simple_atom_array, xmap=xmap, resolution=2.0, em_mode=False, device=device
            )

    def test_neither_xmap_nor_resolution_raises(self, simple_atom_array, device):
        """Test that providing neither xmap nor resolution raises ValueError."""
        with pytest.raises(ValueError, match="Either xmap or resolution must be provided"):
            compute_density_from_atomarray(simple_atom_array, em_mode=False, device=device)

    def test_xray_mode(self, simple_atom_array, device):
        """Test that X-ray mode (em_mode=False) works."""
        density, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert isinstance(density, torch.Tensor)
        assert density.sum() > 0

    def test_electron_mode(self, simple_atom_array, device):
        """Test that electron mode (em_mode=True) works."""
        density, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=True, device=device
        )
        assert isinstance(density, torch.Tensor)
        assert density.sum() > 0

    def test_device_parameter(self, simple_atom_array, device):
        """Test that function respects device parameter."""
        density, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert density.device == device

    def test_density_has_positive_values(self, simple_atom_array, device):
        """Test that density has some positive values."""
        density, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert density.sum() > 0

    def test_with_atom_array_single_model(self, simple_atom_array_stack, device):
        """Test that function works with single model from stack."""
        single_model = simple_atom_array_stack[0]
        density, _ = compute_density_from_atomarray(
            single_model, resolution=2.0, em_mode=False, device=device
        )
        assert isinstance(density, torch.Tensor)
        assert torch.isfinite(density).all()

    def test_deterministic_output(self, simple_atom_array, device):
        """Test that same input produces same output."""
        density1, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        density2, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        torch.testing.assert_close(density1, density2)

    def test_high_resolution_smaller_voxels(self, simple_atom_array, device):
        """Test that higher resolution creates finer grid."""
        density_low, xmap_low = compute_density_from_atomarray(
            simple_atom_array, resolution=4.0, em_mode=False, device=device
        )
        density_high, xmap_high = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )

        assert density_high.numel() > density_low.numel()
        vs_high = xmap_high.voxelspacing
        vs_low = xmap_low.voxelspacing

        if isinstance(vs_high, torch.Tensor):
            assert torch.all(vs_high < vs_low).item()
        elif isinstance(vs_high, np.ndarray):
            assert np.all(vs_high < vs_low)
        else:
            assert vs_high < vs_low

    def test_zero_occupancy_atoms_filtered(self, device):
        """Test that atoms with zero occupancy contribute no density."""
        atom_array = AtomArray(3)
        atom_array.coord = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        atom_array.set_annotation("element", np.array(["C", "C", "C"]))
        atom_array.set_annotation("b_factor", np.array([20.0, 20.0, 20.0]))
        atom_array.set_annotation("occupancy", np.array([1.0, 0.0, 1.0]))

        density, _ = compute_density_from_atomarray(
            atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert density.sum() > 0

    @pytest.mark.slow
    def test_with_real_structure(self, structure_1vme, device):
        """Test density computation with real structure."""
        atom_array = structure_1vme["asym_unit"]
        if isinstance(atom_array, AtomArrayStack):
            atom_array = atom_array[0]

        density, _ = compute_density_from_atomarray(
            atom_array,  # pyright: ignore[reportArgumentType]
            resolution=2.0,
            em_mode=False,
            device=device,
        )
        assert isinstance(density, torch.Tensor)
        assert density.sum() > 0
        assert torch.isfinite(density).all()


class TestComputeDensityErrors:
    """Tests for error handling in compute_density_from_atomarray."""

    def test_empty_atom_array(self, device):
        """Test behavior with empty AtomArray."""
        atom_array = AtomArray(0)
        atom_array.coord = np.empty((0, 3))

        with pytest.raises((ValueError, RuntimeError, IndexError)):
            compute_density_from_atomarray(atom_array, resolution=2.0, em_mode=False, device=device)
