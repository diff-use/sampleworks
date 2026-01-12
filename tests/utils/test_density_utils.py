"""Tests for density_utils module."""

from typing import cast

import numpy as np
import pytest
import torch
from biotite.structure import AtomArray, AtomArrayStack
from sampleworks.core.forward_models.xray.real_space_density import XMap_torch
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import XMap
from sampleworks.utils.density_utils import (
    compute_density_from_atomarray,
    create_synthetic_grid,
)


class TestCreateSyntheticGrid:
    """Tests for create_synthetic_grid function."""

    @pytest.fixture(
        params=[
            {"resolution": 1.0, "padding": 5.0},
            {"resolution": 2.0, "padding": 5.0},
            {"resolution": 2.0, "padding": 10.0},
            {"resolution": 4.0, "padding": 5.0},
        ]
    )
    def synthetic_grid(self, request, simple_atom_array: AtomArray):
        params = request.param
        xmap = create_synthetic_grid(
            simple_atom_array, resolution=params["resolution"], padding=params["padding"]
        )
        resolution = params["resolution"]
        padding = params["padding"]
        return xmap, resolution, padding

    def test_returns_xmap(self, synthetic_grid):
        """Test that output is XMap type."""
        xmap, resolution, padding = synthetic_grid
        assert isinstance(xmap, XMap)

        assert xmap.array.ndim == 3
        assert np.all(xmap.array == 0.0)

        expected_spacing = resolution / 4.0
        if isinstance(xmap.voxelspacing, np.ndarray):
            assert np.allclose(xmap.voxelspacing, expected_spacing, rtol=1e-6)
        else:
            assert abs(xmap.voxelspacing - expected_spacing) < 1e-6

    def test_grid_contains_structure(self, simple_atom_array: AtomArray):
        """Test that grid bounds include all coordinates."""
        xmap = create_synthetic_grid(simple_atom_array, resolution=2.0, padding=5.0)
        coords = cast(np.ndarray, simple_atom_array.coord)

        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)

        assert np.all(xmap.origin <= min_coords - 5.0)
        uc_dims = np.array([xmap.unit_cell.a, xmap.unit_cell.b, xmap.unit_cell.c])  # pyright: ignore[reportOptionalMemberAccess]
        grid_end = xmap.origin + uc_dims
        assert np.all(grid_end >= max_coords + 5.0)

    def test_voxel_spacing(self, synthetic_grid):
        """Test that voxel spacing equals resolution / 4.0."""
        xmap, resolution, padding = synthetic_grid
        expected_spacing = resolution / 4.0
        if isinstance(xmap.voxelspacing, np.ndarray):
            assert np.allclose(xmap.voxelspacing, expected_spacing, rtol=1e-6)
        else:
            assert abs(xmap.voxelspacing - expected_spacing) < 1e-6

    def test_padding_applied(self, simple_atom_array: AtomArray):
        """Test that grid is expanded by padding amount."""
        padding = 10.0
        xmap = create_synthetic_grid(simple_atom_array, resolution=2.0, padding=padding)
        coords = cast(np.ndarray, simple_atom_array.coord)

        min_coords = coords.min(axis=0)
        expected_origin = min_coords - padding

        np.testing.assert_allclose(xmap.origin, expected_origin, rtol=1e-5)

    def test_orthogonal_unit_cell(self, synthetic_grid):
        """Test that unit cell has 90 degree angles."""
        xmap, resolution, padding = synthetic_grid
        assert xmap.unit_cell.alpha == 90.0  # pyright: ignore[reportOptionalMemberAccess]
        assert xmap.unit_cell.beta == 90.0  # pyright: ignore[reportOptionalMemberAccess]
        assert xmap.unit_cell.gamma == 90.0  # pyright: ignore[reportOptionalMemberAccess]

    def test_space_group_p1(self, synthetic_grid):
        """Test that space group is P1."""
        xmap, resolution, padding = synthetic_grid
        space_group_str = str(xmap.unit_cell.space_group)  # pyright: ignore[reportOptionalMemberAccess]
        assert "P1" in space_group_str or space_group_str == "P 1"

    def test_handles_atomarray_stack(self, simple_atom_array_stack: AtomArrayStack):
        """Test that function works with AtomArrayStack input."""
        xmap = create_synthetic_grid(simple_atom_array_stack, resolution=2.0)
        assert isinstance(xmap, XMap)
        assert xmap.array.ndim == 3

    def test_filters_nan_coordinates(self, atom_array_with_nan_coords: AtomArray):
        """Test that NaN coordinates are excluded from bounds calculation."""
        xmap = create_synthetic_grid(atom_array_with_nan_coords, resolution=2.0)
        assert isinstance(xmap, XMap)
        assert np.isfinite(xmap.origin).all()

    def test_custom_padding(self, simple_atom_array: AtomArray):
        """Test that non-default padding values work."""
        padding_values = [0.0, 2.0, 15.0]
        for padding in padding_values:
            xmap = create_synthetic_grid(simple_atom_array, resolution=2.0, padding=padding)
            assert isinstance(xmap, XMap)

    def test_resolution_affects_grid_shape(self, simple_atom_array: AtomArray):
        """Test that smaller resolution creates more voxels."""
        resolution_low = 4.0
        resolution_high = 1.0
        xmap_low_res = create_synthetic_grid(simple_atom_array, resolution=resolution_low)
        xmap_high_res = create_synthetic_grid(simple_atom_array, resolution=resolution_high)

        volume_low = np.prod(xmap_low_res.array.shape)
        volume_high = np.prod(xmap_high_res.array.shape)

        resolution_ratio = resolution_low / resolution_high
        # account for grid boundary effects by multiplying by 0.8
        expected_min_ratio = (resolution_ratio**3) * 0.8
        assert volume_high > volume_low * expected_min_ratio

    def test_array_shape_ordering(self, synthetic_grid):
        """Test that array shape is (nz, ny, nx)."""
        xmap, resolution, padding = synthetic_grid
        assert xmap.array.ndim == 3
        assert xmap.array.shape[0] > 0
        assert xmap.array.shape[1] > 0
        assert xmap.array.shape[2] > 0

    def test_empty_array_initialized(self, synthetic_grid):
        """Test that array is initialized to zeros."""
        xmap, resolution, padding = synthetic_grid
        assert np.all(xmap.array == 0.0)


class TestComputeDensityFromAtomarray:
    """Tests for compute_density_from_atomarray function."""

    def test_returns_density_tensor(self, simple_atom_array: AtomArray, device: torch.device):
        """Test that first output is torch.Tensor."""
        density, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert isinstance(density, torch.Tensor)

    def test_returns_xmap_torch(self, simple_atom_array: AtomArray, device: torch.device):
        """Test that second output is XMap_torch."""
        _, xmap_torch = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert isinstance(xmap_torch, XMap_torch)

    def test_density_shape_matches_grid(self, simple_atom_array: AtomArray, device: torch.device):
        """Test that density tensor shape matches grid dimensions."""
        density, xmap_torch = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        expected_shape = xmap_torch.array.shape
        assert density.shape == expected_shape

    def test_density_is_finite(self, simple_atom_array: AtomArray, device: torch.device):
        """Test that density contains no NaN or Inf values."""
        density, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert torch.isfinite(density).all()

    def test_with_resolution_parameter(self, simple_atom_array: AtomArray, device: torch.device):
        """Test that resolution parameter creates synthetic grid."""
        density, xmap_torch = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert isinstance(density, torch.Tensor)
        assert density.numel() > 0

    def test_with_xmap_parameter(
        self, simple_atom_array: AtomArray, synthetic_grid, device: torch.device
    ):
        """Test that function uses provided XMap."""
        xmap, resolution, padding = synthetic_grid
        density, xmap_torch = compute_density_from_atomarray(
            simple_atom_array, xmap=xmap, em_mode=False, device=device
        )
        assert isinstance(density, torch.Tensor)
        assert density.shape == xmap.array.shape

    def test_both_xmap_and_resolution_raises(
        self, simple_atom_array: AtomArray, synthetic_grid, device: torch.device
    ):
        """Test that providing both xmap and resolution raises ValueError."""
        xmap, resolution, padding = synthetic_grid
        with pytest.raises(ValueError, match="Cannot provide both xmap and resolution"):
            compute_density_from_atomarray(
                simple_atom_array, xmap=xmap, resolution=2.0, em_mode=False, device=device
            )

    def test_neither_xmap_nor_resolution_raises(
        self, simple_atom_array: AtomArray, device: torch.device
    ):
        """Test that providing neither xmap nor resolution raises ValueError."""
        with pytest.raises(ValueError, match="Either xmap or resolution must be provided"):
            compute_density_from_atomarray(simple_atom_array, em_mode=False, device=device)

    def test_xray_mode(self, simple_atom_array: AtomArray, device: torch.device):
        """Test that X-ray mode (em_mode=False) works."""
        density, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert isinstance(density, torch.Tensor)
        assert density.sum() > 0

    def test_electron_mode(self, simple_atom_array: AtomArray, device: torch.device):
        """Test that electron mode (em_mode=True) works."""
        density, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=True, device=device
        )
        assert isinstance(density, torch.Tensor)
        assert density.sum() > 0

    def test_device_parameter(self, simple_atom_array: AtomArray, device: torch.device):
        """Test that function respects device parameter."""
        density, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert density.device == device

    def test_density_has_positive_values(self, simple_atom_array: AtomArray, device: torch.device):
        """Test that density has some positive values."""
        density, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        assert density.sum() > 0

    def test_with_atom_array_single_model(
        self, simple_atom_array_stack: AtomArrayStack, device: torch.device
    ):
        """Test that function works with single model from stack."""
        single_model = simple_atom_array_stack[0]
        density, _ = compute_density_from_atomarray(
            single_model,  # pyright: ignore[reportArgumentType]
            resolution=2.0,
            em_mode=False,
            device=device,
        )
        assert isinstance(density, torch.Tensor)
        assert torch.isfinite(density).all()

    def test_deterministic_output(self, simple_atom_array: AtomArray, device: torch.device):
        """Test that same input produces same output."""
        density1, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        density2, _ = compute_density_from_atomarray(
            simple_atom_array, resolution=2.0, em_mode=False, device=device
        )
        torch.testing.assert_close(density1, density2)

    def test_high_resolution_smaller_voxels(
        self, simple_atom_array: AtomArray, device: torch.device
    ):
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

    def test_zero_occupancy_atoms_filtered(self, device: torch.device):
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

    def test_with_real_structure(
        self, structure_1vme: dict, density_map_1vme: XMap, device: torch.device
    ):
        """Test density computation with real structure."""
        atom_array = structure_1vme["asym_unit"]
        if isinstance(atom_array, AtomArrayStack):
            atom_array = atom_array[0]

        density_no_xmap, _ = compute_density_from_atomarray(
            atom_array,  # pyright: ignore[reportArgumentType]
            resolution=2.0,
            em_mode=False,
            device=device,
        )
        assert isinstance(density_no_xmap, torch.Tensor)
        assert density_no_xmap.sum() > 0
        assert torch.isfinite(density_no_xmap).all()

        density_xmap, _ = compute_density_from_atomarray(
            atom_array,  # pyright: ignore[reportArgumentType]
            xmap=density_map_1vme,  # resolution 1.8 A
            em_mode=False,
            device=device,
        )
        assert isinstance(density_xmap, torch.Tensor)
        assert density_xmap.sum() > 0
        assert torch.isfinite(density_xmap).all()

        assert density_no_xmap.shape != density_xmap.shape  # different resolutions


class TestComputeDensityErrors:
    """Tests for error handling in compute_density_from_atomarray."""

    def test_empty_atom_array(self, device: torch.device):
        """Test behavior with empty AtomArray."""
        atom_array = AtomArray(0)
        atom_array.coord = np.empty((0, 3))

        with pytest.raises((ValueError, RuntimeError, IndexError)):
            compute_density_from_atomarray(atom_array, resolution=2.0, em_mode=False, device=device)
