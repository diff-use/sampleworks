# ruff: noqa
# pyright: ignore
import torch

from . import CUDA_AVAILABLE, dilate_points_cuda


class DilateAtomCentricCUDA(torch.autograd.Function):
    """Custom CUDA-accelerated atom-centric density dilation operation.

    This implementation provides efficient forward and backward passes for computing
    density maps from atomic positions and radial profiles.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        atom_coords_grid: torch.Tensor,  # [batch_size, N_atoms, 3] grid units
        atom_occupancies: torch.Tensor,  # [batch_size, N_atoms]
        radial_profiles: torch.Tensor,  # [batch_size, N_atoms, N_radial_points]
        radial_profiles_derivatives: torch.Tensor,  # [batch_size, N_atoms, N_radial_points]
        r_step: float,  # Scalar step size for radial_profiles
        rmax_cartesian: float,  # Max radius in Cartesian
        lmax_grid_units: torch.Tensor,  # [3] tensor with [lx, ly, lz]
        grid_dims: torch.Tensor,  # [3] tensor with [Dz, Dy, Dx]
        grid_to_cartesian_matrix: torch.Tensor,  # [3, 3] transformation matrix
    ) -> torch.Tensor:
        """Forward pass: computes density grid from atomic positions and radial profiles.

        Parameters
        ----------
        atom_coords_grid : torch.Tensor
            Atomic coordinates in grid units, shape [batch_size, symmetry_ops, N_atoms, 3]
        atom_occupancies : torch.Tensor
            Atomic occupancies, shape [batch_size, N_atoms]
        radial_profiles : torch.Tensor
            Pre-calculated radial density values P(r), shape [batch_size, N_atoms, N_radial_points]
        radial_profiles_derivatives : torch.Tensor
            Pre-calculated derivatives of radial density P'(r), shape [batch_size, N_atoms, N_radial_points]
        r_step : float
            Step size for radial_profiles sampling
        rmax_cartesian : float
            Maximum radius for an atom's influence in Cartesian space
        lmax_grid_units : torch.Tensor
            Maximum extent in grid units along each axis, shape [3]
        grid_dims : torch.Tensor
            Dimensions of output grid [Dz, Dy, Dx], shape [3]
        grid_to_cartesian_matrix : torch.Tensor
            Transformation matrix from grid to Cartesian coordinates, shape [3, 3]

        Returns
        -------
        torch.Tensor
            Output density grid, shape [batch_size, Dz, Dy, Dx]
        """

        # Clone inputs and store the original tensors in context
        ctx.save_for_backward(
            atom_coords_grid,
            atom_occupancies,
            radial_profiles,
            radial_profiles_derivatives,
            lmax_grid_units,
            grid_dims,
            grid_to_cartesian_matrix,
        )
        ctx.r_step = r_step
        ctx.rmax_cartesian = rmax_cartesian

        atom_coords_grid = atom_coords_grid.contiguous()
        atom_occupancies = atom_occupancies.contiguous()
        radial_profiles = radial_profiles.contiguous()
        lmax_grid_units = lmax_grid_units.to(torch.int32).contiguous()
        grid_dims = grid_dims.to(torch.int32).contiguous()
        grid_to_cartesian_matrix = grid_to_cartesian_matrix.contiguous()

        if CUDA_AVAILABLE:
            torch.cuda.synchronize()

            output_density_grid = dilate_points_cuda.forward(
                atom_coords_grid,
                atom_occupancies,
                radial_profiles,
                r_step,
                rmax_cartesian,
                lmax_grid_units,
                grid_dims,
                grid_to_cartesian_matrix,
            )

            torch.cuda.synchronize()
        else:
            raise RuntimeError("CUDA is not available.")

        return output_density_grid

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        """Backward pass: computes gradients with respect to inputs.

        Parameters
        ----------
        grad_output : torch.Tensor
            Gradient of loss with respect to output density grid

        Returns
        -------
        Tuple[Optional[torch.Tensor], ...]
            Gradients with respect to each input from the forward pass
        """
        (
            atom_coords_grid,
            atom_occupancies,
            radial_profiles,
            radial_profiles_derivatives,
            lmax_grid_units,
            grid_dims,
            grid_to_cartesian_matrix,
        ) = ctx.saved_tensors

        r_step = ctx.r_step
        rmax_cartesian = ctx.rmax_cartesian

        grad_output = grad_output.contiguous()

        if CUDA_AVAILABLE:
            grad_atom_coords_grid, grad_atom_occupancies, grad_radial_profiles = (
                dilate_points_cuda.backward(
                    grad_output,
                    atom_coords_grid,
                    atom_occupancies,
                    radial_profiles,
                    radial_profiles_derivatives,
                    r_step,
                    rmax_cartesian,
                    lmax_grid_units,
                    grid_dims,
                    grid_to_cartesian_matrix,
                )
            )
        else:
            raise RuntimeError("Backward failed: CUDA is not available.")

        return (
            grad_atom_coords_grid,  # atom_coords_grid
            grad_atom_occupancies,  # atom_occupancies
            grad_radial_profiles,  # radial_profiles
            None,  # radial_profiles_derivatives
            None,  # r_step
            None,  # rmax_cartesian
            None,  # lmax_grid_units
            None,  # grid_dims
            None,  # grid_to_cartesian_matrix
        )


def dilate_atom_centric(
    atom_coords_grid: torch.Tensor,
    atom_occupancies: torch.Tensor,
    radial_profiles: torch.Tensor,
    radial_profiles_derivatives: torch.Tensor,
    r_step: float,
    rmax_cartesian: float,
    lmax_grid_units: torch.Tensor,
    grid_dims: torch.Tensor | tuple,
    grid_to_cartesian_matrix: torch.Tensor,
) -> torch.Tensor:
    device = atom_coords_grid.device
    dtype = atom_coords_grid.dtype

    atom_coords_grid = (
        atom_coords_grid.clone().to(device=device, dtype=dtype).contiguous()
    )
    atom_occupancies = (
        atom_occupancies.clone().to(device=device, dtype=dtype).contiguous()
    )
    radial_profiles = (
        radial_profiles.clone().to(device=device, dtype=dtype).contiguous()
    )
    radial_profiles_derivatives = (
        radial_profiles_derivatives.clone().to(device=device, dtype=dtype).contiguous()
    )

    if isinstance(lmax_grid_units, torch.Tensor):
        lmax_grid_units = (
            torch.ceil(lmax_grid_units)
            .clone()
            .to(dtype=torch.int32, device=device)
            .contiguous()
        )
    else:
        lmax_grid_units = torch.ceil(
            torch.tensor(lmax_grid_units, dtype=torch.int32, device=device)
        ).contiguous()

    if isinstance(grid_dims, torch.Tensor):
        grid_dims = grid_dims.clone().to(dtype=torch.int32, device=device).contiguous()
    else:
        grid_dims = torch.tensor(
            grid_dims, dtype=torch.int32, device=device
        ).contiguous()

    grid_to_cartesian_matrix = (
        grid_to_cartesian_matrix.clone().to(device=device, dtype=dtype).contiguous()
    )

    # Debug validation
    # print(f"Final check - coords: is_contiguous={atom_coords_grid.is_contiguous()}, "
    #       f"is_cuda={atom_coords_grid.is_cuda}, storage_offset={atom_coords_grid.storage_offset()}, "
    #       f"stride={atom_coords_grid.stride()}")

    # Force synchronize before kernel call to ensure memory is fully moved to device
    torch.cuda.synchronize(device)

    return DilateAtomCentricCUDA.apply(
        atom_coords_grid,  # [B, sym_ops, N, 3]
        atom_occupancies,  # [B, N]
        radial_profiles,  # [B, N, R]
        radial_profiles_derivatives,  # [B, N, R]
        r_step,
        rmax_cartesian,
        lmax_grid_units,  # [3]
        grid_dims,  # [3]
        grid_to_cartesian_matrix,  # [3, 3]
    )
