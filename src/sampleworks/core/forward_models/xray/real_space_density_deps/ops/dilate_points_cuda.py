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
        atom_coords_grid: torch.Tensor,  # [batch_size, symmetry_ops, N_atoms, 3] grid units
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
        atom_coords_grid: torch.Tensor
            Atomic coordinates in grid units, shape [batch_size, symmetry_ops, N_atoms, 3]
        atom_occupancies: torch.Tensor
            Atomic occupancies, shape [batch_size, N_atoms]
        radial_profiles: torch.Tensor
            Pre-calculated radial density values P(r), shape [batch_size, N_atoms, N_radial_points]
        radial_profiles_derivatives: torch.Tensor
            Pre-calculated derivatives of radial density P'(r), shape [batch_size, N_atoms, N_radial_points]
        r_step: float
            Step size for radial_profiles sampling
        rmax_cartesian: float
            Maximum radius for an atom's influence in Cartesian space
        lmax_grid_units: torch.Tensor
            Maximum extent in grid units along each axis, shape [3]
        grid_dims: torch.Tensor
            Dimensions of output grid [Dz, Dy, Dx], shape [3]
        grid_to_cartesian_matrix: torch.Tensor
            Transformation matrix from grid to Cartesian coordinates, shape [3, 3]

        Returns
        -------
        torch.Tensor
            Output density grid, shape [batch_size, Dz, Dy, Dx]
        """
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
    def setup_context(ctx, inputs, output):
        """Setup context for backward pass.

        This method is called after forward() and saves information needed
        for the backward pass. Required for functorch compatibility.

        Parameters
        ----------
        ctx: Context object
            Context to save information for backward pass
        inputs: Tuple
            All inputs passed to forward() in order
        output: torch.Tensor
            Output from forward()
        """
        (
            atom_coords_grid,
            atom_occupancies,
            radial_profiles,
            radial_profiles_derivatives,
            r_step,
            rmax_cartesian,
            lmax_grid_units,
            grid_dims,
            grid_to_cartesian_matrix,
        ) = inputs

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

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor | None, ...]:
        """Backward pass: computes gradients with respect to inputs.

        Parameters
        ----------
        grad_output: torch.Tensor
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

    @staticmethod
    def vmap(info, in_dims, *args):
        """Vectorize dilate operation across vmap batch dimension.

        Handles case where vmap adds an extra batch dimension to inputs that
        already have a batch dimension. Uses reshape-and-merge strategy for
        efficiency: merges vmap batch with existing batch, runs CUDA kernel
        once, then reshapes output.

        Parameters
        ----------
        info : object
            vmap metadata (unused)
        in_dims : tuple
            Batching dimension for each input. Expected pattern:
            (0, 0, 0, 0, None, None, None, None, None) meaning first 4
            tensor inputs are batched at dim 0, scalars and grid params
            are not batched.
        *args : tuple
            All input arguments to forward() in order

        Returns
        -------
        tuple[torch.Tensor, int]
            - output: Batched density grid
            - out_dim: 0 (indicating vmap batch is at position 0)
        """
        (
            atom_coords_grid,
            atom_occupancies,
            radial_profiles,
            radial_profiles_derivatives,
            r_step,
            rmax_cartesian,
            lmax_grid_units,
            grid_dims,
            grid_to_cartesian_matrix,
        ) = args

        if in_dims[0] is None:
            output = DilateAtomCentricCUDA.apply(*args)
            return output, None

        # coords, occupancies, radial_profiles, radial_profiles_derivatives can be vmapped
        if in_dims[0] == 0:  # atom_coords_grid is vmapped
            # Shape: [vmap_B, batch_B, sym_ops, N_atoms, 3]
            vmap_B = atom_coords_grid.shape[0]
            batch_B = atom_coords_grid.shape[1]
            sym_ops = atom_coords_grid.shape[2]
            N_atoms = atom_coords_grid.shape[3]

            coords_merged = atom_coords_grid.reshape(
                vmap_B * batch_B, sym_ops, N_atoms, 3
            ).contiguous()
        else:
            # Shape: [batch_B, sym_ops, N_atoms, 3]
            vmap_B = 1
            batch_B = atom_coords_grid.shape[0]
            sym_ops = atom_coords_grid.shape[1]
            N_atoms = atom_coords_grid.shape[2]

            coords_merged = atom_coords_grid

        if in_dims[1] == 0:  # atom_occupancies is vmapped
            # Shape: [vmap_B, batch_B, N_atoms]
            occs_merged = atom_occupancies.reshape(
                vmap_B * batch_B, N_atoms
            ).contiguous()
        else:
            # Shape: [batch_B, N_atoms], need to expand for vmap
            occs_merged = atom_occupancies.repeat_interleave(vmap_B, dim=0).contiguous()

        N_radial = radial_profiles.shape[-1]
        if in_dims[2] == 0:  # radial_profiles is vmapped
            # Shape: [vmap_B, batch_B, N_atoms, N_radial]
            profiles_merged = radial_profiles.reshape(
                vmap_B * batch_B, N_atoms, N_radial
            ).contiguous()
        else:
            # Shape: [batch_B, N_atoms, N_radial], need to expand for vmap
            profiles_merged = radial_profiles.repeat_interleave(
                vmap_B, dim=0
            ).contiguous()

        if in_dims[3] == 0:  # radial_profiles_derivatives is vmapped
            # Shape: [vmap_B, batch_B, N_atoms, N_radial]
            derivs_merged = radial_profiles_derivatives.reshape(
                vmap_B * batch_B, N_atoms, N_radial
            ).contiguous()
        else:
            # Shape: [batch_B, N_atoms, N_radial], need to expand for vmap
            derivs_merged = radial_profiles_derivatives.repeat_interleave(
                vmap_B, dim=0
            ).contiguous()

        output_merged = DilateAtomCentricCUDA.apply(
            coords_merged,
            occs_merged,
            profiles_merged,
            derivs_merged,
            r_step,
            rmax_cartesian,
            lmax_grid_units,
            grid_dims,
            grid_to_cartesian_matrix,
        )

        if in_dims[0] == 0:
            Dz, Dy, Dx = output_merged.shape[-3:]
            output = output_merged.reshape(vmap_B, batch_B, Dz, Dy, Dx)
            return output, 0
        else:
            return output_merged, None


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
