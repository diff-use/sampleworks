// dilate_points_ext.cpp
#include <torch/extension.h>
#include <vector>

// Forward declarations
torch::Tensor dilate_atom_centric_forward_cuda(
    const torch::Tensor& atom_coords_grid,
    const torch::Tensor& atom_occupancies,
    const torch::Tensor& radial_profiles,
    const float r_step,
    const float rmax_cartesian,
    const torch::Tensor& lmax_grid_units,
    const torch::Tensor& grid_dims,
    const torch::Tensor& grid_to_cartesian_matrix);

std::vector<torch::Tensor> dilate_atom_centric_backward_cuda(
    const torch::Tensor& grad_output,
    const torch::Tensor& atom_coords_grid,
    const torch::Tensor& atom_occupancies,
    const torch::Tensor& radial_profiles,
    const torch::Tensor& radial_profiles_derivatives,
    const float r_step,
    const float rmax_cartesian,
    const torch::Tensor& lmax_grid_units,
    const torch::Tensor& grid_dims,
    const torch::Tensor& grid_to_cartesian_matrix);

// Wrapper functions
torch::Tensor dilate_atom_centric_forward(
    const torch::Tensor& atom_coords_grid,
    const torch::Tensor& atom_occupancies,
    const torch::Tensor& radial_profiles,
    const float r_step,
    const float rmax_cartesian,
    const torch::Tensor& lmax_grid_units,
    const torch::Tensor& grid_dims,
    const torch::Tensor& grid_to_cartesian_matrix) {

    // Basic input validation
    TORCH_CHECK(atom_coords_grid.dim() == 4,
                "atom_coords_grid must be 4D tensor [batch, n_symmetry_ops, n_atoms, 3]");
    TORCH_CHECK(atom_coords_grid.size(3) == 3,
                "atom_coords_grid last dimension must be 3 (coordinates)");

    // Call CUDA implementation
    return dilate_atom_centric_forward_cuda(
        atom_coords_grid,
        atom_occupancies,
        radial_profiles,
        r_step,
        rmax_cartesian,
        lmax_grid_units,
        grid_dims,
        grid_to_cartesian_matrix);
}

std::vector<torch::Tensor> dilate_atom_centric_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& atom_coords_grid,
    const torch::Tensor& atom_occupancies,
    const torch::Tensor& radial_profiles,
    const torch::Tensor& radial_profiles_derivatives,
    const float r_step,
    const float rmax_cartesian,
    const torch::Tensor& lmax_grid_units,
    const torch::Tensor& grid_dims,
    const torch::Tensor& grid_to_cartesian_matrix) {

    // Basic input validation
    TORCH_CHECK(atom_coords_grid.dim() == 4,
                "atom_coords_grid must be 4D tensor [batch, n_symmetry_ops, n_atoms, 3]");
    TORCH_CHECK(atom_occupancies.dim() == 2,
                "atom_occupancies must be 2D tensor [batch, n_atoms]");
    TORCH_CHECK(radial_profiles.dim() == 3,
                "radial_profiles must be 3D tensor [batch, n_atoms, n_radial_points]");

    // Call CUDA implementation
    return dilate_atom_centric_backward_cuda(
        grad_output,
        atom_coords_grid,
        atom_occupancies,
        radial_profiles,
        radial_profiles_derivatives,
        r_step,
        rmax_cartesian,
        lmax_grid_units,
        grid_dims,
        grid_to_cartesian_matrix);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &dilate_atom_centric_forward, "Atom Centric Dilate Forward");
    m.def("backward", &dilate_atom_centric_backward, "Atom Centric Dilate Backward");
}
