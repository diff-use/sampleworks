// dilate_points_impl.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <assert.h>

// Forward declaration of CUDA kernels (defined in dilate_points_kernel.cu)
__global__ void dilate_atom_centric_forward_kernel(
    const float *atom_coords_grid,
    const float *atom_occupancies,
    const float *radial_profiles,
    const float r_step,
    const float rmax_cartesian,
    const int *lmax_grid_units,
    const int *grid_dims,
    const float *grid_to_cartesian_matrix,
    float *output_density_grid,
    const int batch_size,
    const int n_symmetry_ops,
    const int N_atoms,
    const int N_radial_points);

__global__ void dilate_atom_centric_backward_kernel(
    const float *grad_output_density_grid,
    const float *atom_coords_grid,
    const float *atom_occupancies,
    const float *radial_profiles,
    const float *radial_profiles_derivatives,
    const float r_step,
    const float rmax_cartesian,
    const int *lmax_grid_units,
    const int *grid_dims,
    const float *grid_to_cartesian_matrix,
    float *grad_atom_coords_grid,
    float *grad_atom_occupancies,
    float *grad_radial_profiles,
    const int batch_size,
    const int n_symmetry_ops,
    const int N_atoms,
    const int N_radial_points);

// Implementation of forward pass
torch::Tensor dilate_atom_centric_forward_cuda(
    const torch::Tensor &atom_coords_grid,
    const torch::Tensor &atom_occupancies,
    const torch::Tensor &radial_profiles,
    const float r_step,
    const float rmax_cartesian,
    const torch::Tensor &lmax_grid_units,
    const torch::Tensor &grid_dims,
    const torch::Tensor &grid_to_cartesian_matrix)
{
    // Ensure all inputs are on the same device and are properly allocated
    auto device = atom_coords_grid.device();
    at::DeviceGuard device_guard(device);

    // Create fresh tensor copies with correct memory layout
    auto coords = atom_coords_grid.to(device, atom_coords_grid.dtype(), true, true);
    auto occupancies = atom_occupancies.to(device, atom_occupancies.dtype(), true, true);
    auto profiles = radial_profiles.to(device, radial_profiles.dtype(), true, true);
    auto lmax = lmax_grid_units.to(device, torch::kInt32, true, true);
    auto dims = grid_dims.to(device, torch::kInt32, true, true);
    auto matrix = grid_to_cartesian_matrix.to(device, grid_to_cartesian_matrix.dtype(), true, true);

    // Explicitly get sizes to avoid any tensor access during kernel execution
    const int batch_size = coords.size(0);
    const int n_symmetry_ops = coords.size(1);
    const int N_atoms = coords.size(2);
    const int N_radial_points = profiles.size(2);
    const int Dz = dims.index({0}).item<int>();
    const int Dy = dims.index({1}).item<int>();
    const int Dx = dims.index({2}).item<int>();

    auto output = torch::zeros({batch_size, Dz, Dy, Dx},
                             torch::TensorOptions()
                             .dtype(coords.dtype())
                             .device(device)
                             .layout(torch::kStrided));

    // Get raw pointers AFTER all tensor operations
    float* coords_ptr = coords.data_ptr<float>();
    float* occupancies_ptr = occupancies.data_ptr<float>();
    float* profiles_ptr = profiles.data_ptr<float>();
    int* lmax_ptr = lmax.data_ptr<int>();
    int* dims_ptr = dims.data_ptr<int>();
    float* matrix_ptr = matrix.data_ptr<float>();
    float* output_ptr = output.data_ptr<float>();

    // Force synchronization to ensure all memory operations complete
    cudaDeviceSynchronize();

    // Debug printf
    // printf("CUDA Memory: coords=%p, occ=%p, profiles=%p, output=%p\n",
    //        coords_ptr, occupancies_ptr, profiles_ptr, output_ptr);
    // printf("Dims: batch=%d, sym_ops=%d, atoms=%d, radial=%d, grid=[%d,%d,%d]\n",
    //        batch_size, n_symmetry_ops, N_atoms, N_radial_points, Dz, Dy, Dx);

    const int threads_per_block = 256;
    const int total_atoms = batch_size * n_symmetry_ops * N_atoms;
    const int blocks = (total_atoms + threads_per_block - 1) / threads_per_block;

    // Create a dedicated stream for this operation
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    dilate_atom_centric_forward_kernel<<<blocks, threads_per_block, 0, stream>>>(
        coords_ptr,
        occupancies_ptr,
        profiles_ptr,
        r_step,
        rmax_cartesian,
        lmax_ptr,
        dims_ptr,
        matrix_ptr,
        output_ptr,
        batch_size,
        n_symmetry_ops,
        N_atoms,
        N_radial_points);

    // Wait for kernel completion before returning
    cudaStreamSynchronize(stream);
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(error));
        TORCH_CHECK(false, "CUDA error in forward kernel: ", cudaGetErrorString(error));
    }

    cudaStreamDestroy(stream);

    return output;
}

// Implementation of backward pass
std::vector<torch::Tensor> dilate_atom_centric_backward_cuda(
    const torch::Tensor &grad_output,
    const torch::Tensor &atom_coords_grid,
    const torch::Tensor &atom_occupancies,
    const torch::Tensor &radial_profiles,
    const torch::Tensor &radial_profiles_derivatives,
    const float r_step,
    const float rmax_cartesian,
    const torch::Tensor &lmax_grid_units,
    const torch::Tensor &grid_dims,
    const torch::Tensor &grid_to_cartesian_matrix)
{

    // Input validation
    TORCH_CHECK(atom_coords_grid.dim() == 4,
                "atom_coords_grid should be 4D [batch, n_symmetry_ops, n_atoms, 3]");
    TORCH_CHECK(atom_occupancies.dim() == 2,
                "atom_occupancies should be 2D [batch, n_atoms]");
    TORCH_CHECK(radial_profiles.dim() == 3,
                "radial_profiles should be 3D [batch, n_atoms, n_radial_points]");
    TORCH_CHECK(radial_profiles_derivatives.dim() == 3,
                "radial_profiles_derivatives should be 3D [batch, n_atoms, n_radial_points]");

    auto batch_size = atom_coords_grid.size(0);
    auto n_symmetry_ops = atom_coords_grid.size(1);
    auto N_atoms = atom_coords_grid.size(2);
    auto N_radial_points = radial_profiles.size(2);

    // Verify tensor shapes
    auto Dz = grid_dims.index({0}).item<int>();
    auto Dy = grid_dims.index({1}).item<int>();
    auto Dx = grid_dims.index({2}).item<int>();

    TORCH_CHECK(grad_output.sizes() == torch::IntArrayRef({batch_size, Dz, Dy, Dx}),
                "grad_output shape mismatch, expected [", batch_size, ", ", Dz, ", ", Dy, ", ", Dx, "]");

    // Initialize gradient tensors
    auto options = torch::TensorOptions()
                       .dtype(atom_coords_grid.dtype())
                       .device(atom_coords_grid.device());

    auto grad_atom_coords_grid = torch::zeros_like(atom_coords_grid);
    auto grad_atom_occupancies = torch::zeros_like(atom_occupancies);
    auto grad_radial_profiles = torch::zeros_like(radial_profiles);

    // Configure CUDA kernel launch
    const int threads_per_block = 256;
    const int total_atoms = batch_size * n_symmetry_ops * N_atoms;
    const int blocks = (total_atoms + threads_per_block - 1) / threads_per_block;

    // Launch kernel
    dilate_atom_centric_backward_kernel<<<blocks, threads_per_block>>>(
        grad_output.data_ptr<float>(),
        atom_coords_grid.data_ptr<float>(),
        atom_occupancies.data_ptr<float>(),
        radial_profiles.data_ptr<float>(),
        radial_profiles_derivatives.data_ptr<float>(),
        r_step,
        rmax_cartesian,
        lmax_grid_units.data_ptr<int>(),
        grid_dims.data_ptr<int>(),
        grid_to_cartesian_matrix.data_ptr<float>(),
        grad_atom_coords_grid.data_ptr<float>(),
        grad_atom_occupancies.data_ptr<float>(),
        grad_radial_profiles.data_ptr<float>(),
        batch_size,
        n_symmetry_ops,
        N_atoms,
        N_radial_points);

    // Check for CUDA errors
    cudaError_t error = cudaGetLastError();
    TORCH_CHECK(error == cudaSuccess,
                "CUDA error in backward kernel: ", cudaGetErrorString(error));

    return {grad_atom_coords_grid, grad_atom_occupancies, grad_radial_profiles};
}
