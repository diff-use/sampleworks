import torch


def trilinear_interpolation_torch(
    density_map: torch.Tensor, points_zyx: torch.Tensor
) -> torch.Tensor:
    """
    Trilinear interpolation on a 3D map.

    Parameters:
    ----------
    density_map: torch.Tensor
        The 3D density map, expected to have shape [Mz, My, Mx],
        where Mz, My, Mx are the grid dimensions along z, y, and x axes.
    points_zyx: torch.Tensor
        A tensor of points at which to interpolate values, with shape
        [..., N_points, 3], where "..." represents any number of leading
        batch dimensions. Each innermost [N_points, 3] slice represents
        a set of points, with each point having (z, y, x) coordinates
        in grid units.
        It is assumed that any necessary periodic boundary conditions
        (modulo operations) have already been applied to these points
        if the map represents a unit cell.

    Returns:
    -------
    torch.Tensor
        A tensor of shape [..., N_points] containing the interpolated values,
        matching the leading dimensions of `points_zyx`.

    Notes:
    -----
    - Assumes integer grid coordinates correspond to voxel corners.
    - For points on a grid line, torch.floor determines the lower indexed voxel.
    - The device and dtype of the output will match `density_map`.
    """
    if not isinstance(density_map, torch.Tensor) or not isinstance(points_zyx, torch.Tensor):
        raise TypeError("Inputs 'density_map' and 'points_zyx' must be torch.Tensors.")
    if density_map.ndim != 3:
        raise ValueError(
            f"Input 'density_map' must be a 3D tensor (shape [Mz, My, Mx]), "
            f"got shape {density_map.shape}."
        )
    if points_zyx.shape[-1] != 3:
        raise ValueError(
            f"The last dimension of 'points_zyx' must be 3 (for z,y,x coordinates), "
            f"got shape {points_zyx.shape}."
        )

    Mz, My, Mx = density_map.shape
    device = density_map.device
    dtype = density_map.dtype

    # Store original shape for reshaping the output
    original_shape = points_zyx.shape
    num_points_total = original_shape[:-1].numel()  # Product of all dimensions except the last one

    # Reshape points_zyx to [N_total_points, 3] for processing
    points_zyx_flat = points_zyx.reshape(num_points_total, 3)
    points_zyx_flat = points_zyx_flat.to(device=device, dtype=dtype)

    z, y, x = points_zyx_flat[:, 0], points_zyx_flat[:, 1], points_zyx_flat[:, 2]

    # Get the integer part of the coordinates
    z0 = torch.floor(z).long()
    y0 = torch.floor(y).long()
    x0 = torch.floor(x).long()

    # Calculate the fractional parts (weights for the '+1' index)
    zd = z - z0.to(dtype)
    yd = y - y0.to(dtype)
    xd = x - x0.to(dtype)

    # Clamp base indices to be within the valid map range [0, Dim-1]
    z0_clamped = torch.clamp(z0, 0, Mz - 1)
    y0_clamped = torch.clamp(y0, 0, My - 1)
    x0_clamped = torch.clamp(x0, 0, Mx - 1)

    # Calculate the 'upper' voxel indices using modulo for periodic boundaries
    z1 = (z0_clamped + 1) % Mz
    y1 = (y0_clamped + 1) % My
    x1 = (x0_clamped + 1) % Mx

    # Gather the density values at the 8 corners of the voxel
    v000 = density_map[z0_clamped, y0_clamped, x0_clamped]
    v001 = density_map[z0_clamped, y0_clamped, x1]
    v010 = density_map[z0_clamped, y1, x0_clamped]
    v011 = density_map[z0_clamped, y1, x1]
    v100 = density_map[z1, y0_clamped, x0_clamped]
    v101 = density_map[z1, y0_clamped, x1]
    v110 = density_map[z1, y1, x0_clamped]
    v111 = density_map[z1, y1, x1]

    # Interpolate along x-axis
    c00 = v000 * (1 - xd) + v001 * xd
    c01 = v010 * (1 - xd) + v011 * xd
    c10 = v100 * (1 - xd) + v101 * xd
    c11 = v110 * (1 - xd) + v111 * xd

    # Interpolate along y-axis
    c0 = c00 * (1 - yd) + c01 * yd
    c1 = c10 * (1 - yd) + c11 * yd

    # Interpolate along z-axis
    interpolated_values_flat = c0 * (1 - zd) + c1 * zd

    # Reshape the output to match the leading dimensions of points_zyx,
    # with the last dimension being the number of points.
    output_shape = list(original_shape[:-1])
    interpolated_values = interpolated_values_flat.reshape(output_shape)

    return interpolated_values


def tricubic_interpolation_torch(
    density_map: torch.Tensor, points_zyx: torch.Tensor
) -> torch.Tensor:
    """
    Tricubic interpolation on a 3D map.

    Parameters:
    ----------
    density_map: torch.Tensor
        The 3D density map, expected to have shape [Mz, My, Mx],
        where Mz, My, Mx are the grid dimensions along z, y, and x axes.
    points_zyx: torch.Tensor
        A tensor of points at which to interpolate values, with shape
        [..., N_points, 3], where "..." represents any number of leading
        batch dimensions. Each innermost [N_points, 3] slice represents
        a set of points, with each point having (z, y, x) coordinates
        in grid units.

    Returns:
    -------
    torch.Tensor
        A tensor of shape [..., N_points] containing the interpolated values,
        matching the leading dimensions of `points_zyx`.
    """
    if not isinstance(density_map, torch.Tensor) or not isinstance(points_zyx, torch.Tensor):
        raise TypeError("Inputs must be torch.Tensors.")
    if density_map.ndim != 3:
        raise ValueError(f"Density map must be 3D, got shape {density_map.shape}.")
    if points_zyx.shape[-1] != 3:
        raise ValueError(f"Last dimension must be 3, got shape {points_zyx.shape}.")

    Mz, My, Mx = density_map.shape
    device = density_map.device
    dtype = density_map.dtype

    original_shape = points_zyx.shape
    num_points_total = original_shape[:-1].numel()

    points_zyx_flat = points_zyx.reshape(num_points_total, 3)
    points_zyx_flat = points_zyx_flat.to(device=device, dtype=dtype)

    z, y, x = points_zyx_flat[:, 0], points_zyx_flat[:, 1], points_zyx_flat[:, 2]

    # Get integer coordinates of the box containing each point
    nz = torch.floor(z).long()
    ny = torch.floor(y).long()
    nx = torch.floor(x).long()

    # Calculate fractional coordinates within the unit cube
    xd = x - nx.to(dtype)
    yd = y - ny.to(dtype)
    zd = z - nz.to(dtype)

    # Get indices for the 64 corners needed (8 corners,
    # each with 8 neighbors for derivatives)
    indices = []
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                indices.append(
                    (
                        torch.remainder(nz + i, Mz),
                        torch.remainder(ny + j, My),
                        torch.remainder(nx + k, Mx),
                    )
                )

    # Function to compute the cubic polynomial coefficients
    def compute_cubic_coefficients(f_m1, f_0, f_1, f_2):
        a0 = f_0
        a1 = 0.5 * (f_1 - f_m1)
        a2 = 0.5 * (-f_2 + 4 * f_1 - 5 * f_0 + 2 * f_m1)
        a3 = 0.5 * (f_2 - 3 * f_1 + 3 * f_0 - f_m1)
        return a0, a1, a2, a3

    # Extract function values at grid points surrounding each query point
    values = {}
    for di in range(-1, 3):
        for dj in range(-1, 3):
            for dk in range(-1, 3):
                idx_z = torch.remainder(nz + di, Mz)
                idx_y = torch.remainder(ny + dj, My)
                idx_x = torch.remainder(nx + dk, Mx)
                values[(di, dj, dk)] = density_map[idx_z, idx_y, idx_x]

    # Perform tricubic interpolation using nested 1D cubic interpolations
    # First interpolate along x for all z,y pairs
    x_interpolated = {}
    for di in range(-1, 3):
        for dj in range(-1, 3):
            f_m1 = values[(di, dj, -1)]
            f_0 = values[(di, dj, 0)]
            f_1 = values[(di, dj, 1)]
            f_2 = values[(di, dj, 2)]

            a0, a1, a2, a3 = compute_cubic_coefficients(f_m1, f_0, f_1, f_2)
            x_interpolated[(di, dj)] = a0 + a1 * xd + a2 * xd**2 + a3 * xd**3

    # Next interpolate along y
    y_interpolated = {}
    for di in range(-1, 3):
        f_m1 = x_interpolated[(di, -1)]
        f_0 = x_interpolated[(di, 0)]
        f_1 = x_interpolated[(di, 1)]
        f_2 = x_interpolated[(di, 2)]

        a0, a1, a2, a3 = compute_cubic_coefficients(f_m1, f_0, f_1, f_2)
        y_interpolated[di] = a0 + a1 * yd + a2 * yd**2 + a3 * yd**3

    # Finally interpolate along z
    f_m1 = y_interpolated[-1]
    f_0 = y_interpolated[0]
    f_1 = y_interpolated[1]
    f_2 = y_interpolated[2]

    a0, a1, a2, a3 = compute_cubic_coefficients(f_m1, f_0, f_1, f_2)
    interpolated_values_flat = a0 + a1 * zd + a2 * zd**2 + a3 * zd**3

    # Reshape output to match input dimensions
    interpolated_values = interpolated_values_flat.reshape(original_shape[:-1])

    return interpolated_values


def compute_derivatives_torch(
    density_map: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute derivatives of the density map using central finite differences.

    Parameters:
    ----------
    density_map: torch.Tensor
        The 3D density map of shape [Mz, My, Mx]

    Returns:
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Tensors containing derivatives in z, y, and x directions
    """
    Mz, My, Mx = density_map.shape

    # Compute derivatives using central differences with periodic boundaries
    dx = (torch.roll(density_map, -1, dims=2) - torch.roll(density_map, 1, dims=2)) / 2
    dy = (torch.roll(density_map, -1, dims=1) - torch.roll(density_map, 1, dims=1)) / 2
    dz = (torch.roll(density_map, -1, dims=0) - torch.roll(density_map, 1, dims=0)) / 2

    return dz, dy, dx
