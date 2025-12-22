# pyright: ignore

import io
import logging
import os.path
from copy import copy
from numbers import Real
from struct import pack as _pack, unpack as _unpack
from sys import byteorder as _BYTEORDER

import numpy as np
from scipy.ndimage import map_coordinates

from .spacegroups import GetSpaceGroup
from .unitcell import UnitCell


logger = logging.getLogger(__name__)


def extend_to_p1(grid, offset, R_matrix, t_vector, grid_shape_out, out):
    """
    Applies a crystallographic symmetry operation to each point in the 3D array `grid`
    and writes the value into `out` at the transformed coordinate (wrapped by modulo
    grid_shape_out).

    Parameters:
      grid: 3D NumPy array of floats.
        Input density grid with shape (nz, ny, nx)
      offset: Sequence of 3 integers [offset_z, offset_y, offset_x].
        Grid offset in grid coordinates
      R_matrix: 3x3 NumPy array of floats.
        Rotation matrix component of symmetry operation
      t_vector: 1D NumPy array of 3 floats.
        Translation vector component of symmetry operation (in fractional coordinates)
      grid_shape_out: Sequence of 3 integers [nz_out, ny_out, nx_out].
        Shape of output grid for coordinate wrapping
      out: 3D NumPy array of floats.
        Output array where transformed values are written
    """
    # Get grid shape (assumed to be (nz, ny, nx))
    nz, ny, nx = grid.shape
    nz_out, ny_out, nx_out = grid_shape_out

    # Generate index arrays for each coordinate (shape: (nz, ny, nx))
    z_idx, y_idx, x_idx = np.indices((nz, ny, nx))

    # Apply offsets to grid indices (convert to fractional coordinates)
    # Note: offset ordering should match grid ordering (z, y, x)
    grid_z = z_idx + offset[0]  # offset[0] is z offset
    grid_y = y_idx + offset[1]  # offset[1] is y offset
    grid_x = x_idx + offset[2]  # offset[2] is x offset

    # Stack coordinates for matrix multiplication (z, y, x order to match grid indexing)
    coords = np.stack([grid_z, grid_y, grid_x], axis=-1)  # Shape: (nz, ny, nx, 3)
    coords_flat = coords.reshape(-1, 3)  # Shape: (nz*ny*nx, 3)

    # Apply symmetry operation: R @ coords + t
    # Note: R_matrix operates on (x, y, z) coordinates, so we need to reorder
    coords_xyz = coords_flat[:, [2, 1, 0]]  # Convert to (x, y, z) order
    transformed_xyz = (R_matrix @ coords_xyz.T).T + t_vector * np.array([nx_out, ny_out, nz_out])
    transformed_zyx = transformed_xyz[:, [2, 1, 0]]  # Convert back to (z, y, x) order

    # Convert to integer grid coordinates and apply modulo wrapping
    out_coords = np.round(transformed_zyx).astype(int)
    out_z = np.mod(out_coords[:, 0], nz_out)
    out_y = np.mod(out_coords[:, 1], ny_out)
    out_x = np.mod(out_coords[:, 2], nx_out)

    # Reshape back to grid shape for indexing
    out_z = out_z.reshape(nz, ny, nx)
    out_y = out_y.reshape(nz, ny, nx)
    out_x = out_x.reshape(nz, ny, nx)

    # Use advanced indexing to assign grid values to the corresponding positions in out
    out[out_z, out_y, out_x] = grid


class GridParameters:
    def __init__(self, voxelspacing=(1, 1, 1), offset=(0, 0, 0)):
        if isinstance(voxelspacing, Real):
            voxelspacing = [voxelspacing] * 3
        self.voxelspacing = np.asarray(voxelspacing, np.float64)
        self.offset = np.asarray(offset, np.int32)

    def copy(self):
        return GridParameters(self.voxelspacing.copy(), self.offset.copy())


class Resolution:
    def __init__(self, high=None, low=None):
        self.high = high
        self.low = low

    def copy(self):
        return Resolution(self.high, self.low)


class _BaseVolume:
    def __init__(self, array, grid_parameters=None, origin=(0, 0, 0)):
        self.array = array
        if grid_parameters is None:
            grid_parameters = GridParameters()
        self.grid_parameters = grid_parameters
        self.origin = np.asarray(origin, np.float64)

    @property
    def shape(self):
        return self.array.shape

    @property
    def offset(self):
        return self.grid_parameters.offset

    @property
    def voxelspacing(self):
        return self.grid_parameters.voxelspacing

    def tofile(self, fid, fmt=None):
        if fmt is None:
            fmt = os.path.splitext(fid)[-1][1:]
        if fmt in ("ccp4", "map", "mrc"):
            to_mrc(fid, self)
        else:
            raise ValueError("Format is not supported.")


class EMMap(_BaseVolume):
    """A non-periodic volume. Has no notion of a unit cell or space group."""

    def __init__(self, array, grid_parameters=None, origin=(0, 0, 0)):
        super().__init__(array, grid_parameters, origin)

    @classmethod
    def fromfile(cls, fid, fmt=None):
        p = parse_volume(fid)
        density = p.density
        grid_parameters = GridParameters(p.voxelspacing)
        origin = p.origin
        return cls(density, grid_parameters=grid_parameters, origin=origin)

    @classmethod
    def zeros(cls, shape, grid_parameters=None, origin=None):
        array = np.zeros(shape, dtype=np.float64)
        return cls(array, grid_parameters, origin)

    @classmethod
    def zeros_like(cls, volume):
        array = np.zeros_like(volume.array)
        return cls(array, volume.grid_parameters, volume.origin)

    def copy(self):
        return EMMap(
            self.array.copy(),
            grid_parameters=self.grid_parameters.copy(),
            origin=self.origin.copy(),
        )

    def interpolate(self, xyz, order=1):
        # Transform xyz to grid coor.
        grid_coor = xyz - self.origin
        grid_coor /= self.grid_parameters.voxelspacing
        values = map_coordinates(self.array, grid_coor.T[::-1], order=order)
        return values

    def extract(self, xyz, padding=3):
        """Extract a subregion of the EM map around the given coordinates.

        Args:
            xyz (np.ndarray): Cartesian coordinates with shape (n_points, 3)
            padding (float): Padding in Angstroms to add around the coordinates

        Returns:
            EMMap: New EMMap containing the extracted region
        """
        # Convert Cartesian coordinates to grid coordinates
        grid_coor = (
            xyz - self.origin
        ) / self.voxelspacing  # Shape: (n_points, 3) in (x, y, z) order

        # Calculate padding in grid units
        grid_padding = padding / self.voxelspacing  # Shape: (3,) in (x, y, z) order

        # Find bounding box in grid coordinates (x, y, z order)
        lb_xyz = grid_coor.min(axis=0) - grid_padding
        ru_xyz = grid_coor.max(axis=0) + grid_padding
        lb_xyz = np.floor(lb_xyz).astype(int)
        ru_xyz = np.ceil(ru_xyz).astype(int)

        # Clamp bounds to array limits (non-periodic, so no wrapping)
        lb_xyz = np.maximum(lb_xyz, 0)
        ru_xyz = np.minimum(
            ru_xyz, np.array(self.shape[::-1])
        )  # self.shape is (nz, ny, nx), so reverse to (nx, ny, nz)

        # Convert to (z, y, x) order for array slicing
        lb_zyx = lb_xyz[::-1]
        ru_zyx = ru_xyz[::-1]

        # Extract the array region using proper slicing
        array = self.array[
            lb_zyx[0] : ru_zyx[0], lb_zyx[1] : ru_zyx[1], lb_zyx[2] : ru_zyx[2]
        ].copy()

        # Update grid parameters and origin
        grid_parameters = GridParameters(self.voxelspacing)
        new_origin = self.origin + lb_xyz * self.voxelspacing  # Keep in (x, y, z) order

        return EMMap(array, grid_parameters=grid_parameters, origin=new_origin)


class XMap(_BaseVolume):
    """A periodic volume with a unit cell and space group."""

    def __init__(
        self,
        array,
        grid_parameters=None,
        unit_cell=None,
        resolution=None,
        hkl=None,
        origin=None,
    ):
        super().__init__(array, grid_parameters)
        self.unit_cell = unit_cell
        self.hkl = hkl
        self.resolution = resolution
        self.cutoff_dict = {}
        if origin is None:
            self.origin = np.zeros(3, np.float64)
        else:
            self.origin = np.asarray(origin)

    @classmethod
    def fromfile(cls, fname, fmt=None, resolution=None, label="FWT,PHWT"):
        if fmt is None:
            fmt = os.path.splitext(fname)[1]
        if fmt in (".ccp4", ".mrc", ".map"):
            if resolution is None:
                raise ValueError(
                    f"{fname} is a CCP4/MRC/MAP file. Please provide a resolution (use "
                    "the '-r'/'--resolution' flag)."
                )
            parser = parse_volume(fname, fmt=fmt)
            a, b, c = parser.abc
            alpha, beta, gamma = parser.angles
            spacegroup = parser.spacegroup
            if spacegroup == 0:
                raise RuntimeError(
                    f"File {fname} is 2D image or image stack. Please convert to a 3D map."
                )
            unit_cell = UnitCell(a, b, c, alpha, beta, gamma, spacegroup)
            offset = parser.offset
            array = parser.density
            voxelspacing = parser.voxelspacing
            grid_parameters = GridParameters(voxelspacing, offset)
            resolution = Resolution(high=resolution)
            origin = parser.origin
            xmap = cls(
                array,
                grid_parameters,
                unit_cell=unit_cell,
                resolution=resolution,
                origin=origin,
            )
        elif fmt == ".mtz":
            from .mtzfile import MTZFile
            from .transformer import SFTransformer

            mtz = MTZFile(fname)
            hkl = np.asarray(list(zip(mtz["H"], mtz["K"], mtz["L"])), np.int32)
            try:
                crystal = mtz["HKL_base"]
            except KeyError:
                crystal = mtz.crystals[0]
            uc_par = [getattr(crystal, attr) for attr in "a b c alpha beta gamma".split()]
            unit_cell = UnitCell(*uc_par)
            space_group = GetSpaceGroup(mtz.ispg)
            # symops = [SymOpFromString(string) for string in mtz.symops]
            # space_group = SpaceGroup(
            #    number=mtz.symi['ispg'],
            #    num_sym_equiv=mtz.symi['nsym'],
            #    num_primitive_sym_equiv=mtz.symi['nsymp'],
            #    short_name=mtz.symi['spgname'],
            #    point_group_name=mtz.symi['pgname'],
            #    crystal_system=mtz.symi['symtyp'],
            #    pdb_name=mtz.symi['spgname'],
            #    symop_list=symops,
            # )
            unit_cell.space_group = space_group
            f_column, phi_column = label.split(",")
            try:
                f = mtz[f_column]
                phi = mtz[phi_column]
            except KeyError:
                raise KeyError("Could not find columns in MTZ file.")

            t = SFTransformer(hkl, f, phi, unit_cell)
            grid = t()
            abc = [getattr(unit_cell, x) for x in "a b c".split()]
            voxelspacing = [x / n for x, n in zip(abc, grid.shape[::-1])]
            grid_parameters = GridParameters(voxelspacing)
            low = 1 / np.sqrt(mtz.resmin)
            high = 1 / np.sqrt(mtz.resmax)
            resolution = Resolution(high=high, low=low)
            xmap = cls(
                grid,
                grid_parameters,
                unit_cell=unit_cell,
                resolution=resolution,
                hkl=hkl,
            )
        else:
            raise RuntimeError("File format not recognized.")
        return xmap

    @classmethod
    def zeros_like(cls, xmap):
        array = np.zeros_like(xmap.array)
        try:
            uc = xmap.unit_cell.copy()
        except AttributeError:
            uc = None
        hkl = copy(xmap.hkl)
        return cls(
            array,
            grid_parameters=xmap.grid_parameters.copy(),
            unit_cell=uc,
            hkl=hkl,
            resolution=xmap.resolution.copy(),
            origin=xmap.origin.copy(),
        )

    def asymmetric_unit_cell(self):
        raise NotImplementedError

    @property
    def unit_cell_shape(self):
        shape = np.round(self.unit_cell.abc / self.grid_parameters.voxelspacing).astype(int)
        return shape

    def canonical_unit_cell(self):
        # Calculate the canonical unit cell shape (grid points along each axis)
        shape = np.round(self.unit_cell.abc / self.grid_parameters.voxelspacing).astype(int)[
            ::-1
        ]  # Reverse to get (nz, ny, nx) ordering

        # Create output array and XMap
        array = np.zeros(shape, np.float64)
        grid_parameters = GridParameters(self.voxelspacing)
        out = XMap(
            array,
            grid_parameters=grid_parameters,
            unit_cell=self.unit_cell,
            hkl=self.hkl,
            resolution=self.resolution,
        )

        # Get grid offset (convert to z, y, x ordering to match grid indexing)
        offset = np.asarray([self.offset[2], self.offset[1], self.offset[0]], np.int32)

        # Apply each symmetry operation using proper R matrix and t vector
        for symop in self.unit_cell.space_group.symop_list:
            R_matrix = symop.R  # 3x3 rotation matrix
            t_vector = symop.t  # 3x1 translation vector in fractional coordinates

            extend_to_p1(
                self.array,
                offset,
                R_matrix,
                t_vector,
                out.shape,  # grid_shape_out
                out.array,
            )

        return out

    def is_canonical_unit_cell(self):
        return np.allclose(self.shape, self.unit_cell_shape[::-1]) and np.allclose(self.offset, 0)

    def extract(self, orth_coor, padding=3.0):
        """Create a copy of the map around the atomic coordinates provided.

        Args:
            orth_coor (np.ndarray[(n_atoms, 3), dtype=np.float]):
                a collection of Cartesian atomic coordinates
            padding (float): amount of padding (in Angstrom) to add around the
                returned electron density map
        Returns:
            XMap: the new map object around the coordinates
        """
        if not self.is_canonical_unit_cell():
            raise RuntimeError("XMap should contain full unit cell.")
        logger.debug(f"Extracting map around {len(orth_coor)} atoms")

        # Convert Cartesian coordinates to fractional coordinates
        frac_coor = orth_coor @ self.unit_cell.orth_to_frac.T

        # Convert fractional coordinates to grid coordinates
        # Note: unit_cell_shape is in (nx, ny, nz) order,
        # but we need (nz, ny, nx) for grid indexing
        grid_coor = (
            frac_coor * self.unit_cell_shape
        )  # Shape: (n_atoms, 3) - coordinates in (x, y, z) order

        # Subtract offset (convert offset from (x, y, z) to match grid coordinates)
        grid_coor -= self.offset

        # Calculate padding in grid units for each axis
        grid_padding = padding / self.voxelspacing  # Shape: (3,) in (x, y, z) order

        # Find the bounding box in grid coordinates (in x, y, z order)
        lb_xyz = grid_coor.min(axis=0) - grid_padding  # Lower bounds (x, y, z)
        ru_xyz = grid_coor.max(axis=0) + grid_padding  # Upper bounds (x, y, z)
        lb_xyz = np.floor(lb_xyz).astype(int)
        ru_xyz = np.ceil(ru_xyz).astype(int)

        # Convert to (z, y, x) order for array indexing
        lb = lb_xyz[::-1]  # (z, y, x) order
        ru = ru_xyz[::-1]  # (z, y, x) order
        shape = ru - lb  # Extract shape in (z, y, x) order

        logger.debug(f"From old map size (voxels): {self.shape}")
        logger.debug(f"Extract between corners:    {lb}, {ru}")
        logger.debug(f"New map size (voxels):      {shape}")

        # Create new grid parameters with updated offset
        new_offset = self.offset + lb_xyz  # Keep offset in (x, y, z) order
        grid_parameters = GridParameters(self.voxelspacing, new_offset)

        # Extract the density map region with proper periodic boundary conditions
        # Create coordinate arrays for each dimension
        z_indices = np.arange(lb[0], ru[0]) % self.shape[0]  # z dimension
        y_indices = np.arange(lb[1], ru[1]) % self.shape[1]  # y dimension
        x_indices = np.arange(lb[2], ru[2]) % self.shape[2]  # x dimension

        # Use numpy's advanced indexing with proper broadcasting
        z_grid, y_grid, x_grid = np.meshgrid(z_indices, y_indices, x_indices, indexing="ij")
        density_map = self.array[z_grid, y_grid, x_grid]

        return XMap(
            density_map,
            grid_parameters=grid_parameters,
            unit_cell=self.unit_cell,
            resolution=self.resolution,
            hkl=self.hkl,
            origin=self.origin,
        )

    def interpolate(self, xyz):
        # Transform xyz to grid coor.
        uc = self.unit_cell
        orth_to_grid = uc.orth_to_frac * self.unit_cell_shape.reshape(3, 1)
        if not np.allclose(self.origin, 0):
            xyz = xyz - self.origin
        grid_coor = orth_to_grid @ xyz.T
        grid_coor -= self.offset.reshape(3, 1)
        if self.is_canonical_unit_cell():
            grid_coor %= self.unit_cell_shape.reshape(3, 1)
        values = map_coordinates(self.array, grid_coor[::-1], order=1)
        return values

    def set_space_group(self, space_group):
        self.unit_cell.space_group = GetSpaceGroup(space_group)


class ASU:
    """Assymetric Unit Cell"""

    def __init__(self, array, grid_parameters=None, unit_cell=None, resolution=None, hkl=None):
        raise NotImplementedError
        super().__init__(array, grid_parameters, unit_cell, resolution, hkl)


# Volume parsers
def parse_volume(fid, fmt=None):
    try:
        fname = fid.name
    except AttributeError:
        fname = fid

    if fmt is None:
        fmt = os.path.splitext(fname)[-1]
    if fmt == ".ccp4":
        p = CCP4Parser(fname)
    elif fmt in (".map", ".mrc"):
        p = MRCParser(fname)
    else:
        raise ValueError("Extension of file is not supported.")
    return p


class CCP4Parser:
    HEADER_SIZE = 1024
    HEADER_TYPE = (
        "i" * 10
        + "f" * 6
        + "i" * 3
        + "f" * 3
        + "i" * 3
        + "f" * 27
        + "c" * 8
        + "f" * 1
        + "i" * 1
        + "c" * 800
    )
    HEADER_FIELDS = (
        "nc nr ns mode ncstart nrstart nsstart nx ny nz xlength ylength "
        "zlength alpha beta gamma mapc mapr maps amin amax amean ispg "
        "nsymbt lskflg skwmat skwtrn extra xstart ystart zstart map "
        "machst rms nlabel label"
    ).split()
    HEADER_CHUNKS = [1] * 25 + [9, 3, 12] + [1] * 3 + [4, 4, 1, 1, 800]

    def __init__(self, fid):
        if isinstance(fid, str):
            fhandle = open(fid, "rb")
        elif isinstance(fid, io.IOBase):
            fhandle = fid
        else:
            raise ValueError("Input should either be a file or filename.")

        self.fhandle = fhandle
        self.fname = fhandle.name

        # first determine the endiannes of the file
        self._get_endiannes()
        # get the header
        self._get_header()
        self.abc = tuple(self.header[key] for key in ("xlength", "ylength", "zlength"))
        self.angles = tuple(self.header[key] for key in ("alpha", "beta", "gamma"))
        self.shape = tuple(self.header[key] for key in ("nx", "ny", "nz"))
        self.voxelspacing = tuple(length / n for length, n in zip(self.abc, self.shape))
        self.spacegroup = int(self.header["ispg"])
        self.cell_shape = [self.header[key] for key in "nz ny nx".split()]
        self._get_offset()
        self._get_origin()
        # Get the symbol table and ultimately the density
        self._get_symbt()
        self._get_density()
        self.fhandle.close()

    def _get_endiannes(self):
        self.fhandle.seek(212)
        b = self.fhandle.read(1)

        m_stamp = hex(ord(b))
        if m_stamp == "0x44":
            endian = "<"
        elif m_stamp == "0x11":
            endian = ">"
        else:
            raise ValueError("Endiannes is not properly set in file. Check the file format.")
        self._endian = endian
        self.fhandle.seek(0)

    def _get_header(self):
        header = _unpack(self._endian + self.HEADER_TYPE, self.fhandle.read(self.HEADER_SIZE))
        self.header = {}
        index = 0
        for field, nchunks in zip(self.HEADER_FIELDS, self.HEADER_CHUNKS):
            end = index + nchunks
            if nchunks > 1:
                self.header[field] = header[index:end]
            else:
                self.header[field] = header[index]
            index = end
        self.header["label"] = "".join(x.decode("utf-8") for x in self.header["label"])

    def _get_offset(self):
        self.offset = [0] * 3
        self.offset[self.header["mapc"] - 1] = self.header["ncstart"]
        self.offset[self.header["mapr"] - 1] = self.header["nrstart"]
        self.offset[self.header["maps"] - 1] = self.header["nsstart"]

    def _get_origin(self):
        self.origin = (0, 0, 0)

    def _get_symbt(self):
        self.symbt = self.fhandle.read(self.header["nsymbt"])

    def _get_density(self):
        # Determine the dtype of the file based on the mode
        mode = self.header["mode"]
        if mode == 0:
            dtype = "i1"
        elif mode == 1:
            dtype = "i2"
        elif mode == 2:
            dtype = "f4"

        # Read the density
        storage_shape = tuple(self.header[key] for key in ("ns", "nr", "nc"))
        self.density = np.fromfile(self.fhandle, dtype=self._endian + dtype).reshape(storage_shape)

        # Reorder axis so that nx is fastest changing.
        maps, mapr, mapc = (self.header[key] for key in ("maps", "mapr", "mapc"))
        if maps == 3 and mapr == 2 and mapc == 1:
            pass
        elif maps == 3 and mapr == 1 and mapc == 2:
            self.density = np.swapaxes(self.density, 1, 2)
        elif maps == 2 and mapr == 1 and mapc == 3:
            self.density = np.swapaxes(self.density, 1, 2)
            self.density = np.swapaxes(self.density, 1, 0)
        elif maps == 1 and mapr == 2 and mapc == 3:
            self.density = np.swapaxes(self.density, 0, 2)
        else:
            msg = f"Density storage order ({maps} {mapr} {mapc}) not supported."
            raise NotImplementedError(msg)
        self.density = np.ascontiguousarray(self.density, dtype=np.float64)


class MRCParser(CCP4Parser):
    def _get_origin(self):
        origin_fields = "xstart ystart zstart".split()
        origin = [self.header[field] for field in origin_fields]
        self.origin = origin


def to_mrc(fid, volume, labels=[], fmt=None):
    if fmt is None:
        fmt = os.path.splitext(fid)[-1][1:]

    if fmt not in ("ccp4", "mrc", "map"):
        raise ValueError("Format is not recognized. Use ccp4, mrc, or map.")

    dtype = volume.array.dtype.name
    if dtype == "int8":
        mode = 0
    elif dtype in ("int16", "int32"):
        mode = 1
    elif dtype in ("float32", "float64"):
        mode = 2
    else:
        raise TypeError(f"Data type ({dtype})is not supported.")

    if fmt == "ccp4":
        nxstart, nystart, nzstart = volume.offset
        origin = [0, 0, 0]
        uc = volume.unit_cell
        xl, yl, zl = uc.a, uc.b, uc.c
        alpha, beta, gamma = uc.alpha, uc.beta, uc.gamma
        ispg = uc.space_group.number
        ns, nr, nc = volume.unit_cell_shape[::-1]
    elif fmt in ("mrc", "map"):
        nxstart, nystart, nzstart = [0, 0, 0]
        origin = volume.origin
        xl, yl, zl = (vs * n for vs, n in zip(volume.voxelspacing, reversed(volume.shape)))
        alpha = beta = gamma = 90
        ispg = 1
        ns, nr, nc = volume.shape

    nz, ny, nx = volume.shape
    mapc, mapr, maps = [1, 2, 3]
    nsymbt = 0
    lskflg = 0
    skwmat = [0.0] * 9
    skwtrn = [0.0] * 3
    fut_use = [0.0] * 12
    str_map = list("MAP ")
    str_map = "MAP "
    # TODO machst are similar for little and big endian
    if _BYTEORDER == "little":
        machst = list("\x44\x41\x00\x00")
    elif _BYTEORDER == "big":
        machst = list("\x44\x41\x00\x00")
    else:
        raise ValueError(f"Byteorder {_BYTEORDER} is not recognized")
    labels = [" "] * 800
    nlabels = 0
    min_density = volume.array.min()
    max_density = volume.array.max()
    mean_density = volume.array.mean()
    std_density = volume.array.std()

    with open(fid, "wb") as out:
        out.write(_pack("i", nx))
        out.write(_pack("i", ny))
        out.write(_pack("i", nz))
        out.write(_pack("i", mode))
        out.write(_pack("i", nxstart))
        out.write(_pack("i", nystart))
        out.write(_pack("i", nzstart))
        out.write(_pack("i", nc))
        out.write(_pack("i", nr))
        out.write(_pack("i", ns))
        out.write(_pack("f", xl))
        out.write(_pack("f", yl))
        out.write(_pack("f", zl))
        out.write(_pack("f", alpha))
        out.write(_pack("f", beta))
        out.write(_pack("f", gamma))
        out.write(_pack("i", mapc))
        out.write(_pack("i", mapr))
        out.write(_pack("i", maps))
        out.write(_pack("f", min_density))
        out.write(_pack("f", max_density))
        out.write(_pack("f", mean_density))
        out.write(_pack("i", ispg))
        out.write(_pack("i", nsymbt))
        out.write(_pack("i", lskflg))
        for f in skwmat:
            out.write(_pack("f", f))
        for f in skwtrn:
            out.write(_pack("f", f))
        for f in fut_use:
            out.write(_pack("f", f))
        for f in origin:
            out.write(_pack("f", f))
        for c in str_map:
            out.write(_pack("c", c.encode("ascii")))
        for c in machst:
            out.write(_pack("c", c.encode("ascii")))
        out.write(_pack("f", std_density))
        # max 10 labels
        # nlabels = min(len(labels), 10)
        # TODO labels not handled correctly
        # for label in labels:
        #     list_label = [c for c in label]
        #     llabel = len(list_label)
        #     if llabel < 80:
        #
        #     # max 80 characters
        #     label = min(len(label), 80)
        out.write(_pack("i", nlabels))
        for c in labels:
            out.write(_pack("c", c.encode("ascii")))
        # write density
        modes = [np.int8, np.int16, np.float32]
        volume.array.astype(modes[mode]).tofile(out)
