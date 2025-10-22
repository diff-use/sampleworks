# pyright: ignore

"""Minimal components of qFit's transformer module necessary to implement
qFit's volume.py

Author: Karson Chrispens
Date: 16 Feb 2025
"""

import numpy as np
from numpy.fft import irfftn

from .unitcell import UnitCell


class SFTransformer:
    def __init__(
        self, hkl: np.ndarray, f: np.ndarray, phi: np.ndarray, unit_cell: UnitCell
    ):
        self.hkl = hkl
        self.f = f
        self.phi = phi
        self.unit_cell = unit_cell
        self._resolution = np.min(unit_cell.abc / np.abs(hkl).max(axis=0))

    def __call__(self, nyquist: float = 2) -> np.ndarray:
        h, k, l = self.hkl.T
        voxelspacing = self._resolution / (2 * nyquist)
        shape = np.ceil(self.unit_cell.abc / voxelspacing)
        shape = shape[::-1].astype(int)
        fft_grid = np.zeros(shape, dtype=np.complex128)

        start_sf = self.f * np.exp(-1j * np.deg2rad(self.phi))
        start_sf = np.nan_to_num(start_sf.astype(np.complex64))

        symops = self.unit_cell.space_group.symop_list
        primitive_symops = symops[: self.unit_cell.space_group.num_primitive_sym_equiv]

        hsym = np.zeros_like(h)
        ksym = np.zeros_like(k)
        lsym = np.zeros_like(l)

        for symop in primitive_symops:
            for n, msym in enumerate((hsym, ksym, lsym)):
                msym.fill(0)
                rot = np.asarray(symop.R.T)[n]
                for r, m in zip(rot, (h, k, l)):
                    if r != 0:
                        msym += int(r) * m

            if np.allclose(symop.t, 0):
                sf = start_sf
            else:
                delta_phi = np.rad2deg(np.inner((-2 * np.pi * symop.t), self.hkl))
                sf = start_sf * np.exp(-1j * np.deg2rad(delta_phi.ravel()))

            fft_grid[lsym, ksym, hsym] = sf
            fft_grid[-lsym, -ksym, -hsym] = sf.conj()

        nx = shape[-1]
        grid = irfftn(fft_grid[:, :, : nx // 2 + 1])
        grid -= grid.mean()
        grid /= grid.std()

        return grid
