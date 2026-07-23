"""Coordinate-space bond-geometry regularizer for inference-time latent optimization (IT-opt).

When the density reward is optimized aggressively, the latent update can distort the denoised
structure -- stretching covalent bonds and driving non-bonded atoms into steric clashes -- to buy a
little density fit. That is the overshoot observed on high-baseline targets (e.g. 1VME, where
z-optimization raised clashes while dropping RSCC). This module adds a differentiable penalty on the
denoised coordinates that the optimizer must trade against, so it cannot reach a good density score
through broken geometry.

It is a faithful port of the reference it_opt ``BondLengthLossFunction``
(``it_opt/protenix/src/losses/bond_length_loss_function.py``): a bonded-pair length hinge plus a
non-bonded steric-clash hinge. Both are bounded ``relu`` hinges (not the reference's
``exp(relu(...))`` clash term, which explodes), so the gradients stay well scaled. It is meant to be
an additive term inside ``LatentOptimization``'s per-step loss, not a standalone objective, and does
nothing unless its weight is set.
"""

from __future__ import annotations

import gemmi
import numpy as np
import torch
from biotite.structure import AtomArray, connect_via_residue_names
from jaxtyping import Float
from torch import Tensor


def _covalent_radius(element: str) -> float:
    """Covalent radius of an element (Å), or 0 for an unknown/blank symbol.

    Model atom arrays are almost always clean single-symbol elements, but density inputs occasionally
    carry a placeholder element ('?'); we return 0 rather than raise so one odd atom cannot abort the
    whole run.
    """
    try:
        return float(gemmi.Element(element).covalent_r)
    except Exception:  # noqa: BLE001 -- a bad element symbol just gets a zero radius
        return 0.0


class BondGeometryReward:
    """Penalize distorted bonds and steric clashes in the denoised structure.

    The topology (which atoms are bonded, their ideal bond lengths, and the per-atom-pair collision
    distances) is read once from ``atom_array`` at construction, so each step is only cheap tensor
    ops on the coordinates. ``atom_array`` MUST be the atom set whose ordering matches the coordinate
    tensor the reward sees -- the *model* atom array, not the deposited structure -- or the bond
    indices will not line up with the coordinates.

    Despite the ``Reward`` name (matching the module's convention), this is a penalty: it is added to,
    and minimized as part of, ``LatentOptimization``'s loss.
    """

    def __init__(
        self,
        atom_array: AtomArray,
        weight: float,
        device: torch.device | str,
        *,
        bond_tolerance: float = 0.2,
        clash_padding: float = 0.4,
        bond_power: int = 2,
    ):
        """Build the bond topology and collision-distance matrix from ``atom_array`` once.

        ``weight`` scales the whole penalty; ``bond_tolerance`` (Å) is the slack a bond may deviate
        from its ideal length before it is penalized; ``clash_padding`` (Å) is added to the sum of
        covalent radii to set how close non-bonded atoms may approach before they count as clashing;
        ``bond_power`` is the exponent on the bond-length hinge (2 = quadratic, as in the reference).
        """
        self.weight = weight
        self.bond_tolerance = bond_tolerance
        self.clash_padding = clash_padding
        self.bond_power = bond_power
        self._bonds, self._bond_lengths = self._build_bonds(atom_array, device)
        self._collision_distances = self._build_collision_distances(atom_array, device)

    # ================================ topology (built once) ================================

    def _build_bonds(self, atom_array: AtomArray, device) -> tuple[Tensor, Tensor]:
        """Bonded-atom index pairs and their ideal lengths, from residue-name templates.

        biotite's ``connect_via_residue_names`` infers the intra- and inter-residue bonds from the
        standard component templates; ``get_all_bonds`` returns, per atom, the indices of its bonded
        partners padded with -1. We collapse that to a unique set of index pairs and take each bond's
        ideal length as the sum of the two atoms' covalent radii. Returns an ``[n_bonds, 2]`` index
        tensor and an ``[n_bonds]`` length tensor.
        """
        partners_per_atom, _ = connect_via_residue_names(atom_array).get_all_bonds()
        pairs = {
            tuple(sorted((int(i), int(j))))
            for i, partners in enumerate(partners_per_atom)
            for j in partners
            if j != -1
        }
        pair_array = np.array(sorted(pairs)) if pairs else np.empty((0, 2), dtype=int)
        lengths = [
            _covalent_radius(atom_array[a].element) + _covalent_radius(atom_array[b].element)
            for a, b in pair_array
        ]
        bonds = torch.tensor(pair_array, dtype=torch.long, device=device)
        bond_lengths = torch.tensor(lengths, dtype=torch.float32, device=device)
        return bonds, bond_lengths

    def _build_collision_distances(self, atom_array: AtomArray, device) -> Tensor:
        """The ``[n_atoms, n_atoms]`` matrix of minimum non-clashing distances (covalent-radii sums)."""
        radii = np.array([_covalent_radius(atom_array[i].element) for i in range(len(atom_array))])
        r = torch.tensor(radii, dtype=torch.float32, device=device)
        return r[None] + r[:, None]

    # ============================== the two penalty terms ==============================

    def bond_length_loss(self, coords: Float[Tensor, "e n 3"]) -> Tensor:
        """Hinge on bonded-pair length deviation beyond ``bond_tolerance``.

        A bond is free while it stays within ``bond_tolerance`` of its ideal length; past that the
        deviation is raised to ``bond_power`` and summed over all bonds and ensemble members. Keeps
        the optimizer from stretching or compressing covalent bonds to chase density.
        """
        if self._bonds.numel() == 0:
            return coords.new_zeros(())
        lengths = (coords[:, self._bonds[:, 0]] - coords[:, self._bonds[:, 1]]).norm(dim=-1)
        excess = ((lengths - self._bond_lengths).abs() - self.bond_tolerance).relu()
        return excess.pow(self.bond_power).sum()

    def collision_loss(self, coords: Float[Tensor, "e n 3"]) -> Tensor:
        """Hinge on non-bonded atoms closer than their collision distance + ``clash_padding``.

        This is the steric-clash term. Bonded and self pairs are excluded by pushing their distance
        out of range. The overlap is reduced over the ensemble by its worst-case member per atom pair
        (matching the reference), then summed. Note the pairwise-distance tensor is O(n_atoms**2), so
        memory scales with the square of the atom count.
        """
        distances = (coords[:, :, None] - coords[:, None]).norm(dim=-1)
        i, j = self._bonds[:, 0], self._bonds[:, 1]
        mask = torch.zeros_like(distances)
        mask[:, i, j] = 3.0
        mask[:, j, i] = 3.0
        mask[:, i, i] = 3.0
        mask[:, j, j] = 3.0
        distances = distances + mask
        overlap = (self._collision_distances + self.clash_padding - distances).max(dim=0)[0].relu()
        return overlap.sum()

    def __call__(self, coords: Float[Tensor, "e n 3"]) -> Tensor:
        """Weighted sum of the bond-length and collision penalties -- the value added to the loss."""
        return self.weight * (self.bond_length_loss(coords) + self.collision_loss(coords))
