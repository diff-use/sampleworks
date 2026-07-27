"""Coordinate-space bond-geometry regularizer for inference-time latent optimization (IT-opt).

When the density reward is optimized aggressively, the latent update can distort the denoised
structure -- stretching covalent bonds and driving non-bonded atoms into steric clashes -- to buy a
little density fit. That is the overshoot observed on high-baseline targets (e.g. 1VME, where
z-optimization raised clashes while dropping RSCC). This module adds a differentiable penalty on the
denoised coordinates that the optimizer must trade against, so it cannot reach a good density score
through broken geometry.

It is a faithful port of the reference it_opt ``BondLengthLossFunction``
(``it_opt/protenix/src/losses/bond_length_loss_function.py``): a bonded-pair length hinge plus a
non-bonded steric-clash hinge. Both are bounded hinges -- the violation's positive part, clamped
at zero (not the reference's ``exp(relu(...))`` clash term, which explodes) -- so the gradients stay
well scaled. It is meant to be an additive term inside ``LatentOptimization``'s per-step loss, not a
standalone objective, and does nothing unless its weight is set.
"""

from __future__ import annotations

import gemmi
import numpy as np
import torch
from biotite.structure import AtomArray, connect_via_residue_names
from torch import Tensor


# ---- Reading the type hints in this file --------------------------------------
# A ": type" after a name (or "-> type" after a function) is only a HINT -- it CLAIMS what a
# value should be, but nothing enforces it: pass the wrong type and Python still runs the code,
# and deleting every hint changes nothing. Hints are for humans (and optional checkers like ty).
# (In the table, "|" means "or".)
#
#   with the hint                 plain Python           what it claims (useless in runtime)
#   element: str                  element                should be a string
#   atom_array: AtomArray         atom_array             should be a biotite AtomArray
#   coords: Tensor                coords                 should be a Tensor (~ a numpy array)
#   bond_power: int = 2           bond_power = 2         should be an int
#   bond_tolerance: float = 0.2   bond_tolerance = 0.2   should be a float
#   device: torch.device | str    device                 should be a torch.device or a str
#   f(...) -> float               f(...)                 f should return a float
#   f(...) -> Tensor              f(...)                 f should return a Tensor
# -------------------------------------------------------------------------------


def _covalent_radius(element: str) -> float:
    """Covalent radius of an element (Å), or 0 for an unknown/blank symbol.

    Model atom arrays are almost always clean single-symbol elements, but density inputs sometimes
    carry a placeholder element ('?'); we return 0 rather than raise so one odd atom can't abort the
    whole run.
    """
    try:
        return float(gemmi.Element(element).covalent_r)
    except Exception:  # noqa: BLE001 -- a bad element symbol just gets a zero radius
        return 0.0


class BondGeometryReward:
    """Penalize distorted bonds and steric clashes in the denoised structure.

    The topology (which atoms are bonded, their ideal bond lengths, the per-atom-pair collision
    distances, and which pairs to score for clashes) is read once from ``atom_array`` at
    construction, so each step is just cheap tensor ops on the coords. ``atom_array`` MUST be the
    atom set whose ordering matches the coordinate tensor the reward sees -- the *model* atom array,
    not the deposited structure -- or the bond indices will not line up with the coordinates.

    Despite the ``Reward`` name (the module's convention), this is a penalty: it is added to,
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
        ``bond_power`` is the exponent on the bond-length hinge (2 = quadratic, per the reference).
        """
        self.weight = weight
        self.bond_tolerance = bond_tolerance
        self.clash_padding = clash_padding
        self.bond_power = bond_power

        # Each bond connects atom a to atom b and has an ideal length; these three arrays line up,
        # one entry per bond.
        self._bond_atom_a, self._bond_atom_b, self._bond_lengths = self._build_bonds(
            atom_array, device
        )
        self._collision_distances = self._build_collision_distances(atom_array, device)
        self._scored_pairs = self._build_scored_pairs(len(atom_array), device)

    # ================================ topology (built once) ================================

    def _build_bonds(self, atom_array: AtomArray, device):
        """Bonded-atom endpoints and ideal lengths, inferred from residue-name templates.

        biotite's ``connect_via_residue_names`` infers the intra- and inter-residue bonds from the
        standard component templates; ``get_all_bonds`` returns, per atom, the indices of its bonded
        partners padded with -1. We collect each bond once; its ideal length is the sum of the
        two atoms' covalent radii.

        Returns three aligned 1-D tensors, each of length n_bonds: the first endpoint of every bond,
        the second endpoint, and the ideal length.
        """
        partners_per_atom, _ = connect_via_residue_names(atom_array).get_all_bonds()

        # Collect each bond once as a sorted (low, high) index pair, so it is not listed as both
        # i-j and j-i.
        pairs = set()
        for atom_i, partners in enumerate(partners_per_atom):
            for atom_j in partners:
                if atom_j == -1:  # padding entry, not a real partner
                    continue
                bond = (min(atom_i, int(atom_j)), max(atom_i, int(atom_j)))
                pairs.add(bond)
        pair_array = np.array(sorted(pairs)) if pairs else np.empty((0, 2), dtype=int)

        # Ideal length of a bond = sum of the two atoms' covalent radii.
        lengths = [
            _covalent_radius(atom_array[a].element) + _covalent_radius(atom_array[b].element)
            for a, b in pair_array
        ]

        atom_a = torch.tensor(pair_array[:, 0], dtype=torch.long, device=device)
        atom_b = torch.tensor(pair_array[:, 1], dtype=torch.long, device=device)
        bond_lengths = torch.tensor(lengths, dtype=torch.float32, device=device)
        return atom_a, atom_b, bond_lengths

    def _build_collision_distances(self, atom_array: AtomArray, device) -> Tensor:
        """The [n_atoms, n_atoms] matrix of minimum non-clashing distances (covalent-radii sums).

        Entry (i, j) is the sum of atom i's and atom j's covalent radii; two non-bonded atoms closer
        than this (plus ``clash_padding``) are overlapping.
        """
        radii = np.array([_covalent_radius(atom_array[i].element) for i in range(len(atom_array))])
        r = torch.tensor(radii, dtype=torch.float32, device=device)

        # radius_i + radius_j for every pair: a column plus a row vector broadcasts to [n, n].
        radius_column = r.unsqueeze(1)  # [n, 1]
        radius_row = r.unsqueeze(0)  # [1, n]
        return radius_column + radius_row

    def _build_scored_pairs(self, n_atoms: int, device) -> Tensor:
        """Boolean [n_atoms, n_atoms] mask: which atom pairs count toward the clash penalty.

        We penalize every pair EXCEPT an atom with itself (distance 0) and directly-bonded atoms
        (they should sit close together). Built once from the fixed bond topology; the mask
        is symmetric, so each clashing pair is counted twice in the sum, matching the reference.
        """
        scored = torch.ones((n_atoms, n_atoms), dtype=torch.bool, device=device)
        # an atom never clashes with itself
        scored.fill_diagonal_(False)
        # directly-bonded atoms are supposed to sit close, so don't count them as clashes
        scored[self._bond_atom_a, self._bond_atom_b] = False
        scored[self._bond_atom_b, self._bond_atom_a] = False
        return scored

    # ============================== the two penalty terms ==============================

    def bond_length_loss(self, coords: Tensor) -> Tensor:
        """Hinge on bonded-pair length deviation beyond ``bond_tolerance``.

        ``coords`` is the denoised coordinate tensor, shape [ensemble_members, atoms, 3]. A bond is
        free within ``bond_tolerance`` of its ideal length; past that the deviation is
        raised to ``bond_power`` and summed over all bonds and ensemble members. Keeps the optimizer
        from stretching or compressing covalent bonds to chase density.
        """
        if self._bond_lengths.numel() == 0:
            return coords.new_zeros(())

        # Current length of every bond, for every ensemble member: [ensemble, n_bonds].
        pos_a = coords[:, self._bond_atom_a]
        pos_b = coords[:, self._bond_atom_b]
        lengths = (pos_a - pos_b).norm(dim=-1)

        deviation = (lengths - self._bond_lengths).abs()  # distance from the ideal length
        excess = (deviation - self.bond_tolerance).clamp(min=0)  # part beyond the tolerance
        return excess.pow(self.bond_power).sum()

    def collision_loss(self, coords: Tensor) -> Tensor:
        """Hinge on non-bonded atoms closer than their collision distance + ``clash_padding``.

        ``coords`` is the denoised coordinate tensor, shape [ensemble, atoms, 3]. This is the
        steric-clash term: overlap is reduced over the ensemble to its worst member per atom
        pair (matching the reference), then summed over the scored pairs. Note the pairwise-distance
        tensor is O(n_atoms**2), so memory scales with the square of the atom count.
        """
        # Distance between every pair of atoms, for each ensemble member: [ensemble, n, n].
        # torch.cdist gives the same distances without building the [ensemble, n, n, 3] grid of
        # coordinate differences a manual broadcast would (3x the distance matrix). Verified on the
        # pinned torch (2.7): its gradient is finite and matches the broadcast form even for
        # coincident atoms, so the historical cdist zero-distance NaN does not apply here.
        distances = torch.cdist(coords, coords)

        # How far each pair is inside its allowed distance (positive = overlapping).
        min_distance = self._collision_distances + self.clash_padding
        overlap_per_member = (min_distance - distances).clamp(min=0)

        # Worst overlap across the ensemble for each pair, then drop self- and bonded pairs.
        worst_overlap = overlap_per_member.max(dim=0).values  # [n, n]
        scored_overlap = worst_overlap * self._scored_pairs  # zero out the unscored pairs
        return scored_overlap.sum()

    def __call__(self, coords: Tensor) -> Tensor:
        """Weighted sum of the bond-length and collision penalties, added to the loss."""
        return self.weight * (self.bond_length_loss(coords) + self.collision_loss(coords))
