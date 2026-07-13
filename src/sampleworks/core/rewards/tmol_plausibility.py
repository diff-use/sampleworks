from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np
import torch
from jaxtyping import Float, Int
from tmol.io.pose_stack_construction import pose_stack_from_canonical_form

"""Physical-plausibility reward via tmol (differentiable Rosetta ``beta2016`` energy).

This reward complements the density-fit reward (:mod:`core.rewards.real_space_density`):
where the density term is the *likelihood* (per-atom agreement with the experimental map),
this term is the *prior* (inter-atomic chemistry / physical realism). It scores a conformer's
plausibility by rendering it into a tmol ``PoseStack`` and evaluating the differentiable
``beta2016`` score function, so gradients flow back to the input coordinates for gradient
guidance (and the scalar can equally be used evaluate-only for Feynman-Kac steering).

tmol needs a fully typed molecular system (residue types + per-residue-type canonical atom
order), whereas the input coordinates arrive in SampleWorks/biotite atom order. The bridge is a
scatter map, built once in :meth:`TmolPlausibilityReward.prepare` and applied per call:

- ``sel``          : indices, in the incoming coordinate order, of the protein atoms tmol scores
- ``res_of_atom``  : residue index each selected atom belongs to        (parallel to ``sel``)
- ``can_of_atom``  : canonical slot each selected atom occupies          (parallel to ``sel``)

Hydrogens and terminal OXT atoms are left as NaN in the canonical grid; tmol's
``pose_stack_from_canonical_form`` builds them differentiably from the heavy atoms.
"""



# Standard amino acids + MSE (selenomethionine). tmol can build missing leaf atoms (H, OXT)
# for these from their heavy atoms, but cannot build e.g. water hydrogens from a lone oxygen,
# so waters / ligands / modified residues are dropped from the plausibility score.
_AA20 = frozenset(
    {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
        "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL", "MSE",
    }
)


def build_tmol_map(atom_array: Any, co: Any, device: torch.device) -> dict[str, Any]:
    """Build the scatter map from a biotite ``AtomArray`` to tmol's canonical grid.

    The map is constant across a trajectory: it records, for each protein atom, where it lives
    in the incoming coordinate tensor (``sel``) and where it must be placed in tmol's
    per-residue canonical slots (``res_of_atom`` / ``can_of_atom``), plus the per-residue
    ``chain_id`` and ``res_types`` tensors tmol needs to construct the pose.

    Parameters
    ----------
    atom_array : biotite.structure.AtomArray
        Atoms in the order the reward's ``__call__`` will receive coordinates. If the array
        carries an ``occupancy`` annotation, zero-occupancy atoms are dropped first (mirroring
        the density reward); otherwise all atoms are used.
    co : tmol CanonicalOrdering
        From ``tmol.io.canonical_ordering.default_canonical_ordering()``. Provides
        ``restype_io_equiv_classes`` (residue-type index), ``restypes_atom_index_mapping``
        (atom-name -> canonical slot), and ``max_n_canonical_atoms``.
    device : torch.device
        Device for the returned tensors.

    Returns
    -------
    dict
        Keys ``sel``, ``res_of_atom``, ``can_of_atom`` (``long`` [n_protein_atoms]);
        ``chain_id``, ``res_types`` (``int32`` [1, n_res]); ``n_res`` (int);
        ``max_can`` (int, ``co.max_n_canonical_atoms``).

    Notes
    -----
    Atoms whose residue type is unknown to tmol are skipped silently; recognized residues with
    an unrecognized atom name are collected and printed once as ``UNMAPPED`` (they are dropped
    from the score rather than raising, so a stray atom name does not abort a run).
    """
    if "occupancy" in atom_array.get_annotation_categories():
        aa = atom_array[atom_array.occupancy > 0]
    else:
        aa = atom_array

    resname = [str(x).strip() for x in aa.res_name]
    atname = [str(x).strip() for x in aa.atom_name]
    chain = [str(x) for x in aa.chain_id]
    resid = np.asarray(aa.res_id)
    valid = _AA20 & set(co.restype_io_equiv_classes)

    sel: list[int] = []
    res_of_atom: list[int] = []
    can_of_atom: list[int] = []
    res_types_list: list[int] = []
    chain_of_res: list[str] = []
    missing: list[tuple[str, str]] = []

    prev_key: tuple[str, int] | None = None
    r = -1
    for i in range(len(aa)):
        if resname[i] not in valid:
            continue
        key = (chain[i], int(resid[i]))
        if key != prev_key:  # start of a new protein residue
            r += 1
            prev_key = key
            res_types_list.append(co.restype_io_equiv_classes.index(resname[i]))
            chain_of_res.append(chain[i])
        slot_map = co.restypes_atom_index_mapping[resname[i]]
        if atname[i] not in slot_map:
            missing.append((resname[i], atname[i]))
            continue
        sel.append(i)
        res_of_atom.append(r)
        can_of_atom.append(slot_map[atname[i]])
    n_res = r + 1

    if missing:
        print("UNMAPPED tmol atoms:", Counter(missing).most_common())

    chains = list(dict.fromkeys(chain_of_res))
    return {
        "sel": torch.tensor(sel, dtype=torch.long, device=device),
        "res_of_atom": torch.tensor(res_of_atom, dtype=torch.long, device=device),
        "can_of_atom": torch.tensor(can_of_atom, dtype=torch.long, device=device),
        "chain_id": torch.tensor(
            [chains.index(c) for c in chain_of_res], dtype=torch.int32, device=device
        ).unsqueeze(0),
        "res_types": torch.tensor(res_types_list, dtype=torch.int32, device=device).unsqueeze(0),
        "n_res": n_res,
        "max_can": co.max_n_canonical_atoms,
    }


class TmolPlausibilityReward:
    """Differentiable physical-plausibility reward (tmol ``beta2016`` energy).

    Satisfies ``RewardFunctionProtocol``: ``__call__`` returns a scalar loss (lower = more
    plausible), differentiable w.r.t. the input coordinates. Elements, B-factors, and
    occupancies are ignored -- atom identity comes from the scatter map, not the per-call
    scattering inputs.

    The scatter map is built lazily via :meth:`prepare`, because it must match the atom order
    of the coordinates the sampler actually feeds in (``model_atom_array or atom_array`` from
    the processed structure), which is only known after model-specific preprocessing.

    Parameters
    ----------
    co : tmol CanonicalOrdering
        ``default_canonical_ordering()``.
    pbt : tmol PackedBlockTypes
        ``default_packed_block_types(device)``.
    sfxn : tmol ScoreFunction
        ``tmol.beta2016_score_function(device)``.
    device : torch.device
        Device the pose/score run on.
    weight : float, optional
        Multiplier on the returned energy (default 1.0).
    """

    def __init__(self, co: Any, pbt: Any, sfxn: Any, device: torch.device, weight: float = 1.0):
        self.co = co
        self.pbt = pbt
        self.sfxn = sfxn
        self.device = device
        self.weight = weight
        self._m: dict[str, Any] | None = None

    def prepare(self, atom_array: Any) -> "TmolPlausibilityReward":
        """Build the scatter map from the array whose order matches the sampler coordinates.

        Call once before sampling with ``processed.model_atom_array or processed.atom_array``.

        Parameters
        ----------
        atom_array : biotite.structure.AtomArray
            The atom array in the order ``__call__`` will receive coordinates.

        Returns
        -------
        TmolPlausibilityReward
            ``self``, for chaining.
        """
        self._m = build_tmol_map(atom_array, self.co, self.device)
        return self

    def __call__(
        self,
        coordinates: Float[torch.Tensor, "batch n_atoms 3"],
        elements: Int[torch.Tensor, "batch n_atoms"] | None = None,
        b_factors: Float[torch.Tensor, "batch n_atoms"] | None = None,
        occupancies: Float[torch.Tensor, "batch n_atoms"] | None = None,
        unique_combinations: torch.Tensor | None = None,
        inverse_indices: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, ""]:
        """Score conformer plausibility. Call ``.backward()`` for gradients w.r.t. coordinates.

        Parameters
        ----------
        coordinates : Float[torch.Tensor, "batch n_atoms 3"]
            Atomic coordinates in the atom order :meth:`prepare` was built against.
        elements, b_factors, occupancies, unique_combinations, inverse_indices
            Accepted for protocol compatibility; ignored (atom identity comes from the map).

        Returns
        -------
        Float[torch.Tensor, ""]
            Scalar loss = ``weight * sum_b energy(pose_b)``. Each pose is scored independently,
            so the per-particle gradient ``d loss / d coords_b`` depends only on pose ``b``.
        """
        m = self._m
        if m is None:
            raise RuntimeError(
                "TmolPlausibilityReward.prepare(atom_array) must be called before __call__."
            )

        batch = coordinates.shape[0]
        # NaN-fill the canonical grid; tmol builds missing leaf atoms (H, OXT) from heavy atoms.
        coords = coordinates.new_full((batch, m["n_res"], m["max_can"], 3), float("nan"))
        # scatter: pick protein atoms (sel) -> place in their canonical slots (differentiable).
        coords[:, m["res_of_atom"], m["can_of_atom"]] = coordinates[:, m["sel"]]

        pose = pose_stack_from_canonical_form(
            self.co,
            self.pbt,
            m["chain_id"].expand(batch, -1),
            m["res_types"].expand(batch, -1),
            coords,
            None,
            None,
            None,
        )
        # render_whole_pose_scoring_module returns a per-pose total energy: shape [batch].
        energy = self.sfxn.render_whole_pose_scoring_module(pose)(pose.coords)
        return self.weight * energy.sum()
