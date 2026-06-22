"""RMSD metric, developed starting from :class:`sampleworks.metrics.lddt.AllAtomLDDT`.

The metric returns a global per model RMSD and a per residue RMSD dictionary,
so it can be plugged into the same clustering code as the LDDT
metric.
"""

from typing import Any, cast

import numpy as np
from atomworks.io.transforms.atom_array import ensure_atom_array_stack
from atomworks.ml.transforms.atom_array import add_global_token_id_annotation
from biotite.structure import AtomArray, AtomArrayStack, rmsd, superimpose

from sampleworks.metrics.metric import Metric
from sampleworks.utils.atom_array_utils import filter_to_common_atoms


class AllAtomRMSD(Metric):
    """Computes all atom RMSD from AtomArrays.

    Parameters
    ----------
    superimpose
        If True, superimpose predicted models onto the reference via Kabsch
        before computing RMSD. Default False is correct for comparisons in a shared crystallographic
        frame, but set True when the predicted and reference structures live in different frames.
    log_rmsd_for_every_batch
        If True, include per model RMSDs as ``all_atom_rmsd_<i>`` in the
        output dict.
    """

    def __init__(
        self,
        superimpose: bool = False,
        log_rmsd_for_every_batch: bool = False,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.superimpose = superimpose
        self.log_rmsd_for_every_batch = log_rmsd_for_every_batch

    @property
    def kwargs_to_compute_args(self) -> dict[str, Any]:
        return {
            "predicted_atom_array_stack": "predicted_atom_array_stack",
            "ground_truth_atom_array_stack": "ground_truth_atom_array_stack",
            "selection": "selection",
        }

    @property
    def optional_kwargs(self) -> frozenset[str]:
        return frozenset({"selection"})

    def compute(
        self,
        predicted_atom_array_stack: AtomArrayStack | AtomArray,
        ground_truth_atom_array_stack: AtomArrayStack | AtomArray,
        selection: str | None = None,
    ) -> dict[str, Any]:
        """Calculate all-atom RMSD between predicted and ground-truth structures.

        Parameters
        ----------
        predicted_atom_array_stack
            Predicted coordinates as AtomArray(Stack).
        ground_truth_atom_array_stack
            Ground truth coordinates as AtomArray(Stack).
        selection
            Optional selection string (AtomArray.mask() syntax) restricting
            which residues appear in ``residue_rmsd_scores``. It does NOT
            restrict the atoms used to compute the global RMSD, nor (when
            ``superimpose=True``) the atoms used for Kabsch superposition -
            both of those always use every atom common to the predicted and
            reference stacks.

        Returns
        -------
        dict[str, Any]
           A dictionary with all-atom RMSDs:

            - ``best_of_1_rmsd``: global RMSD for the first model.
            - ``best_of_{N}_rmsd``: minimum global RMSD across all N models.
            - ``residue_rmsd_scores``: per residue RMSDs, one list per
              residue keyed by ``chain_id + res_id``.

           When ``selection`` is supplied, two additional keys are present:

            - ``segment_rmsd``: list of all-atom RMSDs (one per predicted
              model) restricted to the atoms in ``selection``, computed after
              the same global Kabsch superposition used for ``best_of_1_rmsd``.
            - ``best_of_{N}_segment_rmsd``: minimum of ``segment_rmsd``.
        """
        # 1. Annotate token IDs so atoms can be grouped into residues downstream.
        predicted_atom_array_stack = add_global_token_id_annotation(predicted_atom_array_stack)
        ground_truth_atom_array_stack = add_global_token_id_annotation(
            ground_truth_atom_array_stack
        )

        # 2. Restrict both stacks to atoms present in both structures, in matching order.
        _pred, _gt = filter_to_common_atoms(
            predicted_atom_array_stack, ground_truth_atom_array_stack
        )
        pred_aa_stack = ensure_atom_array_stack(_pred)
        gt_aa_stack = ensure_atom_array_stack(_gt)

        if pred_aa_stack.array_length() == 0:
            raise RuntimeError("No atoms in common between the two structures.")

        gt_ref = gt_aa_stack[0]
        tok_idx = cast(np.ndarray, gt_ref.token_id).astype(np.int64)

        # Resolve the selection mask BEFORE superposition since .mask() exists only on the
        # atomworks loaded array, not the biotite stack superimpose() returned array
        selected_token_ids: set[int] | None = None
        selection_mask: np.ndarray | None = None
        if selection is not None:
            try:
                selection_mask = pred_aa_stack.mask(selection)
            except AttributeError as e:
                raise RuntimeError(
                    "pred_aa_stack does not support mask(). Load atom arrays with "
                    "`atomworks.io.utils.io_utils.load_any()` to access this method."
                ) from e
            selected_arr = cast(AtomArray, pred_aa_stack[0, selection_mask])
            if selected_arr.token_id is not None:
                selected_token_ids = {int(t) for t in np.unique(selected_arr.token_id)}

        # 3. Optional Kabsch superposition (always on every common atom, regardless of selection).
        if self.superimpose:
            pred_aa_stack, _ = superimpose(gt_ref, pred_aa_stack)

        global_rmsd = np.asarray(rmsd(gt_ref, pred_aa_stack))

        # 4. Build residue-level output: for each token, call biotite.rmsd on the
        # atoms that share that token, keyed by "{chain_id}{res_id}".
        chain_id = cast(np.ndarray, gt_ref.chain_id)
        res_id = cast(np.ndarray, gt_ref.res_id)
        residue_keys = np.char.add(chain_id, res_id.astype(str))
        token_to_residue_id_map = {int(k): str(v) for k, v in zip(tok_idx, residue_keys)}

        residue_rmsd_scores: dict[str, list[float]] = {}
        for tk in np.unique(tok_idx):
            tk_int = int(tk)
            if selected_token_ids is not None and tk_int not in selected_token_ids:
                continue
            atom_mask = tok_idx == tk
            residue_rmsd = np.asarray(rmsd(gt_ref[atom_mask], pred_aa_stack[:, atom_mask]))
            residue_rmsd_scores[token_to_residue_id_map[tk_int]] = residue_rmsd.tolist()

        result: dict[str, Any] = {
            "best_of_1_rmsd": float(global_rmsd[0]),
            f"best_of_{len(global_rmsd)}_rmsd": float(global_rmsd.min()),
            "residue_rmsd_scores": residue_rmsd_scores,
        }

        if selection_mask is not None:
            # Segment all atom RMSD
            segment_rmsd = np.asarray(
                rmsd(gt_ref[selection_mask], pred_aa_stack[:, selection_mask])
            )
            result["segment_rmsd"] = [float(r) for r in segment_rmsd]
            result[f"best_of_{len(segment_rmsd)}_segment_rmsd"] = float(segment_rmsd.min())

        if self.log_rmsd_for_every_batch:
            result.update(
                {f"all_atom_rmsd_{i}": float(global_rmsd[i]) for i in range(len(global_rmsd))}
            )

        return result
