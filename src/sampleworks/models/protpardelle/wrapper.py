"""Wrapper for Protpardelle-1c models.

Follows the ``FlowModelWrapper`` protocol in
``sampleworks.models.protocol`` so Protpardelle can be used interchangeably
with other structure predictors in sampling pipelines.

This wrapper targets the *sequence-conditioned* all-atom Protpardelle-1c
models (``task: "ai-allatom"``, e.g. the model defined by ``cc89.yaml``).
These models take a protein sequence as their only conditioning input and
generate an all-atom structure by running the full reverse-diffusion
trajectory internally. However, note that we pass a "sequence" as coordinates
for all possible atoms (Protpardelle's atom37 representation), with unused
atoms zeroed out.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import biotite.structure as struc
import numpy as np
import torch
from atomworks.enums import ChainType
from jaxtyping import Float, Int
from loguru import logger
from protpardelle.common import residue_constants
from protpardelle.core.models import load_model, Protpardelle
from protpardelle.data.sequence import seq_to_aatype
from torch import Tensor

from sampleworks.eval.structure_utils import get_asym_unit_from_structure
from sampleworks.models.protocol import GenerativeModelInput
from sampleworks.utils.framework_utils import match_batch


# Number of atom37 slots in the all-atom (atom37) representation. Matches
# ``residue_constants.atom_type_num`` and ``struct_model.n_atoms`` in the configs.
ATOM37_NUM_ATOMS = residue_constants.atom_type_num


# Number of aatype tokens for the 21-token alphabet (20 canonical + X).
# Matches ``data.n_aatype_tokens`` in the Protpardelle-1c configs (e.g. cc89.yaml).
NUM_AATYPE_TOKENS = 21

# Atomworks keeps selenomethionine atoms as ``SE`` while Protpardelle's atom37
# representation uses the canonical methionine sulfur slot, ``SD``.
ATOM37_ATOM_NAME_ALIASES = {"SE": "SD"}



@dataclass(slots=True)
class ProtpardelleConditioning:
    """Sequence-derived conditioning for a Protpardelle sampling run.

    All tensors carry a leading batch (ensemble) dimension and share the same
    padded length ``L``. Padding positions are indicated by ``seq_mask == 0``.

    Attributes
    ----------
    aatype : Int[Tensor, "batch L"]
        Per-residue amino-acid type indices (21-token alphabet). This is the
        sequence the structure is conditioned on (passed as ``gt_aatype``).
    seq_mask : Float[Tensor, "batch L"]
        1.0 for real residues, 0.0 for padding.
    residue_index : Float[Tensor, "batch L"]
        Residue ordering, with chain gaps applied by the model.
    chain_index : Float[Tensor, "batch L"]
        Integer chain id per residue.
    atom_mask : Float[Tensor, "batch L 37"]
        atom37 occupancy mask implied by ``aatype``; defines which atoms the
        model places and how predicted coordinates are flattened.
    atom37_residue_index : Int[Tensor, "atoms"]
        For each of the ``N`` flat atoms in the sampler's ``x_t`` (shape
        ``batch x N x 3``), the residue position ``0..L-1`` that atom belongs to.
        Together with :attr:`atom37_atom_index` this maps the flat coordinate
        tensor into the model's ``batch x L x 37 x 3`` atom37 layout (see
        :meth:`ProtpardelleWrapper._convert_to_atom37`). Shared across the batch
        (a single input structure), so it carries no batch dimension.
    atom37_atom_index : Int[Tensor, "atoms"]
        For each of the ``N`` flat atoms, the atom37 slot ``0..36`` it occupies,
        i.e. ``residue_constants.atom_order[atom_name]``.
    sequences : tuple[str, ...]
        The input one-letter sequences, one per protein chain (for reference).
    x_self_conditioning: Float[Tensor, "batch L 37 3"]
        Self-conditioning for the input structure,
        essentially the previously denoised coordinates (optional).
    """

    aatype: Int[Tensor, "batch L"]
    seq_mask: Float[Tensor, "batch L"]
    residue_index: Float[Tensor, "batch L"]
    chain_index: Float[Tensor, "batch L"]
    atom_mask: Float[Tensor, "batch L 37"]
    atom37_residue_index: Int[Tensor, "atoms"]
    atom37_atom_index: Int[Tensor, "atoms"]
    sequences: tuple[str, ...]
    x_self_conditioning: Float[Tensor, "batch L 37 3"] | None = None
    _initialized: bool = field(default=False, init=False, repr=False)

    _FROZEN = frozenset(
        {
            "aatype",
            "seq_mask",
            "residue_index",
            "chain_index",
            "atom_mask",
            "sequences",
        }
    )

    def __post_init__(self) -> None:
        """Mark construction complete so selected conditioning fields become immutable."""
        object.__setattr__(self, "_initialized", True)

    def __setattr__(self, key, value):
        if key in self._FROZEN and getattr(self, "_initialized", False):
            raise AttributeError(
                f"Cannot set attribute {key!r} on {self.__class__.__name__}, it is frozen!"
            )
        object.__setattr__(self, key, value)


@dataclass
class ProtpardelleConfig:
    """Configuration for Protpardelle featurization.

    The denoising loop is driven by the sampleworks sampler (``AF3EDMSampler``),
    which owns the schedule, stochasticity, and step scaling. The only setting
    this wrapper reads is therefore the ensemble size.

    Attributes
    ----------
    ensemble_size : int
        Number of structures to sample for the input sequence (batch dim). Default 8.
    """

    ensemble_size: int = 8


def annotate_structure_for_protpardelle(
    structure: dict,
    *,
    ensemble_size: int = 8,
) -> dict:
    """Annotate an Atomworks structure with Protpardelle-specific configuration.

    Parameters
    ----------
    structure : dict
        Atomworks structure dictionary.
    ensemble_size : int
        See :class:`ProtpardelleConfig`.

    Returns
    -------
    dict
        Structure dict with a ``"_protpardelle_config"`` key added.
    """
    config = ProtpardelleConfig(ensemble_size=ensemble_size)
    return {**structure, "_protpardelle_config": config}


def extract_protein_sequences(structure: dict) -> list[str]:
    """Extract canonical one-letter protein sequences from an Atomworks structure dictionary,
    created by the Atomworks ``parse`` method.

    Parameters
    ----------
    structure : dict
        Atomworks structure dictionary. Sequences are read from
        ``structure["chain_info"]`` using the
        ``processed_entity_canonical_sequence`` of each protein (POLYPEPTIDE_L)
        chain, in chain order.

    Returns
    -------
    list[str]
        One sequence per protein chain.

    Raises
    ------
    ValueError
        If no ``chain_info`` is present or no protein chains are found.
    """
    chain_info = structure.get("chain_info")
    if not chain_info:
        raise ValueError(
            "Protpardelle featurization requires 'chain_info' with canonical "
            "sequences; none was found on the structure."
        )

    sequences: list[str] = []
    for chain_id, info in chain_info.items():
        chain_type: ChainType = info["chain_type"]
        if chain_type == ChainType.POLYPEPTIDE_L:
            sequence = info["processed_entity_canonical_sequence"]
            if sequence:
                sequences.append(sequence)
        else:
            logger.warning(
                f"Skipping non-protein chain {chain_id!r} (chain_type={chain_type}); "
                "Protpardelle only models L-polypeptide chains."
            )

    if not sequences:
        raise ValueError("No L-polypeptide (protein) chains found for Protpardelle.")

    return sequences


class ProtpardelleWrapper:
    """Wrapper for sequence-conditioned Protpardelle-1c all-atom models."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        config_path: str | Path,
        device: torch.device | str,
        model: Protpardelle | None = None,
    ):
        """
        Parameters
        ----------
        config_path : str | Path
            Path to the Protpardelle model config YAML (e.g. ``cc89.yaml``).
            Required unless ``model`` is provided.
        checkpoint_path : str | Path
            Path to the matching ``.pth`` checkpoint with trained weights.
            Required unless ``model`` is provided.
        device : torch.device | str
            Device to load the model on. When ``None``, Protpardelle picks the
            default device (CUDA if available).
        model : Protpardelle | None
            A pre-built Protpardelle model. When given, ``config_path`` and
            ``checkpoint_path`` are still required (but only used for record keeping);
            the model is used directly (useful for testing and advanced reuse).
        """

        self._device = torch.device(device)
        self.config_path = Path(config_path).expanduser().resolve()
        self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
        if model is not None:
            self.model = model.to(self._device).eval()
        else:
            logger.info(
                f"Loading Protpardelle model from {self.config_path.name} "
                f"(checkpoint {self.checkpoint_path.name})"
            )
            self.model = load_model(self.config_path, self.checkpoint_path, device=self._device)

        if self.model.use_mpnn_model:
            logger.warning(
                "Loaded Protpardelle model has a MiniMPNN (sequence design) head; "
                "this wrapper treats the input sequence as fixed conditioning "
                "(gt_aatype) and does not redesign it."
            )

    @property
    def device(self) -> torch.device:
        return self._device

    def featurize(self, structure: dict) -> GenerativeModelInput[ProtpardelleConditioning]:
        """From an Atomworks structure, derive Protpardelle sequence conditioning.

        Reads the canonical protein sequence(s) from the structure and builds
        the sequence mask, residue/chain indices, aatype, and atom37 mask
        required to condition sampling. No diffusion is run here.

        Parameters
        ----------
        structure : dict
            Atomworks structure dictionary, optionally annotated via
            :func:`annotate_structure_for_protpardelle`. Config is read from
            ``structure["_protpardelle_config"]`` if present, else defaults.

        Returns
        -------
        GenerativeModelInput[ProtpardelleConditioning]
            Reference ``x_init`` (sampled from the prior) and sequence
            conditioning for :meth:`step`.
        """
        config = structure.get("_protpardelle_config", ProtpardelleConfig())
        if isinstance(config, dict):
            config = ProtpardelleConfig(**config)

        if "asym_unit" not in structure:
            raise ValueError(
                "Protpardelle featurization requires an 'asym_unit' atom array to "
                "map flat coordinates into the atom37 layout; none was found on "
                "the structure."
            )

        sequences = extract_protein_sequences(structure)

        # Lay out chains exactly as Protpardelle's own sampling helper does so
        # residue/chain indexing (including inter-chain gaps) matches training.
        # Keep the length tensor on CPU because Protpardelle's helper constructs
        # intermediate residue indices on CPU before moving its outputs to the
        # model device.
        prot_lens_per_chain = torch.tensor([[len(seq) for seq in sequences]], dtype=torch.long)
        seq_mask, residue_index, chain_index = self.model.make_seq_mask_for_sampling(
            prot_lens_per_chain=prot_lens_per_chain
        )

        # Concatenate per-chain aatypes in chain order; chains are placed
        # contiguously at the front of the padded sequence by the helper above.
        chain_aatypes = [seq_to_aatype(seq, num_tokens=NUM_AATYPE_TOKENS) for seq in sequences]
        flat_aatype = torch.cat(chain_aatypes).to(self.device)
        padded_len = seq_mask.shape[1]
        aatype = torch.zeros((1, padded_len), dtype=torch.long, device=self.device)
        aatype[0, : flat_aatype.shape[0]] = flat_aatype

        # Expand to the requested ensemble size.
        ensemble_size = config.ensemble_size
        aatype = match_batch(aatype, target_batch_size=ensemble_size)
        seq_mask = match_batch(seq_mask, target_batch_size=ensemble_size)
        residue_index = match_batch(residue_index, target_batch_size=ensemble_size)
        chain_index = match_batch(chain_index, target_batch_size=ensemble_size)

        # Map the flat per-atom coordinates the sampler works with (``x_t``,
        # shape ``batch x N x 3``) onto the model's ``batch x L x 37 x 3`` atom37
        # layout. The mapping is derived from the input structure's atom names.
        atom_array = get_asym_unit_from_structure(structure, atom_array_index=0)
        atom37_residue_index, atom37_atom_index = self._atom37_indices_from_atom_array(atom_array)

        # the atom_mask created here is used in two ways. First it is used to generate
        # the initial noisy coordinates (see initialize_from_prior() below) simply for the
        # atom count. Second it is used in .step() to convert back from atom37 coordinates
        # (B x res x 37 x 3) to (B x atoms x 3) coordinates. It is only used to operate on
        # the structure we will model, never the reference structure which may have missing
        # atoms.
        #
        # This should be the way to make the atom mask:
        #   atom_mask = atom37_mask_from_aatype(aatype, seq_mask) but it doesn't handle OXT,
        # so:
        atom_mask = torch.zeros(
            (padded_len, ATOM37_NUM_ATOMS), dtype=torch.float, device=self.device
        )
        atom_mask[atom37_residue_index, atom37_atom_index] = 1
        atom_mask = match_batch(atom_mask[None, :, :], target_batch_size=ensemble_size)

        conditioning = ProtpardelleConditioning(
            aatype=aatype,
            seq_mask=seq_mask,
            residue_index=residue_index,
            chain_index=chain_index,
            atom_mask=atom_mask,
            atom37_residue_index=atom37_residue_index,
            atom37_atom_index=atom37_atom_index,
            sequences=tuple(sequences),
            x_self_conditioning=None,
        )

        # x_init is a shape-compatible reference drawn from a Gaussian prior.
        # TODO: Claude put this in. I don't think we really need it. Our sampler
        #   will call the initalize_from_prior() method, see, e.g.,
        #   sampleworks.core.scalers.pure_guidance.PureGuidance.sample
        num_atoms = int(atom_mask[0].sum().item())
        x_init = self.initialize_from_prior(batch_size=ensemble_size, shape=(num_atoms, 3))

        return GenerativeModelInput(x_init=x_init, conditioning=conditioning)

    def _atom37_indices_from_atom_array(
        self, atom_array
    ) -> tuple[Int[Tensor, " atoms"], Int[Tensor, " atoms"]]:
        """Derive per-atom atom37 destination indices from an Atomworks atom array.

        For each atom in ``atom_array`` (the order the sampler's flat ``x_t``
        follows), computes the residue position ``0..L-1`` and the atom37 slot
        ``0..36`` (``residue_constants.atom_order[atom_name]``). This mirrors how
        :func:`protpardelle.data.pdb_io.read_pdb` scatters atoms into the per-residue
        ``pos`` array of shape ``(37, 3)``.

        Parameters
        ----------
        atom_array : biotite.structure.AtomArray
            Single-model atom array with ``atom_name`` / ``res_id`` / ``chain_id``
            annotations (a single frame; stacks are reduced to frame 0 upstream).

        Returns
        -------
        tuple[Int[Tensor, "atoms"], Int[Tensor, "atoms"]]
            ``(atom37_residue_index, atom37_atom_index)``, both ``long`` tensors of
            length ``N`` on :attr:`device`.

        Raises
        ------
        ValueError
            If any atom name is not a recognized atom37 atom type, which would
            break the one-to-one correspondence with ``x_t``.
        """
        atom_names = np.asarray(atom_array.atom_name)
        atom37_names = np.asarray(
            [ATOM37_ATOM_NAME_ALIASES.get(str(name), str(name)) for name in atom_names]
        )

        unknown = sorted(set(atom37_names) - set(residue_constants.atom_order))
        if unknown:
            raise ValueError(
                "asym_unit contains atom names not present in the atom37 "
                f"representation: {unknown}. Protpardelle's atom37 conversion "
                "requires every atom to map to a canonical atom type."
            )

        atom_slots = np.array(
            [residue_constants.atom_order[name] for name in atom37_names], dtype=np.int64
        )

        # Contiguous residue ordinal (0..L-1) per atom, in atom order across chains.
        residue_starts = struc.get_residue_starts(atom_array)
        is_start = np.zeros(len(atom_names), dtype=np.int64)
        is_start[residue_starts] = 1
        residue_ordinals = np.cumsum(is_start) - 1

        atom37_residue_index = torch.as_tensor(
            residue_ordinals, dtype=torch.long, device=self.device
        )
        atom37_atom_index = torch.as_tensor(atom_slots, dtype=torch.long, device=self.device)
        return atom37_residue_index, atom37_atom_index

    def _convert_to_atom37(
        self,
        conditioning: ProtpardelleConditioning,
        x_t: Float[Tensor, "batch atoms 3"],
    ) -> Float[Tensor, "batch L 37 3"]:
        """Scatter flat coordinates ``x_t`` into Protpardelle's atom37 layout.

        Our sampler represents structures as ``x_t`` of shape ``batch x N x 3``
        (``N`` = number of atoms), whereas Protpardelle expects
        ``batch x L x 37 x 3`` (``L`` = residues, 37 = atom37 slots). Using the
        per-atom residue/slot maps computed in :meth:`featurize`, this places each
        atom's coordinate at ``[batch, residue, slot]`` exactly like the ``pos``
        tensor built in :func:`protpardelle.data.pdb_io.read_pdb`. Slots with no
        corresponding atom stay zero.

        The conversion is a differentiable scatter (``index_put``), so gradients
        flow from the returned tensor back to ``x_t``.

        Parameters
        ----------
        conditioning : ProtpardelleConditioning
            Conditioning from :meth:`featurize`; supplies ``L`` (via ``seq_mask``)
            and the ``atom37_residue_index`` / ``atom37_atom_index`` maps.
        x_t : Float[Tensor, "batch atoms 3"]
            Flat per-atom coordinates.

        Returns
        -------
        Float[Tensor, "batch L 37 3"]
            Coordinates in atom37 layout, on the same device/dtype as ``x_t``.

        Raises
        ------
        ValueError
            If ``x_t``'s atom count disagrees with the atom37 index maps.
        """
        batch_size, num_atoms, _ = x_t.shape
        num_residues = conditioning.seq_mask.shape[1]

        residue_index = conditioning.atom37_residue_index.to(x_t.device)
        atom_index = conditioning.atom37_atom_index.to(x_t.device)

        if residue_index.shape[0] != num_atoms:
            raise ValueError(
                f"x_t has {num_atoms} atoms but the atom37 index maps describe "
                f"{residue_index.shape[0]} atoms; these must match."
            )

        x_t_atom37 = torch.zeros(
            (batch_size, num_residues, ATOM37_NUM_ATOMS, 3),
            dtype=x_t.dtype,
            device=x_t.device,
        )

        # Flatten (batch, atom) so a single index_put scatters every coordinate.
        batch_index = torch.arange(batch_size, device=x_t.device).repeat_interleave(num_atoms)
        flat_residue_index = residue_index.repeat(batch_size)
        flat_atom_index = atom_index.repeat(batch_size)
        values = x_t.reshape(batch_size * num_atoms, 3)

        # Out-of-place index_put keeps the op differentiable w.r.t. ``values``
        # (hence ``x_t``); the freshly-zeroed destination needs no gradient.
        x_t_atom37 = x_t_atom37.index_put(
            (batch_index, flat_residue_index, flat_atom_index), values
        )
        return x_t_atom37

    def _expand_noise_level(
        self,
        t: Float[Tensor, "*batch"] | float,
        conditioning: ProtpardelleConditioning,
        dtype: torch.dtype,
    ) -> Float[Tensor, "batch L"]:
        """Convert a sampler timestep/noise scalar to Protpardelle's ``B x L`` tensor.

        Parameters
        ----------
        t : Float[Tensor, "*batch"] | float
            Sampler noise level for the current EDM step. May be a scalar or one
            value per ensemble member.
        conditioning : ProtpardelleConditioning
            Conditioning whose sequence mask defines the batch and padded length.
        dtype : torch.dtype
            Floating dtype to use for the model call.

        Returns
        -------
        Float[Tensor, "batch L"]
            Noise level broadcast across valid/padded sequence positions on the
            wrapper device.
        """
        seq_mask = conditioning.seq_mask.to(device=self.device, dtype=dtype)
        noise_level = torch.as_tensor(t, device=self.device, dtype=dtype)

        if noise_level.ndim == 0:
            return noise_level.expand_as(seq_mask).clone()

        if noise_level.ndim == 1:
            if noise_level.shape[0] != seq_mask.shape[0]:
                raise ValueError(
                    f"Noise level batch size {noise_level.shape[0]} does not match "
                    f"conditioning batch size {seq_mask.shape[0]}."
                )
            return noise_level[:, None].expand_as(seq_mask).clone()

        if noise_level.shape != seq_mask.shape:
            raise ValueError(
                f"Noise level shape {tuple(noise_level.shape)} must be scalar, "
                f"batch-sized, or match seq_mask shape {tuple(seq_mask.shape)}."
            )
        return noise_level

    def step(
        self,
        x_t: Float[Tensor, "batch atoms 3"],
        t: Float[Tensor, "*batch"] | float,
        *,
        features: GenerativeModelInput[ProtpardelleConditioning] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        """
        Prepare data for and run the forward pass of the Protpardelle model.
        To do this, it converts our coordinate representation to its "atom37" format
        which has one row per _residue_ and a coordinate for all 37 possible atom types.
        See protpardelle-1c/src/protpardelle/core/models.py:L1760
        (commit ee378400f25b801fa481028000f9060183d7fb4c on branch main)

        The returned tensor is the final all-atom prediction, flattened to the
        atoms implied by the input sequence (the ``seq_mask``  attribute in the
        conditioning).

        Parameters
        ----------
        x_t : Float[Tensor, "batch atoms 3"]
            Noisy structure at timestep :math:`t`.
        t : Float[Tensor, "*batch"] | float
            Current timestep/noise level (:math:`\hat{t}` from EDM schedule).
        features : GenerativeModelInput[ProtpardelleConditioning] | None
            Model features as returned by ``featurize``.

        Returns
        -------
        Float[Tensor, "batch atoms 3"]
            Predicted coordinates for the present atoms, one row of length
            ``atoms`` per ensemble member.
        """
        if features is None or features.conditioning is None:
            raise ValueError("features with conditioning required for step()")

        cond = features.conditioning
        x_t = x_t.to(device=self.device)

        # Our x_t is B x N x 3 (N = number of atoms); Protpardelle expects
        # B x L x 37 x 3. Scatter into the atom37 layout (gradient-preserving).
        x_t_atom37 = self._convert_to_atom37(cond, x_t)

        noise_level = self._expand_noise_level(t, cond, x_t_atom37.dtype)

        seq_mask = cond.seq_mask.to(device=self.device, dtype=x_t_atom37.dtype)
        residue_index = cond.residue_index.to(device=self.device)
        chain_index = cond.chain_index.to(device=self.device)
        struct_self_cond = cond.x_self_conditioning
        if struct_self_cond is not None:
            struct_self_cond = struct_self_cond.to(device=self.device, dtype=x_t_atom37.dtype)

        # To understand all these arguments better, you can study the (complicated)
        # Protpardelle.sample method. Hints: partial diffusion is enabled and cc.enabled = False!
        # Inside that method, the call to .forward() is at
        # https://github.com/ProteinDesignLab/protpardelle-1c/blob/ee378400f25b801fa481028000f9060183d7fb4c/src/protpardelle/core/models.py#L1766
        x0, s_logprobs, x_self_cond, s_self_cond = self.model.forward(
            noisy_coords=x_t_atom37,
            noise_level=noise_level,
            seq_mask=seq_mask,
            residue_index=residue_index,
            chain_index=chain_index,
            hotspot_mask=None,  # we don't support this yet
            struct_self_cond=(
                struct_self_cond
                if self.model.config.train.self_cond_train_prob > 0.5  # true for cc89
                else None
            ),
            struct_crop_cond=None,
            sse_cond=None,  # secondary structure conditioning, TODO maybe use?
            adj_cond=None,  # adjacency conditioning, # TODO not sure what this is for exactly
            seq_self_cond=None,  # we don't do sequence design.
            seq_crop_cond=None,
            run_mpnn_model=False,  # we don't want to do sequence design
        )

        # pass the self-conditioning to the next step by updating the features.
        # Detach it since we don't want the gradient flowing back to a previous step.
        # TODO: I wonder if we need to adjust this since we will apply additional guidance.
        features.conditioning.x_self_conditioning = x_self_cond.detach()

        # x0: [batch, L, 37, 3]. Masked-out atom slots are zeroed during
        # ai-allatom sampling, so select the correct atoms via the atom37 mask.
        # The mask is identical across the batch (single input sequence), so a
        # single 2-D mask flattens every ensemble member consistently.
        atom_mask_2d = cond.atom_mask[0].to(device=x0.device).bool()  # [L, 37]
        flat_coords = x0[:, atom_mask_2d]  # [batch, atoms, 3]

        return flat_coords

    # This just generates the input noise. When doing partial diffusion, the
    # noise is added to the input coordinates with some weight..
    def initialize_from_prior(
        self,
        batch_size: int,
        features: GenerativeModelInput[ProtpardelleConditioning] | None = None,
        *,
        shape: tuple[int, ...] | None = None,
    ) -> Float[Tensor, "batch atoms 3"]:
        """Sample reference coordinates from the prior distribution.

        Parameters
        ----------
        batch_size : int
            Number of samples to generate.
        features : GenerativeModelInput[ProtpardelleConditioning] | None
            Features as returned by :meth:`featurize`, used to infer the atom
            count when ``shape`` is not given.
        shape : tuple[int, ...] | None
            Explicit ``(num_atoms, 3)`` shape. Overrides ``features``.

        Returns
        -------
        Float[Tensor, "batch atoms 3"]
            Gaussian-initialized coordinates.

        Raises
        ------
        ValueError
            If neither ``features`` nor a valid ``shape`` is provided.
        """
        if shape is not None:
            if len(shape) != 2 or shape[1] != 3:
                raise ValueError("shape must be of the form (num_atoms, 3)")
            return torch.randn((batch_size, *shape), device=self.device)

        if features is None or features.conditioning is None:
            raise ValueError("Either features or shape must be provided to initialize_from_prior()")

        num_atoms = int(features.conditioning.atom_mask[0].sum().item())
        return torch.randn((batch_size, num_atoms, 3), device=self.device)
