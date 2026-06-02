"""Wrapper for Protpardelle-1c models.

Follows the ``StructureModelWrapper`` protocol in
``sampleworks.models.protocol`` so Protpardelle can be used interchangeably
with other structure predictors in sampling pipelines.

This wrapper targets the *sequence-conditioned* all-atom Protpardelle-1c
models (``task: "ai-allatom"``, e.g. the model defined by ``cc89.yaml``).
These models take a protein sequence as their only conditioning input and
generate an all-atom structure by running the full reverse-diffusion
trajectory internally. Because a single ``step`` call encapsulates the entire
diffusion loop (analogous to AlphaFold2 recycling being hidden inside one
forward pass), this is a ``StructureModelWrapper`` rather than a
``FlowModelWrapper``.
"""

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from atomworks.enums import ChainType
from jaxtyping import Float, Int
from loguru import logger
from protpardelle.core.models import load_model, Protpardelle
from protpardelle.data.atom import atom37_mask_from_aatype
from protpardelle.data.sequence import seq_to_aatype
from torch import Tensor

from sampleworks.models.protocol import GenerativeModelInput
from sampleworks.utils.framework_utils import match_batch


# Number of aatype tokens for the 21-token alphabet (20 canonical + X).
# Matches ``data.n_aatype_tokens`` in the Protpardelle-1c configs (e.g. cc89.yaml).
NUM_AATYPE_TOKENS = 21


# ``Protpardelle.sample`` reads many nested fields off ``conditional_cfg`` and
# ``partial_diffusion`` (even when disabled). These defaults disable all
# conditional generation / partial diffusion and mirror the structure of the
# ``conditional_cfg`` block in the Protpardelle-1c running configs (e.g.
# ``configs/running/sampling_unconditional_allatom.yaml``).
DEFAULT_CONDITIONAL_CFG: dict[str, Any] = {
    "enabled": False,
    "discontiguous_motif_assignment": {
        "enabled": False,
        "strategy": "fixed",
        "fixed_motif_pos": [],
    },
    "num_recurrence_steps": 1,
    "crop_conditional_guidance": {
        "enabled": False,
        "start": 0.0,
        "end": 2.0,
        "freq": 1,
        "freq_start": 0.0,
        "freq_end": 0.0,
        "strategy": "backbone-sidechain",
    },
    "reconstruction_guidance": {
        "enabled": False,
        "start": 0.0,
        "end": 2.0,
        "schedule": "custom",
        "max_scale": 10.0,
        "loss_weights": {"motif": 1.0},
    },
    "replacement_guidance": {
        "enabled": False,
        "start": 0.0,
        "end": 0.92,
    },
}

DEFAULT_PARTIAL_DIFFUSION: dict[str, Any] = {
    "enabled": False,
    "pdb_file_path": None,
    "num_steps": 100,
}


@dataclass(frozen=True, slots=True)
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
    sampling_kwargs : dict[str, Any]
        Keyword arguments forwarded to ``Protpardelle.sample``.
    sequences : tuple[str, ...]
        The input one-letter sequences, one per protein chain (for reference).
    """

    aatype: Int[Tensor, "batch L"]
    seq_mask: Float[Tensor, "batch L"]
    residue_index: Float[Tensor, "batch L"]
    chain_index: Float[Tensor, "batch L"]
    atom_mask: Float[Tensor, "batch L 37"]
    sampling_kwargs: dict[str, Any]
    sequences: tuple[str, ...]


@dataclass
class ProtpardelleConfig:
    """Configuration for Protpardelle featurization and sampling.

    Defaults follow the ``ai-allatom`` / ``uniform_steps`` recipe used by the
    sequence-conditioned Protpardelle-1c models (see the ``allatom_cfg`` block
    of ``configs/running/sampling_partial_diffusion_allatom.yaml`` in the
    Protpardelle-1c repository).

    Attributes
    ----------
    ensemble_size : int
        Number of structures to sample for the input sequence (batch dim).
    num_steps : int
        Number of denoising (ODE discretization) steps.
    s_churn : float
        Stochasticity: ``gamma = s_churn / num_steps`` extra noise per step.
    step_scale : float
        Inverse-temperature scale applied to the score.
    sidechain_mode : bool
        All-atom MiniMPNN side-chain co-design. Left False for ai-allatom
        sequence-conditioned models (which have no MiniMPNN).
    skip_mpnn_proportion : float
        Fraction of steps from the start to skip running MiniMPNN.
    jump_steps : bool
        Use the superposition sampling scheme (mutually exclusive with
        ``uniform_steps``).
    uniform_steps : bool
        All-atom denoising with a uniform noise-level change each step. This is
        the scheme used by ``ai-allatom`` models.
    temperature : float
        Temperature applied to aatype logits (unused when the sequence is
        fully specified, but forwarded for completeness).
    top_p : float
        Top-p truncation for aatype sampling (as above).
    extra_sampling_kwargs : dict[str, Any]
        Additional keyword overrides merged into the ``Protpardelle.sample``
        call, taking precedence over the fields above.
    """

    ensemble_size: int = 1
    num_steps: int = 500
    s_churn: float = 200.0
    step_scale: float = 1.2
    sidechain_mode: bool = False
    skip_mpnn_proportion: float = 1.0
    jump_steps: bool = False
    uniform_steps: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    extra_sampling_kwargs: dict[str, Any] = field(default_factory=dict)


def annotate_structure_for_protpardelle(
    structure: dict,
    *,
    ensemble_size: int = 1,
    num_steps: int = 500,
    s_churn: float = 200.0,
    step_scale: float = 1.2,
    sidechain_mode: bool = False,
    skip_mpnn_proportion: float = 1.0,
    jump_steps: bool = False,
    uniform_steps: bool = True,
    temperature: float = 1.0,
    top_p: float = 1.0,
    extra_sampling_kwargs: dict[str, Any] | None = None,
) -> dict:
    """Annotate an Atomworks structure with Protpardelle-specific configuration.

    Parameters
    ----------
    structure : dict
        Atomworks structure dictionary.
    ensemble_size, num_steps, s_churn, step_scale, sidechain_mode, \
    skip_mpnn_proportion, jump_steps, uniform_steps, temperature, top_p, \
    extra_sampling_kwargs
        See :class:`ProtpardelleConfig`.

    Returns
    -------
    dict
        Structure dict with a ``"_protpardelle_config"`` key added.
    """
    config = ProtpardelleConfig(
        ensemble_size=ensemble_size,
        num_steps=num_steps,
        s_churn=s_churn,
        step_scale=step_scale,
        sidechain_mode=sidechain_mode,
        skip_mpnn_proportion=skip_mpnn_proportion,
        jump_steps=jump_steps,
        uniform_steps=uniform_steps,
        temperature=temperature,
        top_p=top_p,
        extra_sampling_kwargs=extra_sampling_kwargs or {},
    )
    return {**structure, "_protpardelle_config": config}


def extract_protein_sequences(structure: dict) -> list[str]:
    """Extract canonical one-letter protein sequences from a structure.

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
        config_path: str | Path | None = None,
        checkpoint_path: str | Path | None = None,
        device: torch.device | str | None = None,
        model: Protpardelle | None = None,
    ):
        """
        Parameters
        ----------
        config_path : str | Path | None
            Path to the Protpardelle model config YAML (e.g. ``cc89.yaml``).
            Required unless ``model`` is provided.
        checkpoint_path : str | Path | None
            Path to the matching ``.pth`` checkpoint with trained weights.
            Required unless ``model`` is provided.
        device : torch.device | str | None
            Device to load the model on. When ``None``, Protpardelle picks the
            default device (CUDA if available).
        model : Protpardelle | None
            A pre-built Protpardelle model. When given, ``config_path`` and
            ``checkpoint_path`` are ignored and the model is used directly
            (useful for testing and advanced reuse).
        """
        resolved_device = torch.device(device) if device is not None else None

        if model is not None:
            self.model = model.to(resolved_device) if resolved_device is not None else model
            self.config_path = Path(config_path) if config_path is not None else None
            self.checkpoint_path = (
                Path(checkpoint_path) if checkpoint_path is not None else None
            )
        else:
            if config_path is None or checkpoint_path is None:
                raise ValueError(
                    "config_path and checkpoint_path are required when no model is provided."
                )
            self.config_path = Path(config_path).expanduser().resolve()
            self.checkpoint_path = Path(checkpoint_path).expanduser().resolve()
            logger.info(
                f"Loading Protpardelle model from {self.config_path.name} "
                f"(checkpoint {self.checkpoint_path.name})"
            )
            self.model = load_model(
                self.config_path, self.checkpoint_path, device=resolved_device
            )

        self._device = self.model.device

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

        sequences = extract_protein_sequences(structure)

        # Lay out chains exactly as Protpardelle's own sampling helper does so
        # residue/chain indexing (including inter-chain gaps) matches training.
        prot_lens_per_chain = torch.tensor(
            [[len(seq) for seq in sequences]], dtype=torch.long, device=self.device
        )
        seq_mask, residue_index, chain_index = self.model.make_seq_mask_for_sampling(
            prot_lens_per_chain=prot_lens_per_chain
        )

        # Concatenate per-chain aatypes in chain order; chains are placed
        # contiguously at the front of the padded sequence by the helper above.
        chain_aatypes = [
            seq_to_aatype(seq, num_tokens=NUM_AATYPE_TOKENS) for seq in sequences
        ]
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

        atom_mask = atom37_mask_from_aatype(aatype, seq_mask)

        sampling_kwargs = self._build_sampling_kwargs(config)
        conditioning = ProtpardelleConditioning(
            aatype=aatype,
            seq_mask=seq_mask,
            residue_index=residue_index,
            chain_index=chain_index,
            atom_mask=atom_mask,
            sampling_kwargs=sampling_kwargs,
            sequences=tuple(sequences),
        )

        # x_init is a shape-compatible reference drawn from the prior. The
        # Protpardelle sampler initializes its own noise internally, so this is
        # carried only for interface compatibility / downstream alignment.
        num_atoms = int(atom_mask[0].sum().item())
        x_init = self.initialize_from_prior(
            batch_size=ensemble_size, shape=(num_atoms, 3)
        )

        return GenerativeModelInput(x_init=x_init, conditioning=conditioning)

    @staticmethod
    def _build_sampling_kwargs(config: ProtpardelleConfig) -> dict[str, Any]:
        """Assemble the keyword arguments for ``Protpardelle.sample``."""
        sampling_kwargs: dict[str, Any] = {
            "num_steps": config.num_steps,
            "s_churn": config.s_churn,
            "step_scale": config.step_scale,
            "sidechain_mode": config.sidechain_mode,
            "skip_mpnn_proportion": config.skip_mpnn_proportion,
            "jump_steps": config.jump_steps,
            "uniform_steps": config.uniform_steps,
            "temperature": config.temperature,
            "top_p": config.top_p,
            # Disable conditional generation / partial diffusion by default;
            # sample() reads nested fields off these even when disabled.
            "conditional_cfg": copy.deepcopy(DEFAULT_CONDITIONAL_CFG),
            "partial_diffusion": copy.deepcopy(DEFAULT_PARTIAL_DIFFUSION),
        }
        sampling_kwargs.update(config.extra_sampling_kwargs)
        return sampling_kwargs

    def step(
        self,
        features: GenerativeModelInput[ProtpardelleConditioning],
    ) -> Float[Tensor, "batch atoms 3"]:
        """Run the full Protpardelle diffusion trajectory for the input sequence.

        The entire reverse-diffusion loop runs internally; the returned tensor
        is the final all-atom prediction, flattened to the atoms implied by the
        input sequence (the ``atom_mask`` in the conditioning).

        Parameters
        ----------
        features : GenerativeModelInput[ProtpardelleConditioning]
            Features as returned by :meth:`featurize`.

        Returns
        -------
        Float[Tensor, "batch atoms 3"]
            Predicted coordinates for the present atoms, one row of length
            ``atoms`` per ensemble member.
        """
        if features is None or features.conditioning is None:
            raise ValueError("features with conditioning required for step()")

        cond = features.conditioning

        with torch.no_grad():
            aux = self.model.sample(
                seq_mask=cond.seq_mask,
                residue_index=cond.residue_index,
                chain_index=cond.chain_index,
                gt_aatype=cond.aatype,
                **cond.sampling_kwargs,
            )

        # aux["x"]: [batch, L, 37, 3]. Masked-out atom slots are zeroed during
        # ai-allatom sampling, so select the present atoms via the atom37 mask.
        # The mask is identical across the batch (single input sequence), so a
        # single 2-D mask flattens every ensemble member consistently.
        coords = aux["x"]
        atom_mask_2d = cond.atom_mask[0].bool()  # [L, 37]
        flat_coords = coords[:, atom_mask_2d]  # [batch, atoms, 3]

        return flat_coords.float()

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
            raise ValueError(
                "Either features or shape must be provided to initialize_from_prior()"
            )

        num_atoms = int(features.conditioning.atom_mask[0].sum().item())
        return torch.randn((batch_size, num_atoms, 3), device=self.device)
