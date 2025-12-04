import torch
from jaxtyping import ArrayLike, Float, Int

from sampleworks.core.forward_models.xray.real_space_density import (
    DifferentiableTransformer,
    XMap_torch,
)
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.sf import (
    ATOM_STRUCTURE_FACTORS,
    ATOMIC_NUM_TO_ELEMENT,
    ELECTRON_SCATTERING_FACTORS,
)
from sampleworks.core.forward_models.xray.real_space_density_deps.qfit.volume import (
    XMap,
)
from sampleworks.utils.torch_utils import try_gpu


def setup_scattering_params(structure: dict, em: bool = False) -> torch.Tensor:
    """Set up scattering parameters for density calculation."""
    unique_elements = set(structure["asym_unit"].element)
    unique_elements = sorted(
        set(
            [
                (elem.upper() if len(elem) == 1 else elem[0].upper() + elem[1:].lower())
                for elem in unique_elements
            ]
        )
    )
    atomic_num_dict = {
        elem: ATOMIC_NUM_TO_ELEMENT.index(elem) for elem in unique_elements
    }

    if em:
        structure_factors = ELECTRON_SCATTERING_FACTORS
    else:
        structure_factors = ATOM_STRUCTURE_FACTORS

    max_atomic_num = max(atomic_num_dict.values())
    # Use the max atomic number found in the structure for tensor size
    n_coeffs = len(structure_factors["C"][0])
    dense_size = torch.Size([max_atomic_num + 1, n_coeffs, 2])
    scattering_dense_tensor = torch.zeros(dense_size, dtype=torch.float32)

    for elem in unique_elements:
        atomic_num = atomic_num_dict[elem]

        if elem in structure_factors:
            factor = structure_factors[elem]
        else:
            print(f"Warning: Scattering factors for {elem} not found, using C instead")
            factor = structure_factors["C"]

        factor = torch.tensor(factor, dtype=torch.float32).T  # (2, range) -> (range, 2)

        scattering_dense_tensor[atomic_num, :, :] = factor

    return scattering_dense_tensor


class RewardFunction:
    def __init__(
        self,
        xmap: XMap,
        scattering_params: torch.Tensor,
        selection: ArrayLike | torch.Tensor,
        em: bool = False,
        loss_order: int = 2,
        device: torch.device | None = None,
    ):
        """Hardcoded reward function for now, L1 or L2 reward for fitting real space
        electron density.

        TODO: decide whether these should take in structure or take in
        coords, bfactors, etc. separately. Leaning towards separate, as then
        the functions will be pure and we can do grad w.r.t. input."""

        if device is None:
            device = try_gpu()

        self.transformer = DifferentiableTransformer(
            xmap=XMap_torch(xmap, device=device),
            scattering_params=scattering_params,
            em=em,
            device=device,
        )

        # TODO: selection doesn't do anything right now
        self.selection = (
            selection
            if torch.is_tensor(selection)
            else torch.tensor(selection).to(device=device, dtype=torch.bool)
        )
        self.device = device
        if loss_order == 1:
            self.loss = torch.nn.L1Loss()
        elif loss_order == 2:
            self.loss = torch.nn.MSELoss()
        else:
            raise ValueError("Invalid loss_order, must be 1 or 2")

    def precompute_unique_combinations(
        self,
        elements: Int[torch.Tensor, "batch n_atoms"],
        b_factors: Float[torch.Tensor, "batch n_atoms"],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Pre-compute unique (element, b_factor) combinations for vmap compatibility.

        This allows torch.unique to be called outside of vmap contexts, avoiding
        the dynamic shapes.

        Parameters
        ----------
        elements: torch.Tensor
            Atomic elements, shape [batch n_atoms]
        b_factors: torch.Tensor
            Per-atom B-factors, shape [batch n_atoms]

        Returns
        -------
        tuple[torch.Tensor, torch.Tensor]
            unique_combinations: Unique (element, b_factor) pairs
            inverse_indices: Indices to reconstruct original from unique
        """
        device = self.device
        elements_flat = elements.reshape(-1)
        b_factors_flat = b_factors.reshape(-1)
        combined = torch.stack([elements_flat, b_factors_flat], dim=1)
        unique_combinations, inverse_indices = torch.unique(
            combined, dim=0, return_inverse=True
        )
        return unique_combinations.to(device), inverse_indices.to(device)

    def structure_to_reward_input(
        self, structure: dict
    ) -> dict[str, Float[torch.Tensor, "..."]]:
        atom_array = structure["asym_unit"]
        atom_array = atom_array[:, atom_array.occupancy > 0]
        elements = [
            ATOMIC_NUM_TO_ELEMENT.index(
                elem.upper() if len(elem) == 1 else elem[0].upper() + elem[1:].lower()
            )
            for elem in atom_array.element
        ]

        elements = torch.tensor(elements, device=self.device).unsqueeze(0)
        b_factors = torch.from_numpy(atom_array.b_factor).to(self.device).unsqueeze(0)
        occupancies = (
            torch.from_numpy(atom_array.occupancy).to(self.device).unsqueeze(0)
        )

        coordinates = torch.from_numpy(atom_array.coord).to(self.device)

        return {
            "coordinates": coordinates,
            "elements": elements,
            "b_factors": b_factors,
            "occupancies": occupancies,
        }

    def __call__(
        self,
        coordinates: Float[torch.Tensor, "batch n_atoms 3"],
        elements: Float[torch.Tensor, "batch n_atoms"],
        b_factors: Float[torch.Tensor, "batch n_atoms"],
        occupancies: Float[torch.Tensor, "batch n_atoms"],
        unique_combinations: torch.Tensor | None = None,
        inverse_indices: torch.Tensor | None = None,
    ) -> Float[torch.Tensor, ""]:
        """Pure function for computing reward. Call .backward() on this to get gradients
        w.r.t. input.

        Parameters
        ----------
        coordinates: Float[torch.Tensor, "batch n_atoms 3"]
            Atomic coordinates
        elements: Float[torch.Tensor, "batch n_atoms"]
            Atomic elements
        b_factors: Float[torch.Tensor, "batch n_atoms"]
            Per-atom B-factor
        occupancies: Float[torch.Tensor, "batch n_atoms"]
            Per-atom occupancies
        unique_combinations: torch.Tensor | None, optional
            Pre-computed unique (element, b_factor) pairs for vmap compatibility
        inverse_indices: torch.Tensor | None, optional
            Pre-computed inverse indices for vmap compatibility

        Returns
        -------
        torch.Tensor
            Reward value
        """
        density: torch.Tensor = self.transformer(
            coordinates=coordinates,
            elements=elements,
            b_factors=b_factors,
            occupancies=occupancies,
            unique_combinations=unique_combinations,
            inverse_indices=inverse_indices,
        ).sum(0)  # sum over batch dimension

        return self.loss(density, self.transformer.xmap.array)
