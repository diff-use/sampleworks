import torch
from jaxtyping import ArrayLike, Float

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
from sampleworks.utils.setup import try_gpu


def setup_scattering_params(structure: dict, em: bool = False):
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
        selection: ArrayLike,
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

        self.selection = selection
        self.device = device
        if loss_order == 1:
            self.loss = torch.nn.L1Loss()
        elif loss_order == 2:
            self.loss = torch.nn.MSELoss()
        else:
            raise ValueError("Invalid loss_order, must be 1 or 2")

    def structure_to_reward_input(
        self, structure: dict
    ) -> dict[str, Float[torch.Tensor, "..."]]:
        atom_array = structure["asym_unit"]
        atom_array = atom_array[:, atom_array.occupancy > 0]

        elements = torch.from_numpy(atom_array.element).to(self.device)
        b_factors = torch.from_numpy(atom_array.b_factor).to(self.device)
        occupancies = torch.from_numpy(atom_array.occupancy).to(self.device)

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
    ) -> Float[torch.Tensor, ""]:
        """Pure function for computing reward. Call .backward() on this to get gradients
        w.r.t. input.

        Parameters
        ----------
        coordinates : Float[torch.Tensor, "batch n_atoms 3"]
            Atomic coordinates
        elements : Float[torch.Tensor, "batch n_atoms"]
            Atomic elements
        b_factors : Float[torch.Tensor, "batch n_atoms"]
            Per-atom B-factor
        occupancies : Float[torch.Tensor, "batch n_atoms"]
            Per-atom occupancies

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
        ).sum(0)  # sum over batch dimension

        return self.loss(density, self.transformer.xmap.array)
