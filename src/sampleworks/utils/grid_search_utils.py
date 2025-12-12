from pathlib import Path

from biotite.structure import AtomArray, stack
from biotite.structure.io import save_structure


def save_trajectory(
    trajectory, atom_array, output_dir, reward_param_mask, subdir_name, save_every=10
):
    output_dir = Path(output_dir / "trajectory" / subdir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        assert isinstance(atom_array, AtomArray)
    except AssertionError:
        atom_array = atom_array[0]

    for i, coords in enumerate(trajectory):
        ensemble_size = coords.shape[0]
        if i % save_every != 0:
            continue
        array_copy = atom_array.copy()
        array_copy = stack([array_copy] * ensemble_size)
        array_copy.coord[:, reward_param_mask] = coords.detach().numpy()  # type: ignore[reportOptionalSubscript] coords will be subscriptable
        save_structure(str(output_dir / f"trajectory_{i}.cif"), array_copy)


def save_fk_steering_trajectory(
    trajectory, atom_array, output_dir, reward_param_mask, subdir_name, save_every=10
):
    output_dir = Path(output_dir / "trajectory" / subdir_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        assert isinstance(atom_array, AtomArray)
    except AssertionError:
        atom_array = atom_array[0]

    for i, coords in enumerate(trajectory):
        ensemble_size = coords.shape[1]  # first dim is the particle dim
        if i % save_every != 0:
            continue
        array_copy = atom_array.copy()
        array_copy = stack([array_copy] * ensemble_size)
        # we save only the first ensemble out of n_particles, since saving
        # each particle at every step would clog trajectory saving
        array_copy.coord[:, reward_param_mask] = coords[0].detach().numpy()  # type: ignore[reportOptionalSubscript] coords will be subscriptable
        save_structure(output_dir / f"trajectory_{i}.cif", array_copy)


def save_losses(losses, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "losses.txt", "w") as f:
        f.write("step,loss\n")
        for i, loss in enumerate(losses):
            if loss is not None:
                f.write(f"{i},{loss}\n")
            else:
                f.write(f"{i},NA\n")


