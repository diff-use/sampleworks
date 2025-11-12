"""
Utility functions for working with PyTorch tensors and devices.
"""

import torch


DeviceLikeType = str | torch.device | int


def try_gpu():
    """Attempt to select the free-est GPU (by memory) for use with PyTorch.

    Returns
    -------
    torch.device
        GPU with most available memory, or the CPU if no GPUs are available.
    """
    import os
    import subprocess
    from io import StringIO

    import pandas as pd

    try:
        gpu_stats = subprocess.check_output(
            ["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]
        )
        gpu_stats_str = gpu_stats.decode("utf-8")
        gpu_df = pd.read_csv(
            StringIO(gpu_stats_str), names=["memory.used", "memory.free"], skiprows=1
        )
        print(f"GPU usage:\n{gpu_df}")
        gpu_df["memory.free"] = gpu_df["memory.free"].map(
            lambda x: int(x.rstrip(" [MiB]"))
        )
        if gpu_df.empty:
            print("No GPUs found.")
            return torch.device("cpu")
        idx = gpu_df["memory.free"].idxmax()

        available_gpus = os.environ.get("CUDA_VISIBLE_DEVICES", None)
        if available_gpus is not None:
            available_gpus = [int(gpu) for gpu in available_gpus.split(",")]
            if idx not in available_gpus:
                print(
                    f"Selected GPU {idx} is not in CUDA_VISIBLE_DEVICES: \
                    {available_gpus}. Selecting first available GPU: \
                    {available_gpus[0]}."
                )
                return torch.device("cuda:0")

        print(
            "Returning GPU{} with {} free MiB".format(
                idx,
                gpu_df.iloc[idx]["memory.free"],  # type: ignore (index will be convertible to int)
            )
        )

        return torch.device(f"cuda:{idx}")

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Failed to run nvidia-smi: {e}")
        return torch.device("cpu")


def send_tensors_in_dict_to_device(
    d: dict, device: DeviceLikeType, inplace: bool = True
) -> dict:
    """Recursively sends all torch.Tensors in a dictionary to a specified device.

    Parameters
    ----------
    d : dict
        The input dictionary potentially containing torch.Tensors.
    device : str | torch.device | int
        The target device to send the tensors to (e.g., 'cpu', 'cuda:0').
    inplace : bool, optional
        If True, modifies the input dictionary in place. If False,
        returns a new dictionary.
        Default is True.

    Returns
    -------
    dict
        The dictionary with all torch.Tensors sent to the specified device.
    """
    result = d if inplace else {}
    for key, value in d.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device)
        elif isinstance(value, dict):
            result[key] = send_tensors_in_dict_to_device(
                value, device=device, inplace=inplace
            )
        elif isinstance(value, list | tuple):
            result[key] = type(value)(
                send_tensors_in_dict_to_device(item, device=device, inplace=inplace)
                if isinstance(item, dict)
                else item.to(device)
                if isinstance(item, torch.Tensor)
                else item
                for item in value
            )
        else:
            result[key] = value
    return result
