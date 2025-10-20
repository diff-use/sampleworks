"""
Various utility functions useful when initializing classes and modules from Sampleworks.
"""


class DotDict(dict):
    """
    A dictionary subclass that supports attribute-style access.

    This class allows you to access dictionary keys as if they were attributes.
    For example, instead of writing `d['key']`, you can write `d.key`.

    Example usage:
        d = DotDict()
        d.key = 'value'
        print(d.key)  # Output: value
        print(d['key'])  # Output: value

    From: https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    """

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


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
    import torch

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
                idx, gpu_df.iloc[idx]["memory.free"]
            )
        )

        return torch.device(f"cuda:{idx}")

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"Failed to run nvidia-smi: {e}")
        return torch.device("cpu")
