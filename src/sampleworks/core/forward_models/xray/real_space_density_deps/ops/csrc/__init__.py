import os

from torch.utils.cpp_extension import load


_this_dir = os.path.dirname(os.path.abspath(__file__))

try:
    # Try to load the compiled extension
    dilate_points_cuda = load(
        name="dilate_points_cuda",
        sources=[
            os.path.join(_this_dir, "dilate_points_ext.cpp"),
            os.path.join(_this_dir, "cuda", "dilate_points_impl.cu"),
            os.path.join(_this_dir, "cuda", "dilate_points_kernel.cu"),
        ],
        verbose=True,
    )
    CUDA_AVAILABLE = True
except Exception as e:
    print(f"CUDA extension loading failed: {e}")
    CUDA_AVAILABLE = False
