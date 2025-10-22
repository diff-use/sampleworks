# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="dilate_points_cuda",
    ext_modules=[
        CUDAExtension(
            "dilate_points_cuda",
            [
                "dilate_points_ext.cpp",
                "cuda/dilate_points_impl.cu",
                "cuda/dilate_points_kernel.cu",
            ],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
