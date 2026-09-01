from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


ROOT = Path(__file__).parent


# 把C++注册代码和CUDA Kernel编译、链接成一个动态库（.so）。
setup(
    name="stage13_gemm_ext",
    ext_modules=[
        CUDAExtension(
            name="stage13_gemm_ext",
            # 两个源文件最终进入同一个扩展动态库。
            sources=[
                str(ROOT / "gemm_extension.cpp"),
                str(ROOT / "gemm_extension_cuda.cu"),
            ],
            # C++与CUDA均使用O3；lineinfo保留Profiler所需的源码行信息。
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "-lineinfo"],
            },
        )
    ],
    # 使用Ninja并行执行PyTorch生成的实际编译命令。
    cmdclass={
        "build_ext": BuildExtension.with_options(
            use_ninja=True
        )
    },
)
