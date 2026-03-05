import glob
import os
import shutil
import subprocess
import sys

from setuptools import setup, find_packages, Command
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_nvcc_flags():
    return [
        "-O3",
        "--use_fast_math",
        "-lineinfo",

        "-Xcompiler", "/FS",
        "-Xcompiler", "/bigobj",
    ]


def run_codegen():
    subprocess.check_call([sys.executable, "src/gen_gemm_bank.py"])


class BuildExtWithCodegen(BuildExtension):
    def run(self):
        run_codegen()

        srcs = collect_sources()

        for ext in self.extensions:
            if getattr(ext, "name", "") == "infinite_dreams.infinite_dreams_ext":
                ext.sources = srcs

        super().run()


def collect_sources():
    print("=== Collecting sources ===")
    sources = [
        "src/gemm_ext.cpp",
        "src/gemm_cuda.cu",
    ]
    sources += sorted(glob.glob("src/generated/gemm_bank_*.cu"))
    return sources


class CleanCommand(Command):
    description = "Remove build artifacts"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        paths_to_remove = [
            "build",
            "dist",
            "infinite_dreams.egg-info",
            "src/generated",
            "__pycache__",
        ]

        for path in paths_to_remove:
            if os.path.exists(path):
                print(f"Removing {path}")
                shutil.rmtree(path, ignore_errors=True)

        for ext in glob.glob("**/*.pyd", recursive=True):
            print(f"Removing {ext}")
            os.remove(ext)

        for ext in glob.glob("**/*.so", recursive=True):
            print(f"Removing {ext}")
            os.remove(ext)


setup(
    name="infinite_dreams",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="infinite_dreams.infinite_dreams_ext",
            sources=[],
            extra_compile_args={
                "cxx": ["/O3", "/std:c++17"],
                "nvcc": get_nvcc_flags(),
            },
        )
    ],
    cmdclass={
        "build_ext": BuildExtWithCodegen,
        "clean": CleanCommand,
    },
)
