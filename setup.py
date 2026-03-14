import os
import sys

import pybind11
from setuptools import Extension, setup

extra_args = ["-std=c++17", "-O3"]
if sys.platform == "win32":
    extra_args = ["/std:c++17", "/O2"]

# Base directory for C++ sources
src_dir = os.path.join("src", "triango_ext")

ext_modules = [
    Extension(
        "triango_ext",
        [os.path.join(src_dir, "bindings.cpp"), os.path.join(src_dir, "env.cpp"), os.path.join(src_dir, "mcts.cpp")],
        include_dirs=[pybind11.get_include()],
        language="c++",
        extra_compile_args=extra_args,
    ),
]

setup(
    ext_modules=ext_modules,
)
