##===-----------------------------------------------------------------------------*- Python -*-===##
##                          _
##                         | |
##                       __| | __ ___      ___ ___
##                      / _` |/ _` \ \ /\ / / '_  |
##                     | (_| | (_| |\ V  V /| | | |
##                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
##
##
##  This file is distributed under the MIT License (MIT).
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

import os
import re
import shutil
import subprocess
import sys
import glob

from distutils.version import LooseVersion
from setuptools import setup, find_packages, Command, Extension
from setuptools.command.build_ext import build_ext

# Based on:
#   https://github.com/navdeep-G/setup.py
NAME = "dawn4py"
DESCRIPTION = "High-level DSL toolchain for geophysical fluid dynamics models."
URL = "https://github.com/MeteoSwiss-APN/dawn"
EMAIL = "gridtools@cscs.com"
AUTHOR = "MeteoSwiss / ETH Zurich / Vulcan"

# Note that DAWN_DIR below may be the version pip copies over before building
DAWN_DIR = os.path.join(os.path.dirname(__file__))
BUILD_JOBS = 4

# Select protobuf version
with open(os.path.join(DAWN_DIR, "cmake", "FetchProtobuf.cmake"), "r") as f:
    text = f.read()
    m = re.search(r".*\/protocolbuffers\/protobuf\/archive\/v(?P<version>.*)(?=\.tar)+", text)
    protobuf_version = m.group("version")

# Dependencies
REQUIRED = ["attrs>=19", "black>=19.3b0", f"protobuf>={protobuf_version}", "pytest>=4.3.0"]
EXTRAS = {"dev": ["Jinja2", "pytest", "tox"]}

# Get the Dawn version string
with open(os.path.join(DAWN_DIR, "version.txt"), mode="r") as f:
    VERSION = f.read().strip("\n")


def validate_cmake_install(extensions):
    """Return a cmake executable or \"cmake\" after checking version.
    Raises an exception if cmake is not found."""
    # Check if a recent version of CMake is present
    cmake_executable = os.getenv("CMAKE_EXECUTABLE", default="cmake")
    try:
        out = subprocess.check_output([cmake_executable, "--version"])
    except OSError:
        raise RuntimeError(
            "CMake must be installed to build the following extensions: "
            + ", ".join(e.name for e in extensions)
        )

    cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)", out.decode()).group(1))
    if cmake_version < "3.13.0":
        raise RuntimeError("CMake >= 3.13.0 is required")

    return cmake_executable


def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result


# Based on:
#   https://www.benjack.io/2018/02/02/python-cpp-revisited.html
#   https://gist.github.com/hovren/5b62175731433c741d07ee6f482e2936
class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        assert all(isinstance(ext, CMakeExtension) for ext in self.extensions)

        dawn_build_dir = os.getenv("DAWN_BUILD_DIR", default="")
        has_irs_in_build_lib = (
            len(
                glob.glob(
                    os.path.join(self.build_lib, "dawn4py", "serialization") + "/**/*_pb2.py",
                    recursive=True,
                )
            )
            > 0
        )

        # Check if the extensions exist in the build dir and protos were copied over
        if (
            dawn_build_dir
            and has_irs_in_build_lib
            and all(
                [
                    os.path.exists(
                        os.path.join(dawn_build_dir, "src", self.get_ext_filename(ext.name))
                    )
                    for ext in self.extensions
                ]
            )
        ):
            # All that we need to do is copy over the library
            for ext in self.extensions:
                self.copy_file(
                    os.path.join(dawn_build_dir, "src", self.get_ext_filename(ext.name)),
                    os.path.join(self.build_lib, self.get_ext_filename(ext.name)),
                )

        else:
            # Otherwise, build extension, copying protos over in the process
            cmake_executable = validate_cmake_install(self.extensions)
            self.compile_extension(self.build_temp, cmake=cmake_executable)
            # Move from build_tmp to final position
            for ext in self.extensions:
                self.copy_file(
                    os.path.join(self.build_temp, "src", self.get_ext_filename(ext.name)),
                    os.path.join(self.build_lib, self.get_ext_filename(ext.name)),
                )

        # Install included headers
        self.run_command("install_dawn_includes")

    def compile_extension(self, build_dir, cmake="cmake"):
        cmake_args = os.getenv("CMAKE_ARGS", default="").split(" ") or []

        # Set build folder inside dawn and remove CMake cache if it contains wrong paths.
        # Installing in editable/develop mode builds the extension in the original build path,
        # but a regular `pip install` copies the full tree to a temporary folder
        # before building, which makes CMake fail if a CMake cache had been already generated.
        for cache_file in find_all("CMakeCache.txt", build_dir):
            with open(cache_file, "r") as f:
                text = f.read()
                m = re.search(r"\s*Dawn_BINARY_DIR\s*:\s*STATIC\s*=\s*([\w/\\]+)\s*", text)
                cache_build_dir = m.group(1) if m else ""
                if os.path.dirname(cache_file) != cache_build_dir:
                    shutil.rmtree(os.path.dirname(cache_file), ignore_errors=False)
                    assert not os.path.exists(cache_file)
        os.makedirs(build_dir, exist_ok=True)

        # Run CMake configure
        print("-" * 10, "Running CMake prepare", "-" * 40)
        cmake_args += [
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DBUILD_TESTING=OFF",
            "-DDAWN_REQUIRE_PYTHON=ON",
            "-DDAWN4PY_MODULE_DIR=" + self.build_lib,
        ]
        self.spawn([cmake, "-S", os.path.abspath(DAWN_DIR), "-B", build_dir] + cmake_args)

        # Run CMake build
        print("-" * 10, "Building extensions", "-" * 40)
        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg, "-j", str(BUILD_JOBS)]
        self.spawn([cmake, "--build", build_dir, "--target", "python"] + build_args)


class InstallDawnIncludesCommand(build_ext):
    """A custom command to install in the Python package the Dawn C++ headers for generated code."""

    def run(self):
        """Run command."""
        self.copy_tree(
            os.path.join(DAWN_DIR, "src", "driver-includes"),
            os.path.join(self.build_lib, "dawn4py", "_external_src", "driver-includes"),
        )


setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    description=DESCRIPTION,
    include_package_data=True,
    license="MIT",
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[CMakeExtension("dawn4py._dawn4py")],
    cmdclass={"build_ext": CMakeBuild, "install_dawn_includes": InstallDawnIncludesCommand},
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    zip_safe=False,
)
