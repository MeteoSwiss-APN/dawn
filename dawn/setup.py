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

from distutils.version import LooseVersion
from setuptools import setup, find_packages, Command, Extension
from setuptools.command.build_ext import build_ext


DAWN_DIR = os.path.join(os.path.dirname(__file__))
DAWN_ABS_DIR = os.path.abspath(DAWN_DIR)
BUILD_JOBS = 4

# Select protobuf version
# TODO: avoid parsing python files and adapt to new CMake
with open(os.path.join(DAWN_DIR, "cmake", "FetchProtobuf.cmake"), "r") as f:
    text = f.read()
    m = re.search(r".*\/protocolbuffers\/protobuf\/archive\/v(?P<version>.*)(?=\.tar)+", text)
    protobuf_version = m.group("version")

install_requires = ["attrs>=19", "black>=19.3b0", f"protobuf>={protobuf_version}", "pytest>=4.3.0"]


def validate_cmake_install():
    """Return a cmake executable or \"cmake\" after checking version.
    Raises an exception if cmake is not found."""
    # Check if a recent version of CMake is present
    cmake_executable = os.getenv("CMAKE_EXECUTABLE", default="cmake")
    try:
        out = subprocess.check_output([cmake_executable, "--version"])
    except OSError:
        raise RuntimeError(
            "CMake must be installed to build the following extensions: "
            + ", ".join(e.name for e in self.extensions)
        )

    cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)", out.decode()).group(1))
    if cmake_version < "3.13.0":
        raise RuntimeError("CMake >= 3.13.0 is required")

    return cmake_executable


# Based on:
#   https://www.benjack.io/2018/02/02/python-cpp-revisited.html
#   https://gist.github.com/hovren/5b62175731433c741d07ee6f482e2936
class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        assert all(isinstance(ext, CMakeExtension) for ext in self.extensions)

        cmake_executable = validate_cmake_install()

        # Check if all extensions are already built in a build directory given by DAWN_BUILD_DIR
        built = False
        dawn_build_dir = os.getenv("DAWN_BUILD_DIR", default=None)
        if dawn_build_dir:
            built = all(
                [
                    os.path.exists(
                        os.path.join(dawn_build_dir, "src", self.get_ext_filename(ext.name))
                    )
                    for ext in self.extensions
                ]
            )

        if built:
            for ext in self.extensions:
                self.copy_file(
                    os.path.join(dawn_build_dir, "src", self.get_ext_filename(ext.name)),
                    self.get_ext_fullpath(ext.name),
                )
        else:
            self.build_temp = os.path.join(DAWN_DIR, "build")
            self.compile_extension(self.build_temp, cmake=cmake_executable)
            # Move from build temp to final position
            for ext in self.extensions:
                self.copy_file(
                    os.path.join(self.build_temp, "src", self.get_ext_filename(ext.name)),
                    self.get_ext_fullpath(ext.name),
                )

        # Install included headers
        self.run_command("install_dawn_includes")

    def compile_extension(self, build_dir, cmake="cmake"):
        cmake_args = os.getenv("CMAKE_ARGS", default="").split(" ") or []
        # Build dawn here

        # Set build folder inside dawn and remove CMake cache if it contains wrong paths.
        # Installing in editable/develop mode builds the extension in the original build path,
        # but a regular `pip install` copies the full tree to a temporary folder
        # before building, which makes CMake fail if a CMake cache had been already generated.
        cmake_cache_file = os.path.join(build_dir, "CMakeCache.txt")
        if os.path.exists(cmake_cache_file):
            with open(cmake_cache_file, "r") as f:
                text = f.read()
                m = re.search(r"\s*Dawn_BINARY_DIR\s*:\s*STATIC\s*=\s*([\w/\\]+)\s*", text)
                cache_build_dir = m.group(1) if m else ""
                if str(build_dir) != cache_build_dir:
                    shutil.rmtree(build_dir, ignore_errors=False)
                    shutil.rmtree(os.path.join(DAWN_DIR, "install"), ignore_errors=True)
                    assert not os.path.exists(cmake_cache_file)
        os.makedirs(build_dir, exist_ok=True)
        os.makedirs(os.path.join(DAWN_DIR, "install"), exist_ok=True)

        # Prepare CMake arguments
        cmake_args += ["-DPYTHON_EXECUTABLE=" + sys.executable, "-DBUILD_TESTING=False"]

        cfg = "Debug" if self.debug else "Release"
        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]

        # Run CMake configure
        print("-" * 10, "Running CMake prepare", "-" * 40)
        cmake_cmd = ["cmake", "-S", DAWN_ABS_DIR, "-B", build_dir] + cmake_args
        print("{cwd} $ {cmd}".format(cwd=build_dir, cmd=" ".join(cmake_cmd)))
        subprocess.check_call(cmake_cmd)

        # Run CMake build
        print("-" * 10, "Building extensions", "-" * 40)
        build_args = ["--config", cfg, "-j", str(BUILD_JOBS)]
        cmake_cmd = ["cmake", "--build", build_dir, "--target", "python"] + build_args
        print("{cwd} $ {cmd}".format(cwd=build_dir, cmd=" ".join(cmake_cmd)))
        subprocess.check_call(cmake_cmd)


class InstallDawnIncludesCommand(Command):
    """A custom command to install in the Python package the Dawn C++ headers for generated code."""

    TARGET_SRC_PATH = os.path.join(DAWN_DIR, "src", "dawn4py", "_external_src")

    description = "Install Dawn C++ headers for generated code"
    user_options = []  # (long option, short option, description)

    def initialize_options(self):
        """Set default values for user options."""
        pass

    def finalize_options(self):
        """Post-process options."""
        pass

    def run(self):
        """Run command."""
        # Always copy dawn include folder to dawn4py/_external_src to install newest sources
        target_path = os.path.join(self.TARGET_SRC_PATH, "driver-includes")
        if os.path.exists(target_path):
            shutil.rmtree(target_path, ignore_errors=True)
        shutil.copytree(
            os.path.join(DAWN_DIR, "src", "driver-includes"), target_path,
        )


with open(os.path.join(DAWN_DIR, "version.txt"), mode="r") as f:
    version = f.read().strip("\n")

setup(
    name="dawn4py",
    version=version,
    author="MeteoSwiss / ETH Zurich",
    author_email="gridtools@cscs.com",
    description="High-level DSL toolchain for geophysical fluid dynamics models",
    long_description="",
    include_package_data=True,
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[CMakeExtension("dawn4py._dawn4py")],
    cmdclass={"build_ext": CMakeBuild, "install_dawn_includes": InstallDawnIncludesCommand},
    install_requires=install_requires,
    extras_require={"dev": ["Jinja2", "pytest", "tox"]},
    zip_safe=False,
)
