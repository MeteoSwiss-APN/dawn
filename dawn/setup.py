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

from glob import glob
from distutils.version import LooseVersion
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop

# Based on:
#   https://github.com/navdeep-G/setup.py
NAME = "dawn4py"
DESCRIPTION = "High-level DSL toolchain for geophysical fluid dynamics models."
URL = "https://github.com/MeteoSwiss-APN/dawn"
EMAIL = "gridtools@cscs.com"
AUTHOR = "MeteoSwiss / ETH Zurich / Vulcan"

# Note that DAWN_DIR below may be the version pip copies over before building
DAWN_DIR = os.path.dirname(__file__)
DAWN4PY_DIR = os.path.join(DAWN_DIR, "src", "dawn4py")

BUILD_JOBS = os.getenv("BUILD_JOBS", default=str(os.cpu_count()))

# Select protobuf version
with open(os.path.join(DAWN_DIR, "cmake", "FetchProtobuf.cmake"), "r") as f:
    text = f.read()
    m = re.search(r".*\/protocolbuffers\/protobuf\/archive\/v(?P<version>.*)(?=\.tar)+", text)
    protobuf_version = m.group("version")

# Dependencies
REQUIRED = ["attrs>=19", "black>=19.3b0", f"protobuf>={protobuf_version}", "pytest>=4.3.0"]
EXTRAS = {"dev": ["pytest"]}

# Get the Dawn version string
with open(os.path.join(DAWN_DIR, "version.txt"), mode="r") as f:
    VERSION = f.read().strip("\n")

# Add the main Dawn version file to the dawn4py package
shutil.copyfile(os.path.join(DAWN_DIR, "version.txt"), os.path.join(DAWN4PY_DIR, "version.txt"))

# Copy additional C++ headers for the generated code

for srcdir in ["driver-includes", "interface"]:
    target_path = os.path.join(DAWN4PY_DIR, "_external_src", srcdir)
    if os.path.exists(target_path):
        shutil.rmtree(target_path)
    shutil.copytree(
        os.path.join(DAWN_DIR, "src", srcdir), target_path,
    )

# Based on:
#   https://www.benjack.io/2018/02/02/python-cpp-revisited.html
#   https://gist.github.com/hovren/5b62175731433c741d07ee6f482e2936
class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        assert all(isinstance(ext, CMakeExtension) for ext in self.extensions)

        module_dir = os.getenv("DAWN4PY_MODULE_DIR", default=os.path.join(DAWN_DIR, "src"))

        # Dest dir is here
        dest_dir = os.path.join(DAWN_DIR, "src") if self.inplace else self.build_lib

        irs_in_module = glob(
            os.path.join(module_dir, "dawn4py", "serialization") + "/**/*_pb2.py", recursive=True,
        )
        has_irs_in_module = len(irs_in_module) > 0

        exts_in_module = [
            os.path.join(module_dir, self.get_ext_filename(ext.name)) for ext in self.extensions
        ]
        has_exts_in_module = all(os.path.exists(e) for e in exts_in_module)

        # Check if the extensions and python protobuf files exist in build_dir
        if has_irs_in_module and has_exts_in_module:
            # Copy irs over to dest_dir
            for proto in irs_in_module:
                rel_path = proto.replace(module_dir + "/", "")
                self.copy_file(proto, os.path.join(dest_dir, rel_path))

            # Copy extension over to dest_dir
            for extension in exts_in_module:
                rel_path = extension.replace(module_dir + "/", "")
                self.copy_file(extension, os.path.join(dest_dir, rel_path))

        else:
            # Otherwise, build extension, copying protos over in the process

            # Activate ccache
            # Taken from: https://github.com/h5py/h5py/pull/1382
            # This allows ccache to recognise the files when pip builds in a temp
            # directory. It speeds up repeatedly running tests through tox with
            # ccache configured (CC="ccache gcc"). It should have no effect if
            # ccache is not in use.
            os.environ["CCACHE_BASEDIR"] = DAWN_DIR
            os.environ["CCACHE_NOHASHDIR"] = "1"

            cmake_executable = self.validate_cmake_install(self.extensions)
            self.compile_extension(self.build_temp, cmake=cmake_executable)

    def compile_extension(self, build_dir, cmake="cmake"):
        if os.path.exists(build_dir):
            shutil.rmtree(build_dir, ignore_errors=False)
        cmake_args = os.getenv("CMAKE_ARGS", default="").split(" ") or []

        os.makedirs(build_dir, exist_ok=True)

        # Run CMake configure
        print("-" * 10, "Running CMake prepare", "-" * 40)
        cmake_args += [
            "-DBUILD_TESTING=OFF",
            "-DDAWN_REQUIRE_PYTHON=ON",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DPython3_EXECUTABLE={sys.executable}",
        ]
        if not self.inplace:
            cmake_args.append("-DDAWN4PY_MODULE_DIR=" + self.build_lib)
        self.spawn([cmake, "-S", os.path.abspath(DAWN_DIR), "-B", build_dir] + cmake_args)

        # Run CMake build
        print("-" * 10, "Building extensions", "-" * 40)
        cfg = "Debug" if self.debug else "Release"
        build_args = ["--config", cfg, "-j", BUILD_JOBS]
        self.spawn([cmake, "--build", build_dir, "--target", "python"] + build_args)

    @staticmethod
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
    cmdclass={"build_ext": CMakeBuild},
    install_requires=REQUIRED,
    extras_require=EXTRAS,
    zip_safe=False,
)
