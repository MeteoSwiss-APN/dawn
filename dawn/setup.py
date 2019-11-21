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
BUNDLE_PREFIX = "bundle"
BUNDLE_DIR = os.path.join(DAWN_DIR, BUNDLE_PREFIX)
BUNDLE_ABS_DIR = os.path.abspath(BUNDLE_DIR)

BUILD_JOBS = 4


# Select protobuf version
with open(os.path.join(DAWN_DIR, "cmake", "thirdparty", "DawnAddProtobuf.cmake"), "r") as f:
    text = f.read()
    m = re.search(r"set\(protobuf_version\s+\"([0-9\.]+)\s*\"\)", text)
    protobuf_version = m.group(1)

install_requires = ["attrs>=19", "black>=19.3b0", f"protobuf=={protobuf_version}", "pytest>=4.3.0"]


# Based on:
#   https://www.benjack.io/2018/02/02/python-cpp-revisited.html
#   https://gist.github.com/hovren/5b62175731433c741d07ee6f482e2936
class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(["cmake", "--version"])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: "
                + ", ".join(e.name for e in self.extensions)
            )

        cmake_version = LooseVersion(re.search(r"version\s*([\d.]+)", out.decode()).group(1))
        if cmake_version < "3.12.0":
            raise RuntimeError("CMake >= 3.12.0 is required")

        cmake_args = [
            "-DPYTHON_EXECUTABLE=" + sys.executable,
            "-DUSE_SYSTEM_DAWN=False",
            "-DUSE_SYSTEM_PROTOBUF=False",
            "-DDAWN_BUNDLE_PYTHON=True",
            "-DBUILD_TESTING=False",
        ]

        cfg = "Debug" if self.debug else "Release"
        cmake_args += ["-DCMAKE_BUILD_TYPE=" + cfg]

        env = os.environ.copy()
        env["CXXFLAGS"] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get("CXXFLAGS", ""), self.distribution.get_version()
        )
        # Set build folder inside bundle and remove CMake cache if it contains wrong paths.
        # Installing in editable/develop mode builds the extension in the original build path,
        # but a regular `pip install` copies the full tree to a temporary folder
        # before building, which makes CMake fail if a CMake cache had been already generated.
        self.bundle_build_temp = str(os.path.join(BUNDLE_ABS_DIR, "build"))
        cmake_cache_file = os.path.join(self.bundle_build_temp, "CMakeCache.txt")
        if os.path.exists(cmake_cache_file):
            with open(cmake_cache_file, "r") as f:
                text = f.read()
                m = re.search(r"\s*dawn_BINARY_DIR\s*:\s*STATIC\s*=\s*([\w/\\]+)\s*", text)
                cache_build_dir = m.group(1) if m else ""
                if str(self.bundle_build_temp) != cache_build_dir:
                    shutil.rmtree(self.bundle_build_temp, ignore_errors=False)
                    shutil.rmtree(os.path.join(BUNDLE_ABS_DIR, "install"), ignore_errors=True)
                    assert not os.path.exists(cmake_cache_file)
        os.makedirs(self.bundle_build_temp, exist_ok=True)

        print("-" * 10, "Running CMake prepare", "-" * 40)
        cmake_cmd = ["cmake", BUNDLE_ABS_DIR] + cmake_args
        print("{cwd} $ {cmd}".format(cwd=self.bundle_build_temp, cmd=" ".join(cmake_cmd)))
        subprocess.check_call(cmake_cmd, cwd=self.bundle_build_temp, env=env)
        print("-" * 10, "Building extensions", "-" * 40)
        build_args = ["--config", cfg, "-j", str(BUILD_JOBS), "--target", "dawn"]
        cmake_cmd = ["cmake", "--build", "."] + build_args
        print("{cwd} $ {cmd}".format(cwd=self.bundle_build_temp, cmd=" ".join(cmake_cmd)))
        subprocess.check_call(cmake_cmd, cwd=self.bundle_build_temp)

        # Move from build temp to final position
        for ext in self.extensions:
            self.install_ext(ext)

    def install_ext(self, ext):
        filename = self.get_ext_filename(ext.name)
        source_path = os.path.abspath(
            os.path.join(
                self.bundle_build_temp, "dawn-prefix", "src", "dawn-build", "src", filename
            )
        )
        dest_build_path = os.path.abspath(self.get_ext_fullpath(ext.name))
        self.copy_file(source_path, dest_build_path)


setup(
    name="dawn4py",
    version="0.0.1",  # TODO: automatic update of version tag
    author="MeteoSwiss / ETH Zurich",
    author_email="gridtools@cscs.com",
    description="High-level DSLs toolchain for geophysical fluid dynamics models",
    long_description="",
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=[CMakeExtension("dawn4py._dawn4py")],
    cmdclass={"build_ext": CMakeBuild},
    install_requires=install_requires,
    zip_safe=False,
)
