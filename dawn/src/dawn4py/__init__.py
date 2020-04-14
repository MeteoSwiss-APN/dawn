# -*- coding: utf-8 -*-
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


"""
Python bindings for the C++ Dawn compiler project.
"""

from typing import Any, Dict, Optional, Union

from . import _dawn4py
from . import serialization
from ._dawn4py import SIRSerializerFormat, IIRSerializerFormat
from ._dawn4py import PassGroup, CodegenBackend
from ._dawn4py import default_pass_groups
from ._dawn4py import run_optimizer_sir, run_optimizer_iir
from ._dawn4py import run_codegen, compile_sir

try:
    import os

    with open(os.path.join(os.path.dirname(__file__), "version.txt"), mode="r") as f:
        __version__ = f.read().strip("\n")
        __versioninfo__ = tuple([int(i) for i in __version__.split(".")])

except IOError:
    __version__ = ""
    __versioninfo__ = (-1, -1, -1)

finally:
    del os
