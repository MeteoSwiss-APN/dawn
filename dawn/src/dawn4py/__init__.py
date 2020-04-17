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
import inspect

from . import _dawn4py
from . import serialization
from ._dawn4py import SIRSerializerFormat, IIRSerializerFormat
from ._dawn4py import OptimizerOptions, CodeGenOptions
from ._dawn4py import PassGroup, CodeGenBackend
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


def compile(
    sir: Union[serialization.SIR.SIR, str, bytes],
    *,
    sir_format: SIRSerializerFormat = SIRSerializerFormat.Byte,
    groups: list = default_pass_groups(),
    backend: CodeGenBackend = CodeGenBackend.GridTools,
    **kwargs,
):
    """Compile SIR to source code.
    This is a convenience function which instantiates a temporary :class:`Compiler`
    object and calls its compile method with the provided arguments.
    Parameters
    ----------
    sir:
        SIR of the stencil (in any valid serialized or non serialized form).
    sir_format:
        SIR format (byte or string).
    groups:
        Optimizer pass groups [defaults to :func:`default_pass_groups()`]
    backend:
        Code generation backend (see :class:`Codegen.Backend`).
    **kwargs
        Optional keyword arguments with specific options for the compiler (see :class:`Options`).
    Returns
    -------
    code : `str`
        A text string with the generated code.
    """
    all_optimizer_options = [
        name for name, value in inspect.getmembers(OptimizerOptions) if not name.startswith("__")
    ]
    all_codegen_options = [
        name for name, value in inspect.getmembers(CodeGenOptions) if not name.startswith("__")
    ]

    optimizer_options = {k: v for k, v in kwargs.items() if k in all_optimizer_options}
    codegen_options = {k: v for k, v in kwargs.items() if k in all_codegen_options}

    return compile_sir(
        sir,
        sir_format,
        groups,
        OptimizerOptions(**optimizer_options),
        backend,
        CodeGenOptions(**codegen_options),
    )
