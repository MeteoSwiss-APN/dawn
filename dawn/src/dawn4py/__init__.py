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


def _serialize_sir(sir: Union[serialization.SIR.SIR, str, bytes]):
    serializer_format = SIRSerializerFormat.Byte
    if isinstance(sir, serialization.SIR.SIR):
        sir = sir.SerializeToString()
    elif isinstance(sir, str) and sir.lstrip().startswith("{") and sir.rstrip().endswith("}"):
        serializer_format = SIRSerializerFormat.Json
    elif not isinstance(sir, bytes):
        raise ValueError("Unrecognized SIR data format")
    return sir, serializer_format


def _serialize_instantiations(stencil_instantiation_map: dict):
    # Determine serializer_format based on first stencil instantiation in the dict
    serializer_format = IIRSerializerFormat.Byte
    if len(stencil_instantiation_map) == 0:
        raise ValueError("No stencil instantiations found")
    si = list(stencil_instantiation_map.values())[0]
    if isinstance(si, str) and si.lstrip().startswith("{") and si.rstrip().endswith("}"):
        serializer_format = IIRSerializerFormat.Json

    def _serialize(si, serializer_format):
        if isinstance(si, serialization.IIR.StencilInstantiation):
            return serialization.to_bytes(si)
        else:
            return si

    return (
        {
            name: _serialize(si, serializer_format)
            for name, si in stencil_instantiation_map.items()
        },
        serializer_format,
    )


_OPTIMIZER_OPTIONS = tuple(
    name for name, value in inspect.getmembers(OptimizerOptions) if not name.startswith("__")
)

_CODEGEN_OPTIONS = tuple(
    name for name, value in inspect.getmembers(CodeGenOptions) if not name.startswith("__")
)


def compile(
    sir: Union[serialization.SIR.SIR, str, bytes],
    *,
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

    optimizer_options = {k: v for k, v in kwargs.items() if k in _OPTIMIZER_OPTIONS}
    codegen_options = {k: v for k, v in kwargs.items() if k in _CODEGEN_OPTIONS}

    sir, sir_format = _serialize_sir(sir)
    return _dawn4py.compile_sir(
        sir,
        sir_format,
        groups,
        OptimizerOptions(**optimizer_options),
        backend,
        CodeGenOptions(**codegen_options),
    )


def lower_and_optimize(
    sir: Union[serialization.SIR.SIR, str, bytes], groups: list, **kwargs,
):
    """Compile SIR to source code.
    This is a convenience function which instantiates a temporary :class:`Compiler`
    object and calls its compile method with the provided arguments.
    Parameters
    ----------
    sir:
        SIR of the stencil (in any valid serialized or non serialized form).
    groups:
        Optimizer pass groups [defaults to :func:`default_pass_groups()`]
    **kwargs
        Optional keyword arguments with specific options for the compiler (see :class:`Options`).
    Returns
    -------
    instantiation_map : `dict`
        Optimized stencil instantiations.
    """
    optimizer_options = {k: v for k, v in kwargs.items() if k in _OPTIMIZER_OPTIONS}

    sir, sir_format = _serialize_sir(sir)
    iir_map = _dawn4py.run_optimizer_sir(
        sir, sir_format, groups, OptimizerOptions(**optimizer_options)
    )
    return {
        name: serialization.from_json(string, serialization.IIR.StencilInstantiation)
        for name, string in iir_map.items()
    }


def optimize(
    instantiation_map: dict, groups: list, **kwargs,
):
    """Compile SIR to source code.
    This is a convenience function which instantiates a temporary :class:`Compiler`
    object and calls its compile method with the provided arguments.
    Parameters
    ----------
    instantiation_map:
        Stencil instantiation map (values in any valid serialized or non serialized form).
    groups:
        Optimizer pass groups [defaults to :func:`default_pass_groups()`]
    **kwargs
        Optional keyword arguments with specific options for the compiler (see :class:`Options`).
    Returns
    -------
    instantiation_map : `dict`
        Optimized stencil instantiations.
    """
    optimizer_options = {k: v for k, v in kwargs.items() if k in _OPTIMIZER_OPTIONS}

    instantiation_map, iir_format = _serialize_instantiations(instantiation_map)
    optimized_instantiations = _dawn4py.run_optimizer_iir(
        instantiation_map, iir_format, groups, OptimizerOptions(**optimizer_options)
    )
    return {
        name: serialization.from_json(string, serialization.IIR.StencilInstantiation)
        for name, string in optimized_instantiations.items()
    }


def codegen(
    instantiation_map: dict, *, backend: CodeGenBackend = CodeGenBackend.GridTools, **kwargs,
):
    """Compile SIR to source code.
    This is a convenience function which instantiates a temporary :class:`Compiler`
    object and calls its compile method with the provided arguments.
    Parameters
    ----------
    instantiation_map:
        Stencil instantiation map (values in any valid serialized or non serialized form).
    backend:
        Code generation backend [defaults to GridTools].
    **kwargs
        Optional keyword arguments with specific options for the compiler (see :class:`Options`).
    Returns
    -------
    code : `str`
        A text string with the generated code.
    """
    codegen_options = {k: v for k, v in kwargs.items() if k in _CODEGEN_OPTIONS}

    instantiation_map, iir_format = _serialize_instantiations(instantiation_map)
    return _dawn4py.run_codegen(
        instantiation_map, iir_format, backend, CodeGenOptions(**codegen_options)
    )
