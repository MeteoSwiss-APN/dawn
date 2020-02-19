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
from ._dawn4py import Options, SerializerFormat
from ._version import __version__


def compile(
    sir: Union[serialization.SIR.SIR, str, bytes],
    *,
    unit_info: Optional[Dict[str, Any]] = None,
    **kwargs,
):
    """Compile SIR to source code.

    This is a convenience function which instantiates a temporary :class:`Compiler`
    object and calls its compile method with the provided arguments.

    Parameters
    ----------
    sir :
        SIR of the stencil (in any valid serialized or non serialized form).
    unit_info :
        Optional dictionary to store the string components of the generate translation unit.
    **kwargs
        Optional keyword arguments with specific options for the compiler (see :class:`Options`).

    Returns
    -------
    code : `str`
        A text string with the generated code.
    """

    options = Options(**kwargs)
    return Compiler(options).compile(sir, unit_info=unit_info)


class Compiler(_dawn4py.Compiler):
    """SIR Compiler instance.

    This is a Python wrapper around the actual bindings of the Dawn compiler
    (inside `_dawnpy` module) which provides a slightly more convenient interface.

    Parameters
    ----------
    options : :class:`Options`
        Compiler options (can not be modified after instantiation).
    """

    def _serialize_sir(self, sir: Union[serialization.SIR.SIR, str, bytes]):
        serializer_format = SerializerFormat.Byte
        if isinstance(sir, serialization.SIR.SIR):
            sir = sir.SerializeToString()
        elif isinstance(sir, str) and sir.lstrip().startswith("{") and sir.rstrip().endswith("}"):
            serializer_format = SerializerFormat.Json
        elif not isinstance(sir, bytes):
            raise ValueError(f"Unrecognized SIR data format ({sir})")

        return sir, serializer_format

    def compile(
        self, sir: Union[serialization.SIR.SIR, str, bytes], *, unit_info: Dict[str, Any] = None
    ):
        """Compile SIR to source code.

        Parameters
        ----------
        sir :
            SIR of the stencil (in any valid serialized or non serialized form).
        unit_info :
            Optional dictionary to store the string components of the generate translation unit.
        **kwargs
            Optional keyword arguments with specific options for the compiler (see :class:`Options`).

        Returns
        -------
        code : `str`
            A text string with the generated code.
        """
        sir, serializer_format = self._serialize_sir(sir)
        result = super().compile(sir, serializer_format, unit_info)
        return result
