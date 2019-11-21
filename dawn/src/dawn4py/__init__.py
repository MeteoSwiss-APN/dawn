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

from . import _dawn4py
from ._dawn4py import Options, SerializerFormat, TranslationUnit
from ._version import *
from . import serialization


def compile(sir):
    return Compiler().compile(sir)


def compile_to_source(sir):
    return Compiler().compile_to_source(sir)


class Compiler(_dawn4py.Compiler):
    def _serialize_sir(self, sir):
        serializer_format = SerializerFormat.Byte
        if isinstance(sir, serialization.SIR.SIR):
            sir = sir.SerializeToString()
        elif isinstance(sir, str) and sir.lstrip().startswith("{") and sir.rstrip().endswith("}"):
            serializer_format = SerializerFormat.Json
        elif not isinstance(sir, bytes):
            raise ValueError(f"Unrecognized SIR data format ({sir})")

        return sir, serializer_format

    def compile(self, sir):
        sir, serializer_format = self._serialize_sir(sir)
        result = super().compile(sir, serializer_format)
        return result

    def compile_to_source(self, sir):
        sir, serializer_format = self._serialize_sir(sir)
        result = super().compile_to_source(sir, serializer_format)
        return result
