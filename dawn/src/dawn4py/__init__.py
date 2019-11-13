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


class Compiler(_dawn4py.Compiler):
    def generate(self,  sir):
        sformat = SerializerFormat.Byte
        if isinstance(sir, serialization.SIR.SIR):
            sir = sir.SerializeToString()
        elif isinstance(sir, str) and sir.startswith("{"):
            sformat = SerializerFormat.Json

        result = self.compile_to_source(sir, sformat)
        return result
