#!/usr/bin/python3
# -*- coding: utf-8 -*-
# ===-----------------------------------------------------------------------------*- Python -*-===##
#                          _
#                         | |
#                       __| | __ ___      ___ ___
#                      / _` |/ _` \ \ /\ / / '_  |
#                     | (_| | (_| |\ V  V /| | | |
#                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
#
#
#  This file is distributed under the MIT License (MIT).
#  See LICENSE.txt for details.
#
# ===------------------------------------------------------------------------------------------===##

"""Generator for a stencil using global indices

This program creates the HIR using the Python API of the HIR.
The code is meant as an example for high-level DSLs that could generate HIR from their own
internal IR.
The program contains two parts:
    1. construct the HIR of the example
    2. pass the HIR to the dawn compiler in order to run all optimizer passes and code generation.
       In this example the compiler is configured with the CUDA backend, therefore will code
       generate an optimized CUDA implementation.

"""

import argparse
import ctypes
import os.path
import sys
import textwrap
from ctypes import *
from optparse import OptionParser

from config import __dawn_install_module__, __dawn_install_dawnclib__
from dawn import *
from dawn import sir_printer

dawn = CDLL(__dawn_install_dawnclib__)

from google.protobuf.json_format import MessageToJson


def create_vertical_region_stmt() -> VerticalRegionDeclStmt:
    """ create a vertical region statement for the stencil
    """

    interval = make_interval(Interval.Start, Interval.End, 0, 0)

    # create the out = in[i+1] statement
    body_ast = make_ast(
        [make_assignment_stmt(make_field_access_expr("out", [0, 0, 0]), make_field_access_expr("in", [0, 0, 0]), "=")]
    )

    vertical_region_stmt = make_vertical_region_decl_stmt(body_ast, interval, VerticalRegion.Forward)
    return vertical_region_stmt


def create_boundary_correction_region(value="0", i_interval=None, j_interval=None) -> VerticalRegionDeclStmt:
    interval = make_interval(Interval.Start, Interval.End, 0, 0)
    boundary_body = make_ast(
        [
            make_assignment_stmt(
                make_field_access_expr("out", [0, 0, 0]), make_literal_access_expr(value, BuiltinType.Float), "="
            )
        ]
    )
    vertical_region_stmt = make_vertical_region_decl_stmt(
        boundary_body, interval, VerticalRegion.Forward, IRange=i_interval, JRange=j_interval
    )
    return vertical_region_stmt

    hir = sir_utils.make_sir(
        GridType.Value("Cartesian"),
        "global_indexing.cpp",
        [
            make_stencil(
                "global_indexing",
                make_ast(
                    [
                        create_vertical_region_stmt(),
                        create_boundary_correction_region(
                            value="4", i_interval=make_interval(Interval.End, Interval.End, -1, 0)
                        ),
                        create_boundary_correction_region(
                            value="8", i_interval=make_interval(Interval.Start, Interval.Start, 0, 1)
                        ),
                        create_boundary_correction_region(
                            value="6", j_interval=make_interval(Interval.End, Interval.End, -1, 0)
                        ),
                        create_boundary_correction_region(
                            value="2", j_interval=make_interval(Interval.Start, Interval.Start, 0, 1)
                        ),
                        create_boundary_correction_region(
                            value="1",
                            j_interval=make_interval(Interval.Start, Interval.Start, 0, 1),
                            i_interval=make_interval(Interval.Start, Interval.Start, 0, 1),
                        ),
                        create_boundary_correction_region(
                            value="3",
                            j_interval=make_interval(Interval.Start, Interval.Start, 0, 1),
                            i_interval=make_interval(Interval.End, Interval.End, -1, 0),
                        ),
                        create_boundary_correction_region(
                            value="7",
                            j_interval=make_interval(Interval.End, Interval.End, -1, 0),
                            i_interval=make_interval(Interval.Start, Interval.Start, 0, 1),
                        ),
                        create_boundary_correction_region(
                            value="5",
                            j_interval=make_interval(Interval.End, Interval.End, -1, 0),
                            i_interval=make_interval(Interval.End, Interval.End, -1, 0),
                        ),
                    ]
                ),
                [make_field("in"), make_field("out")],
            )
        ],
    )


parser = OptionParser()
parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False, help="print the SIR")

(options, args) = parser.parse_args()

# Print the SIR to stdout only in verbose mode
if options.verbose:
    T = textwrap.TextWrapper(initial_indent=" " * 1, width=120, subsequent_indent=" " * 1)
    des = sir_printer.SIRPrinter()

    for stencil in hir.stencils:
        des.visit_stencil(stencil)

# serialize the hir to pass it to the compiler
hirstr = hir.SerializeToString()

# create the options to control the compiler
dawn.dawnOptionsCreate.restype = c_void_p
options = dawn.dawnOptionsCreate()

# we set the backend of the compiler to cuda
dawn.dawnOptionsEntryCreateString.restype = c_void_p
dawn.dawnOptionsEntryCreateString.argtypes = [ctypes.c_char_p]

dawn.dawnOptionsSet.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
backend = dawn.dawnOptionsEntryCreateString("c++-naive".encode("utf-8"))
dawn.dawnOptionsSet(options, "Backend".encode("utf-8"), backend)

none = dawn.dawnOptionsEntryCreateString("none".encode("utf-8"))
dawn.dawnOptionsSet(options, "ReorderStrategy".encode("utf-8"), none)

# one = dawn.dawnOptionsEntryCreateInteger(1)
# dawn.dawnOptionsSet(options, "DumpStencilInstantiation".encode("utf-8"), one)

# call the compiler that generates a translation unit

dawn.dawnCompile.restype = c_void_p
dawn.dawnCompile.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
tu = dawn.dawnCompile(hirstr, len(hirstr), options)
stencilname = "global_indexing"
b_stencilName = stencilname.encode("utf-8")
# get the code of the translation unit for the given stencil
dawn.dawnTranslationUnitGetStencil.restype = c_void_p
dawn.dawnTranslationUnitGetStencil.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
code = dawn.dawnTranslationUnitGetStencil(tu, b_stencilName)

# write to file
f = open(os.path.dirname(os.path.realpath(__file__)) + "/data/global_indexing.cpp", "w")
f.write(ctypes.c_char_p(code).value.decode("utf-8"))

f.close()
