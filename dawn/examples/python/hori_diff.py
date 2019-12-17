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
"""Horizontal diffusion stencil HIR generator

This program creates the HIR corresponding to an horizontal diffusion stencil using the Python API of the HIR.
The horizontal diffusion is a basic example that contains horizontal data dependencies that need to be resolved
by the compiler passes.
The code is meant as an example for high-level DSLs that could generate HIR from their own
internal IR.
The program contains two parts:
    1. construct the HIR of the example
    2. pass the HIR to the dawn compiler in order to run all optimizer passes and code generation.
       In this example the compiler is configured with the CUDA backend, therefore will code generate
       an optimized CUDA implementation.

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


def create_vertical_region_stmt() -> VerticalRegionDeclStmt:
    """ create a vertical region statement for the stencil
    """

    interval = make_interval(Interval.Start, Interval.End, 0, 0)

    # create the out = in[i+1] statement
    body_ast = make_ast(
        [
            make_assignment_stmt(
                make_field_access_expr("lap"),
                make_binary_operator(
                    make_binary_operator(
                        make_literal_access_expr("-4.0", BuiltinType.Float),
                        "*",
                        make_field_access_expr("in")
                    ),
                    "+",
                    make_binary_operator(
                        make_field_access_expr("coeff"),
                        "*",
                        make_binary_operator(
                            make_field_access_expr("in", [1, 0, 0]),
                            "+",
                            make_binary_operator(
                                make_field_access_expr("in", [-1, 0, 0]),
                                "+",
                                make_binary_operator(
                                    make_field_access_expr("in", [0, 1, 0]),
                                    "+",
                                    make_field_access_expr("in", [0, -1, 0])
                                )
                            )
                        )
                    )
                ),
                "="
            ),
            make_assignment_stmt(
                make_field_access_expr("out"),
                make_binary_operator(
                    make_binary_operator(
                        make_literal_access_expr("-4.0", BuiltinType.Float),
                        "*",
                        make_field_access_expr("lap")
                    ),
                    "+",
                    make_binary_operator(
                        make_field_access_expr("coeff"),
                        "*",
                        make_binary_operator(
                            make_field_access_expr("lap", [1, 0, 0]),
                            "+",
                            make_binary_operator(
                                make_field_access_expr("lap", [-1, 0, 0]),
                                "+",
                                make_binary_operator(
                                    make_field_access_expr("lap", [0, 1, 0]),
                                    "+",
                                    make_field_access_expr("lap", [0, -1, 0])
                                )
                            )
                        )
                    )
                ),
                "="
            )

        ]
    )

    vertical_region_stmt = make_vertical_region_decl_stmt(body_ast, interval, VerticalRegion.Forward)
    return vertical_region_stmt


hir = make_sir(GridType.Value('Cartesian'), "hori_diff.cpp", [
    make_stencil(
        "hori_diff",
        make_ast([create_vertical_region_stmt()]),
        [make_field("in", make_field_dimensions_cartesian()), make_field("out", make_field_dimensions_cartesian()), make_field("coeff", make_field_dimensions_cartesian()), make_field("lap", make_field_dimensions_cartesian(), is_temporary=True)]
    )

])

parser = OptionParser()
parser.add_option("-v", "--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="print the SIR")

(options, args) = parser.parse_args()

# Print the SIR to stdout only in verbose mode
if options.verbose:
    T = textwrap.TextWrapper(initial_indent=' ' * 1, width=120, subsequent_indent=' ' * 1)
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
dawn.dawnOptionsEntryCreateString.argtypes = [
    ctypes.c_char_p
]
backend = dawn.dawnOptionsEntryCreateString("cuda".encode('utf-8'))

dawn.dawnOptionsSet.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p,
    ctypes.c_void_p
]
dawn.dawnOptionsSet(options, "Backend".encode('utf-8'), backend)

# call the compiler that generates a translation unit
dawn.dawnCompile.restype = c_void_p
dawn.dawnCompile.argtypes = [
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_void_p
]
tu = dawn.dawnCompile(hirstr, len(hirstr), options)
stencilname = "hori_diff"
b_stencilName = stencilname.encode('utf-8')
# get the code of the translation unit for the given stencil
dawn.dawnTranslationUnitGetStencil.restype = c_void_p
dawn.dawnTranslationUnitGetStencil.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p
]
code = dawn.dawnTranslationUnitGetStencil(tu, b_stencilName)

# write to file
f = open(os.path.dirname(os.path.realpath(__file__)) + "/data/hori_diff.cpp", "w")
f.write(ctypes.c_char_p(code).value.decode("utf-8"))

f.close()
