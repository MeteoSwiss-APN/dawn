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

"""Copy stencil HIR generator

This program creates the HIR corresponding to an unstructured stencil using the Python API of the HIR.
The code is meant as an example for high-level DSLs that could generate HIR from their own
internal IR.
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
    body_ast = make_ast([
        make_assignment_stmt(
            make_field_access_expr("out"),
            make_reduction_over_neighbor_expr("+",
                                              make_literal_access_expr(
                                                  "1.0", BuiltinType.Float),
                                              make_field_access_expr("in")),
            "=")
    ])

    vertical_region_stmt = make_vertical_region_decl_stmt(
        body_ast, interval, VerticalRegion.Forward)
    return vertical_region_stmt


hir = make_sir(GridType.Value('Triangular'), "unstructured_stencil.cpp", [
    make_stencil(
        "unstructured_stencil",
        make_ast([create_vertical_region_stmt()]),
        [make_field("in", make_field_dimensions_triangular([LocationType.Value('Cell')], 1)), make_field("out", make_field_dimensions_triangular([LocationType.Value('Edge')], 1))]
    )

])

parser = OptionParser()
parser.add_option("-v", "--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="print the SIR")

(options, args) = parser.parse_args()

# Print the SIR to stdout only in verbose mode
if options.verbose:
    T = textwrap.TextWrapper(
        initial_indent=' ' * 1, width=120, subsequent_indent=' ' * 1)
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
backend = dawn.dawnOptionsEntryCreateString("c++-naive-ico".encode('utf-8'))

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
stencilname = "unstructured_stencil"
b_stencilName = stencilname.encode('utf-8')
# get the code of the translation unit for the given stencil
dawn.dawnTranslationUnitGetStencil.restype = c_void_p
dawn.dawnTranslationUnitGetStencil.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p
]
code = dawn.dawnTranslationUnitGetStencil(tu, b_stencilName)

# write to file
f = open(os.path.dirname(os.path.realpath(__file__))
         + "/data/unstructured_stencil.cpp", "w")
f.write(ctypes.c_char_p(code).value.decode("utf-8"))

f.close()
