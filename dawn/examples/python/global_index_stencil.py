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

import dawn4py
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as sir_utils

OUTPUT_NAME = "global_index_stencil"
OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "data", f"{OUTPUT_NAME}.cpp")


def create_vertical_region_stmt() -> SIR.VerticalRegionDeclStmt:
    """ create a vertical region statement for the stencil
    """

    interval = sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.End, 0, 0)

    # create the out = in[i+1] statement
    body_ast = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("out", [0, 0, 0]),
                sir_utils.make_field_access_expr("in", [0, 0, 0]),
                "=",
            )
        ]
    )

    vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(body_ast, interval, SIR.VerticalRegion.Forward)
    return vertical_region_stmt


def create_boundary_correction_region(value="0", i_interval=None, j_interval=None) -> SIR.VerticalRegionDeclStmt:
    interval = sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.End, 0, 0)
    boundary_body = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("out", [0, 0, 0]),
                sir_utils.make_literal_access_expr(value, SIR.BuiltinType.Float),
                "=",
            )
        ]
    )
    vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(
        boundary_body, interval, SIR.VerticalRegion.Forward, IRange=i_interval, JRange=j_interval
    )
    return vertical_region_stmt


def main(args: argparse.Namespace):
    sir = sir_utils.make_sir(
        sir_utils.GridType.Value("Cartesian"),
        OUTPUT_FILE,
        [
            sir_utils.make_stencil(
                "global_indexing",
                sir_utils.make_ast(
                    [
                        create_vertical_region_stmt(),
                        create_boundary_correction_region(
                            value="4", i_interval=sir_utils.make_interval(SIR.Interval.End, SIR.Interval.End, -1, 0)
                        ),
                        create_boundary_correction_region(
                            value="8", i_interval=sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.Start, 0, 1)
                        ),
                        create_boundary_correction_region(
                            value="6", j_interval=sir_utils.make_interval(SIR.Interval.End, SIR.Interval.End, -1, 0)
                        ),
                        create_boundary_correction_region(
                            value="2", j_interval=sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.Start, 0, 1)
                        ),
                        create_boundary_correction_region(
                            value="1",
                            j_interval=sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.Start, 0, 1),
                            i_interval=sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.Start, 0, 1),
                        ),
                        create_boundary_correction_region(
                            value="3",
                            j_interval=sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.Start, 0, 1),
                            i_interval=sir_utils.make_interval(SIR.Interval.End, SIR.Interval.End, -1, 0),
                        ),
                        create_boundary_correction_region(
                            value="7",
                            j_interval=sir_utils.make_interval(SIR.Interval.End, SIR.Interval.End, -1, 0),
                            i_interval=sir_utils.make_interval(SIR.Interval.Start, SIR.Interval.Start, 0, 1),
                        ),
                        create_boundary_correction_region(
                            value="5",
                            j_interval=sir_utils.make_interval(SIR.Interval.End, SIR.Interval.End, -1, 0),
                            i_interval=sir_utils.make_interval(SIR.Interval.End, SIR.Interval.End, -1, 0),
                        ),
                    ]
                ),
                [sir_utils.make_field("in"), sir_utils.make_field("out")],
            )
        ],
    )

    # print the SIR
    if args.verbose:
        sir_utils.pprint(sir)

    # compile
    code = dawn4py.compile(sir, backend="c++-naive")

    # write to file
    print(f"Writing generated code to '{OUTPUT_PATH}'")
    with open(OUTPUT_PATH, "w") as f:
        f.write(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a simple copy-shift stencil using Dawn compiler")
    parser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", default=False, help="Print the generated SIR",
    )
    main(parser.parse_args())


# parser = OptionParser()
# parser.add_option("-v", "--verbose", action="store_true", dest="verbose", default=False, help="print the SIR")

# (options, args) = parser.parse_args()

# # Print the SIR to stdout only in verbose mode
# if options.verbose:
#     T = textwrap.TextWrapper(initial_indent=" " * 1, width=120, subsequent_indent=" " * 1)
#     des = sir_printer.SIRPrinter()

#     for stencil in hir.stencils:
#         des.visit_stencil(stencil)

# # serialize the hir to pass it to the compiler
# hirstr = hir.SerializeToString()

# # create the options to control the compiler
# dawn.dawnOptionsCreate.restype = c_void_p
# options = dawn.dawnOptionsCreate()

# # we set the backend of the compiler to cuda
# dawn.dawnOptionsEntryCreateString.restype = c_void_p
# dawn.dawnOptionsEntryCreateString.argtypes = [ctypes.c_char_p]

# dawn.dawnOptionsSet.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
# backend = dawn.dawnOptionsEntryCreateString("c++-naive".encode("utf-8"))
# dawn.dawnOptionsSet(options, "Backend".encode("utf-8"), backend)

# none = dawn.dawnOptionsEntryCreateString("none".encode("utf-8"))
# dawn.dawnOptionsSet(options, "ReorderStrategy".encode("utf-8"), none)

# # one = dawn.dawnOptionsEntryCreateInteger(1)
# # dawn.dawnOptionsSet(options, "DumpStencilInstantiation".encode("utf-8"), one)

# # call the compiler that generates a translation unit

# dawn.dawnCompile.restype = c_void_p
# dawn.dawnCompile.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
# tu = dawn.dawnCompile(hirstr, len(hirstr), options)
# stencilname = "global_indexing"
# b_stencilName = stencilname.encode("utf-8")
# # get the code of the translation unit for the given stencil
# dawn.dawnTranslationUnitGetStencil.restype = c_void_p
# dawn.dawnTranslationUnitGetStencil.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
# code = dawn.dawnTranslationUnitGetStencil(tu, b_stencilName)

# # write to file
# f = open(os.path.dirname(os.path.realpath(__file__)) + "/data/global_indexing.cpp", "w")
# f.write(ctypes.c_char_p(code).value.decode("utf-8"))

# f.close()
