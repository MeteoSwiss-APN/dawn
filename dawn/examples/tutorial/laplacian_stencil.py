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

This program creates the HIR corresponding to a copy stencil using the Python API of the HIR.
The copy stencil is a hello world for stencil computations.
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


def create_vertical_region_stmt() -> VerticalRegionDeclStmt:
    """ create a vertical region statement for the stencil
    """

    interval = make_interval(Interval.Start, Interval.End, 0, 0)

    # create the out = in[i+1] statement
    body_ast = make_ast(
        [  
        make_assignment_stmt(
            make_field_access_expr("out", [0, 0, 0]),
            make_binary_operator(
              make_binary_operator(
                make_binary_operator(
                  make_field_access_expr("in", [0, 0, 0]),
                  "*",
                  make_literal_access_expr("-4.0", BuiltinType.Float)
                ),
                "+",
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
                            ),
                        ),
                    ),
                ),
              "/",
              make_binary_operator(
                make_var_access_expr("dx", is_external=True),
                "*",
                make_var_access_expr("dx", is_external=True),
              )
            ),           
            "="
          )
        ]
    )

    vertical_region_stmt = make_vertical_region_decl_stmt(
        body_ast, interval, VerticalRegion.Forward)
    return vertical_region_stmt


stencilsglobals  = GlobalVariableMap()
myglobal = stencilsglobals.map['dx'].double_value = 0.;

hir = make_sir("laplacian_stencil_from_python.cpp", stencils=[
    make_stencil(
        "laplacian_stencil",
        make_ast([create_vertical_region_stmt()]),
        [make_field("out",), make_field("in")]
    )

],global_variables=stencilsglobals)

parser = OptionParser()
parser.add_option("-v", "--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="print the SIR")

(options, args) = parser.parse_args()

# Print the SIR to stdout 
T = textwrap.TextWrapper(
    initial_indent=' ' * 1, width=120, subsequent_indent=' ' * 1)
des = sir_printer.SIRPrinter()

des.visit_global_variables(stencilsglobals)

for stencil in hir.stencils:
    des.visit_stencil(stencil)

# serialize the hir to pass it to the compiler
hirstr = hir.SerializeToString()
fhir = open("./laplacian_stencil_from_python.sir", "wb");
fhir.write(hirstr)
fhir.close

# create the options to control the compiler
dawn.dawnOptionsCreate.restype = c_void_p
options = dawn.dawnOptionsCreate()

# we set the backend of the compiler to cuda
dawn.dawnOptionsEntryCreateString.restype = c_void_p
dawn.dawnOptionsEntryCreateString.argtypes = [
    ctypes.c_char_p
]
backend = dawn.dawnOptionsEntryCreateString("c++-naive".encode('utf-8'))

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
stencilname = "laplacian_stencil"
b_stencilName = stencilname.encode('utf-8')
# get the code of the translation unit for the given stencil
dawn.dawnTranslationUnitGetStencil.restype = c_void_p
dawn.dawnTranslationUnitGetStencil.argtypes = [
    ctypes.c_void_p,
    ctypes.c_char_p
]

dawn.dawnTranslationUnitGetPPDefines.restype = c_void_p
dawn.dawnTranslationUnitGetPPDefines.argtypes = [
    ctypes.c_void_p,
    POINTER(POINTER(ctypes.c_char_p)),
    POINTER(ctypes.c_int)
]

pp_defines = POINTER(ctypes.c_char_p)()
num_pp_defines = ctypes.c_int()
dawn.dawnTranslationUnitGetPPDefines(tu, byref(pp_defines), byref(num_pp_defines))
num_pp_defines = num_pp_defines.value

global_defines = dawn.dawnTranslationUnitGetGlobals(tu)
code = dawn.dawnTranslationUnitGetStencil(tu, b_stencilName)

# write to file
f = open(os.path.dirname(os.path.realpath(__file__))
         + "/laplacian_stencil_from_python.cpp", "w")
for i in range(0,num_pp_defines):
    f.write(ctypes.c_char_p(pp_defines[i]).value.decode("utf-8") + "\n")    
f.write(ctypes.c_char_p(global_defines).value.decode("utf-8"))
f.write(ctypes.c_char_p(code).value.decode("utf-8"))

f.close()
