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

import textwrap
import sys
import argparse
import ctypes
import os.path
from optparse import OptionParser
from ctypes import *

from config import __dawn_install_module__,__dawn_install_dawnclib__ 

from dawn import *
from dawn import sir_printer

dawn = CDLL(__dawn_install_dawnclib__)


def create_vertical_region_stmt() -> VerticalRegionDeclStmt :
    """ create a vertical region statement for the stencil
    """

    interval = makeInterval(Interval.Start, Interval.End, 0, 0)

    # create the out = in[i+1] statement
    body_ast = makeAST(
      [
      makeAssignmentStmt(
        makeFieldAccessExpr("lap"),
        makeBinaryOperator(
          makeBinaryOperator(
            makeLiteralAccessExpr("-4.0", BuiltinType.Float),
            "*",
            makeFieldAccessExpr("in")
          ),
          "+",
          makeBinaryOperator(
            makeFieldAccessExpr("coeff"),
            "*",
            makeBinaryOperator(
              makeFieldAccessExpr("in",[1,0,0]),
              "+",
              makeBinaryOperator(
                makeFieldAccessExpr("in",[-1,0,0]),
                "+",
                makeBinaryOperator(
                  makeFieldAccessExpr("in",[0,1,0]),
                  "+",
                  makeFieldAccessExpr("in",[0,-1,0])
                )
              )
            )
          )  
        ),
        "="
      ),
      makeAssignmentStmt(
        makeFieldAccessExpr("out"),
        makeBinaryOperator(
          makeBinaryOperator(
            makeLiteralAccessExpr("-4.0", BuiltinType.Float),
            "*",
            makeFieldAccessExpr("lap")
          ),
          "+",
          makeBinaryOperator(
            makeFieldAccessExpr("coeff"),
            "*",
            makeBinaryOperator(
              makeFieldAccessExpr("lap",[1,0,0]),
              "+",
              makeBinaryOperator(
                makeFieldAccessExpr("lap",[-1,0,0]),
                "+",
                makeBinaryOperator(
                  makeFieldAccessExpr("lap",[0,1,0]),
                  "+",
                  makeFieldAccessExpr("lap",[0,-1,0])
                )
              )
            )
          )  
        ),
        "="
      )

      ]
    )

    vertical_region_stmt = makeVerticalRegionDeclStmt(body_ast, interval, VerticalRegion.Forward)
    return vertical_region_stmt


hir = makeSIR("hori_diff.cpp", [
        makeStencil(
          "hori_diff",
          makeAST([create_vertical_region_stmt()]),
          [makeField("in"), makeField("out"), makeField("coeff"), makeField("lap", is_temporary=True)]
        )

      ])

parser = OptionParser()
parser.add_option("-v", "--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="print the SIR")

(options, args) = parser.parse_args()

# Print the SIR to stdout only in verbose mode
if options.verbose:
    T = textwrap.TextWrapper(initial_indent=' '*1, width=120,subsequent_indent=' '*1)
    des = sir_printer.SIRPrinter()

    for stencil in hir.stencils:
        des.visitStencil(stencil)

# serialize the hir to pass it to the compiler
hirstr = hir.SerializeToString()

# create the options to control the compiler
options = dawn.dawnOptionsCreate()
# we set the backend of the compiler to cuda
backend = dawn.dawnOptionsEntryCreateString("cuda".encode('utf-8'))
dawn.dawnOptionsSet(options, "Backend".encode('utf-8'), backend)

# call the compiler that generates a translation unit
tu = dawn.dawnCompile(hirstr, len(hirstr), options)
stencilname = "hori_diff"
b_stencilName = stencilname.encode('utf-8')
# get the code of the translation unit for the given stencil
code = dawn.dawnTranslationUnitGetStencil(tu, b_stencilName)

# write to file
f = open(os.path.dirname(os.path.realpath(__file__))+"/data/hori_diff.cpp","w")
f.write(ctypes.c_char_p(code).value.decode("utf-8"))

f.close()
