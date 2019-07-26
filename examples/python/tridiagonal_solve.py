"""Tridiagonal solve computation HIR generator

This program creates the HIR corresponding to a tridiagonal solve computation using the Python API of the HIR.
The tridiagonal solve is a basic example that contains vertical data dependencies that need to be resolved 
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


def create_vertical_region_stmt1() -> VerticalRegionDeclStmt :
    """ create a vertical region statement for the stencil
    """

    interval = makeInterval(Interval.Start, Interval.Start, 0, 0)

    # create the out = in[i+1] statement
    body_ast = makeAST(
      [makeAssignmentStmt(
        makeFieldAccessExpr("c"),
        makeBinaryOperator(
          makeFieldAccessExpr("c"),
          "/",
          makeFieldAccessExpr("b")
        ),
        "="
      )
      ]
    )

    vertical_region_stmt = makeVerticalRegionDeclStmt(body_ast, interval, VerticalRegion.Forward)
    return vertical_region_stmt


def create_vertical_region_stmt2() -> VerticalRegionDeclStmt :
    """ create a vertical region statement for the stencil
    """

    interval = makeInterval(Interval.Start, Interval.End, 1, 0)

    # create the out = in[i+1] statement
    body_ast = makeAST(
    [
      makeVarDeclStmt(
        makeType(BuiltinType.Integer),
        "m", 0, "=",
        makeExpr(
          makeBinaryOperator(
            makeLiteralAccessExpr("1.0", BuiltinType.Float),
            "/",
            makeBinaryOperator(
              makeFieldAccessExpr("b"),
              "-",
              makeBinaryOperator(
                makeFieldAccessExpr("a"),
                "*",
                makeFieldAccessExpr("c", [0, 0, -1])
              )
            )
          )
        )
      ),
      makeAssignmentStmt(
          makeFieldAccessExpr("c"),
          makeBinaryOperator(
             makeFieldAccessExpr("c"),
             "*",
             makeVarAccessExpr("m")
          ),
          "="
      ),
      makeAssignmentStmt(
          makeFieldAccessExpr("d"),
          makeBinaryOperator(
              makeBinaryOperator(
                  makeFieldAccessExpr("d"),
                  "-",
                  makeBinaryOperator(
                    makeFieldAccessExpr("a"),
                    "*",
                    makeFieldAccessExpr("d",[0,0,-1])
                  )
              ),
              "*",
              makeVarAccessExpr("m")
          ),
          "="
      )
    ]
    )

    vertical_region_stmt = makeVerticalRegionDeclStmt(body_ast, interval, VerticalRegion.Forward)
    return vertical_region_stmt


def create_vertical_region_stmt3() -> VerticalRegionDeclStmt :
    """ create a vertical region statement for the stencil
    """

    interval = makeInterval(Interval.Start, Interval.End, 0, -1)

    # create the out = in[i+1] statement
    body_ast = makeAST(
      [makeAssignmentStmt(
        makeFieldAccessExpr("d"),
        makeBinaryOperator(
          makeFieldAccessExpr("c"),
          "*",
          makeFieldAccessExpr("d",[0,0,1])
        ),
        "-="
      )
      ]
    )

    vertical_region_stmt = makeVerticalRegionDeclStmt(body_ast, interval, VerticalRegion.Backward)
    return vertical_region_stmt


parser = OptionParser()
parser.add_option("-v", "--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="print the SIR")

(options, args) = parser.parse_args()

hir = makeSIR("tridiagonal_solve.cpp", [
        makeStencil(
          "tridiagonal_solve",
          makeAST([
              create_vertical_region_stmt1(),
              create_vertical_region_stmt2(),
              create_vertical_region_stmt3()
          ]),
          [makeField("a"), makeField("b"), makeField("c"), makeField("d")]
        )

      ])

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
stencilname = "tridiagonal_solve"
b_stencilName = stencilname.encode('utf-8')
# get the code of the translation unit for the given stencil
code = dawn.dawnTranslationUnitGetStencil(tu, b_stencilName)

# write to file
f = open(os.path.dirname(os.path.realpath(__file__))+"/data/tridiagonal_solve.cpp","w")
f.write(ctypes.c_char_p(code).value.decode("utf-8"))

f.close()
