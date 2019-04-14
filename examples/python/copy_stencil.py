"""Copy stencil HIR generator

This program creates the HIR corresponding to a copy stencil using the Python API of the HIR.
The copy stencil is a hello world for stencil computations.
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

def createVerticalRegionStmt() -> VerticalRegionDeclStmt :
  """ create a vertical region statement for the stencil
  """

  interval = makeInterval(Interval.Start, Interval.End, 0, 0)

  # create the out = in[i+1] statement
  bodyAST = makeAST(
      [makeAssignmentStmt(
        makeFieldAccessExpr("out",[0,0,0]),
        makeFieldAccessExpr("in",[1,0,0]),
        "="
      )
      ]
  )

  verticalRegionStmt = makeVerticalRegionDeclStmt(bodyAST, interval, VerticalRegion.Forward)
  return verticalRegionStmt

hir = makeSIR("copy_stencil.cpp", [
        makeStencil(
          "copy_stencil",
          makeAST([createVerticalRegionStmt()]),
          [makeField("in"), makeField("out")]
        )

      ])

parser = OptionParser()
parser.add_option("-v", "--verbose",
                  action="store_true", dest="verbose", default=False,
                  help="print the SIR")

(options, args) = parser.parse_args()


## Print the SIR to stdout only in verbose mode
if options.verbose:
  T = textwrap.TextWrapper(initial_indent=' '*1, width=120,subsequent_indent=' '*1)
  des = sir_printer.SIRPrinter()

  #des.visitGlobalVariables(hir.global_variables)

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
stencilname = "copy_stencil"
b_stencilName = stencilname.encode('utf-8')
# get the code of the translation unit for the given stencil
code = dawn.dawnTranslationUnitGetStencil(tu, b_stencilName)

# write to file
f = open(os.path.dirname(os.path.realpath(__file__))+"/data/copy_stencil.cpp","w")
f.write(ctypes.c_char_p(code).value.decode("utf-8"))

f.close()
