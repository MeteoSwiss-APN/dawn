##===------------------------------------------------------------------------------*- CMake -*-===##
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

find_package(LATEX COMPONENTS PDFLATEX)

yoda_export_package(
  NAME LATEX
  FOUND ${LATEX_FOUND}
  EXECUTABLE ${PDFLATEX_COMPILER}
)

set(DAWN_HAS_LATEX 0)
if(LATEX_FOUND)
  set(DAWN_HAS_LATEX 1)
endif()
