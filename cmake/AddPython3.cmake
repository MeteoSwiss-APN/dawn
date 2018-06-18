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

#if(NOT DEFINED PYTHON_EXECUTABLE)
  find_package(PythonInterp 3.5 REQUIRED)
#endif() 

# Look for all the required modules
include(mchbuildFindPythonModule)
mchbuild_find_python_module(sphinx REQUIRED)
mchbuild_find_python_module(docutils REQUIRED)

mchbuild_export_package(
  NAME Python3
  FOUND ${PYTHONINTERP_FOUND}
  EXECUTABLE ${PYTHON_EXECUTABLE}
  VERSION ${PYTHON_VERSION_STRING}
)
