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

add_library(DawnValidator
    UnstructuredDimensionChecker.h
    UnstructuredDimensionChecker.cpp
    GridTypeChecker.h
    GridTypeChecker.cpp
    IntegrityChecker.h
    IntegrityChecker.cpp
)

target_add_dawn_standard_props(DawnValidator)
target_link_libraries(DawnValidator PUBLIC DawnIIR DawnSIR)