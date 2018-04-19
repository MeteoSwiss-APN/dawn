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

include(mchbuildExportPackage)

find_package(Boost 1.58 REQUIRED)

mchbuild_export_package(
  NAME Boost
  FOUND ${Boost_FOUND} 
  VERSION "${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION}" 
  INCLUDE_DIRS ${Boost_INCLUDE_DIRS}
)
