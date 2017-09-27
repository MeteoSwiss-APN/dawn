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

find_package(ccache)

dawn_export_package(
  NAME ccache
  FOUND ${CCACHE_FOUND}
  EXECUTABLE ${CCACHE_EXECUTABLE}
)

if(CCACHE_FOUND)
  set(DAWN_HAS_CCACHE 1)
  if(DAWN_USE_CCACHE)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE ${CCACHE_EXECUTABLE})
    set_property(GLOBAL PROPERTY RULE_LAUNCH_LINK ${CCACHE_EXECUTABLE})
  endif()
else()
  set(DAWN_HAS_CCACHE 0)
endif()
