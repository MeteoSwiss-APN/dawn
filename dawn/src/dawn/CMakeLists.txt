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

add_subdirectory(Support)
list(APPEND DAWN_LIBS DawnSupport)

# AST and SIR have a circular dependency, only building one library -- DawnSIR
# {
add_subdirectory(AST)
list(APPEND DAWN_LIBS DawnAST)

add_subdirectory(SIR)
list(APPEND DAWN_LIBS DawnSIR DawnSIRProto)
# }
add_subdirectory(IIR)
list(APPEND DAWN_LIBS DawnIIR DawnIIRProto)

add_subdirectory(Optimizer)
list(APPEND DAWN_LIBS DawnOptimizer)

add_subdirectory(Serialization)
list(APPEND DAWN_LIBS DawnSerialization)

add_subdirectory(CodeGen)
list(APPEND DAWN_LIBS DawnCodeGen)

add_subdirectory(Compiler)
list(APPEND DAWN_LIBS DawnCompiler)

add_subdirectory(Unittest)
# We do not add DawnUnittest to DAWN_LIBS

set(DAWN_LIBS ${DAWN_LIBS} PARENT_SCOPE)

add_library(Dawn INTERFACE)
target_link_libraries(Dawn INTERFACE ${DAWN_LIBS})
