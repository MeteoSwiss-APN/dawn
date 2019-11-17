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
add_subdirectory(AST)
add_subdirectory(SIR)
add_subdirectory(IIR)
add_subdirectory(Optimizer)
add_subdirectory(Serialization)
add_subdirectory(CodeGen)
add_subdirectory(Compiler)
add_subdirectory(Unittest)

set(DAWN_OBJECT_LIBS
    DawnSupport
    DawnAST
    DawnSIR
    DawnSIRProto
    DawnIIR
    DawnIIRProto
    DawnOptimizer
    DawnSerialization
    DawnCodeGen
    DawnCompiler
    DawnUnittest #TODO: we don't want it here but it is needed for some gtclang tests
)
set(DAWN_OBJECT_LIBS ${DAWN_OBJECT_LIBS} PARENT_SCOPE)

foreach(_target ${DAWN_OBJECT_LIBS})
    list(APPEND dawn_objs $<TARGET_OBJECTS:${_target}>)
endforeach()

add_library(${PROJECT_NAME} ${dawn_objs})
add_library(Dawn::Dawn ALIAS ${PROJECT_NAME})
target_link_options(${PROJECT_NAME} PRIVATE ${DAWN_LINKER_FLAGS})
target_link_libraries(${PROJECT_NAME} ${DAWN_OBJECT_LIBS})
