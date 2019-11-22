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

add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

function(add_dawn_library libs)
  foreach(_l ${libs})
    install(
      TARGETS ${_l}
      EXPORT ${PROJECT_NAME}Targets
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
    target_link_libraries(Dawn
      INTERFACE ${PROJECT_NAME}${_l}
    )
  endforeach()
endfunction()

add_subdirectory(Support)
add_dawn_library(DawnSupport)

# AST and SIR have a circular dependency, only building one library -- DawnSIR
# {
add_subdirectory(AST)
add_dawn_library(DawnAST)

add_subdirectory(SIR)
add_dawn_library(DawnSIRProto DawnSIR)
# }

add_subdirectory(IIR)
add_dawn_library(DawnIIR DawnIIRProto)

add_subdirectory(Optimizer)
add_dawn_library(DawnOptimizer)

add_subdirectory(Serialization)
add_dawn_library(DawnSerialization)

add_subdirectory(CodeGen)
add_dawn_library(DawnCodeGen)

add_subdirectory(Compiler)
add_dawn_library(DawnCompiler)

add_subdirectory(Unittest)

install(
  TARGETS protobuf::libprotobuf
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
