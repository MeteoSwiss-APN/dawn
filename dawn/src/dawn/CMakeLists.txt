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
install(
  TARGETS ${PROJECT_NAME}
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

function(add_dawn_library)
  foreach(_l IN LISTS ARGN)
    target_link_libraries(${PROJECT_NAME} INTERFACE ${_l})
    install(
      TARGETS ${_l}
      EXPORT ${PROJECT_NAME}Targets
      LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
      ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    )
  add_library(${PROJECT_NAME}::${_l} ALIAS ${_l})
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
add_dawn_library(DawnIIRProto DawnIIR)

add_subdirectory(Optimizer)
add_dawn_library(DawnOptimizer)

add_subdirectory(Serialization)
add_dawn_library(DawnSerialization)

add_subdirectory(CodeGen)
add_dawn_library(DawnCodeGen)

add_subdirectory(Compiler)
add_dawn_library(DawnCompiler)

if(${PROJECT_NAME}_TESTING)
  add_subdirectory(Unittest)
endif()
