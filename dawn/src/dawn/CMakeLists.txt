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

# GCC 8, still a common compiler, requires linking to a separate library for std::filesystem support.
# Once that is no longer common we can remove this hack
add_library(cpp_extra_libs INTERFACE)
if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 9.0)

    target_link_libraries(cpp_extra_libs INTERFACE stdc++fs)
endif()

install(
  TARGETS cpp_extra_libs
  EXPORT ${PROJECT_NAME}Targets
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

function(add_dawn_library)
  foreach(_l IN LISTS ARGN)
    # Link to cpp_extra_libs -- see above
    target_link_libraries(${_l} PUBLIC cpp_extra_libs)
    # Add to main project interface target
    target_link_libraries(${PROJECT_NAME} INTERFACE ${_l})
    # Add to install target group
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

add_subdirectory(Validator)
add_dawn_library(DawnValidator)

add_subdirectory(Serialization)
add_dawn_library(DawnSerialization)

add_subdirectory(CodeGen)
add_dawn_library(DawnCodeGen)

add_subdirectory(Compiler)
add_dawn_library(DawnCompiler)

if(${PROJECT_NAME}_TESTING)
  add_subdirectory(Unittest)
endif()

include(FetchCxxopts)

add_executable(dawn-opt dawn-opt.cpp)
target_link_libraries(dawn-opt Dawn cxxopts)
set_target_properties(dawn-opt PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
install(
  TARGETS dawn-opt
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

add_executable(dawn-codegen dawn-codegen.cpp)
target_link_libraries(dawn-codegen Dawn cxxopts)
set_target_properties(dawn-codegen PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
install(
  TARGETS dawn-codegen
  EXPORT ${PROJECT_NAME}Targets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
