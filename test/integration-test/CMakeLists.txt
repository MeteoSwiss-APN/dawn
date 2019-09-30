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

# dawn_add_integrationtest_impl
# ----------------------
#
# Compile the given objects into a runnable integration test executable (.exe) and register it within CTest.
# The executable will be stored in ${CMAKE_BINARY_DIR}/bin/integrationtest.
#
#    NAME:STRING=<>     - Name of the unittest exectuable as well as the CMake target to build it.
#    SOURCES:STRING=<>   - List of source files making up the exectuable.
#
macro(dawn_add_integrationtest_impl)
  cmake_parse_arguments(ARG "" "NAME" "SOURCES" ${ARGN})

  if(NOT("${ARG_UNPARSED_ARGUMENTS}" STREQUAL ""))
    message(FATAL_ERROR "dawn_add_integrationtest_impl: invalid argument ${ARG_UNPARSED_ARGUMENTS}")
  endif()

  yoda_add_unittest(
    NAME ${ARG_NAME}
    SOURCES ${ARG_SOURCES}
    DEPENDS DawnUnittestStatic DawnCStatic DawnStatic ${DAWN_EXTERNAL_LIBRARIES} gtest
    OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/integrationtest
    GTEST_ARGS --gtest_color=yes
  )
endmacro()

dawn_add_integrationtest_impl(
  NAME DawnIntegrationTest
  SOURCES TestMain.cpp
          TestIIRDeserializer.cpp
          GenerateInMemoryStencils.cpp
)

yoda_add_executable(
  NAME DawnUpdateIIRReferences
  SOURCES RegenerateIIRMain.cpp
          GenerateInMemoryStencils.cpp
  DEPENDS DawnUnittestStatic DawnCStatic DawnStatic ${DAWN_EXTERNAL_LIBRARIES}
  OUTPUT_DIR ${CMAKE_BINARY_DIR}/bin/integrationtest
)

file(COPY reference_iir DESTINATION ${CMAKE_BINARY_DIR}/bin/integrationtest)
file(COPY reference_iir DESTINATION ${CMAKE_BINARY_DIR}/test/integration-test)