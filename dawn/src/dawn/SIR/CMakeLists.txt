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

set(proto_files
  proto/SIR/SIR.proto
  proto/SIR/statements.proto
)

list(TRANSFORM proto_files REPLACE "[.]proto$" ".pb.cc" OUTPUT_VARIABLE proto_cc)
list(TRANSFORM proto_cc REPLACE "^proto/" "${CMAKE_CURRENT_BINARY_DIR}/")

list(TRANSFORM proto_files REPLACE "[.]proto$" ".pb.h" OUTPUT_VARIABLE proto_h)
list(TRANSFORM proto_h REPLACE "^proto/" "${CMAKE_CURRENT_BINARY_DIR}/")

add_custom_command(
  COMMAND protobuf::protoc --cpp_out=${CMAKE_CURRENT_BINARY_DIR} --proto_path=proto/ ${proto_files}
  OUTPUT  ${proto_cc} ${proto_h}
  DEPENDS ${proto_files}
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMENT "Generating SIR protobuf cpp files"
)

add_library(DawnSIRProto OBJECT ${proto_cc} ${proto_h})

# Generated files need to be compiled with the protobuf headers
target_include_directories(DawnSIRProto
  PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
  ${Protobuf_INCLUDE_DIRS}
)

target_link_libraries(DawnSIRProto PUBLIC protobuf::libprotobuf)

add_library(DawnSIR OBJECT
  AST.h
  ASTExpr.h
  ASTFwd.h
  ASTStmt.h
  ASTStmt.cpp
  ASTStringifier.h
  ASTUtil.h
  ASTUtil.cpp
  ASTVisitor.h
  SIR.cpp
  SIR.h
)

target_include_directories(DawnSIR
  PUBLIC
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/src>
  PRIVATE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
)
target_compile_features(DawnSIR PUBLIC cxx_std_17)
target_compile_options(DawnSIR PRIVATE ${DAWN_CXX_FLAGS})
set_target_properties(DawnSIR PROPERTIES ${DAWN_TARGET_PROPERTIES})
target_link_libraries(DawnSIR PUBLIC DawnSupport DawnSIRProto)

if(ENABLE_PYTHON)
  list(TRANSFORM proto_files REPLACE "[.]proto$" ".pb.py" OUTPUT_VARIABLE proto_py)
  list(TRANSFORM proto_py PREPEND ${CMAKE_CURRENT_BINARY_DIR}/)

  add_custom_command(
    COMMAND protobuf::protoc --python_out=${PROJECT_BINARY_DIR}/python/dawn --proto_path=proto/ ${proto_files}
    OUTPUT  ${proto_py}
    DEPENDS ${proto_files}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    COMMENT "Generating SIR protobuf python files"
  )

#   install(FILES ${sir_proto_python_files} DESTINATION ${DAWN_INSTALL_PYTHON_DIR}/SIR)
#   install(FILES ${CMAKE_SOURCE_DIR}/python/SIR/__init__.py DESTINATION ${DAWN_INSTALL_PYTHON_DIR}/SIR)
endif()
