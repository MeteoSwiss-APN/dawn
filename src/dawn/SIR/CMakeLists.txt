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

include(DawnProtobufGenerate)

# Genreate C++ proto files
dawn_protobuf_generate(
  OUT_FILES sir_proto_cpp_files
  OUT_INCLUDE_DIRS sir_proto_include_dirs
  WDIR ${CMAKE_CURRENT_SOURCE_DIR}/proto
  PROTOS SIR/SIR.proto SIR/statements.proto
  PACKG SIR
  LANGUAGE cpp
)

yoda_add_library(
  NAME DawnSIR
  SOURCES AST.h
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
          proto/SIR/SIR.proto
          ${sir_proto_cpp_files}
  OBJECT
)

yoda_combine_libraries(
  NAME DawnSIR
  OBJECTS DawnSIRObjects
  INSTALL_DESTINATION ${DAWN_INSTALL_LIB_DIR}
  DEPENDS DawnSupportStatic DawnASTStatic protobuf::libprotobuf
  VERSION ${DAWN_VERSION}
  EXPORT_GROUP DawnTargets
)

# only include proto since protobuf uses relative path "statements.proto" -> export to dependent
target_include_directories(DawnSIRObjects PUBLIC ${sir_proto_include_dirs}/../)

target_include_directories(DawnSIRStatic PUBLIC $<BUILD_INTERFACE:${sir_proto_include_dirs}/../> $<INSTALL_INTERFACE:include/>)
target_include_directories(DawnSIRShared PUBLIC $<BUILD_INTERFACE:${sir_proto_include_dirs}/../> $<INSTALL_INTERFACE:include/>)


add_custom_target(SIR_Proto_Generated ALL DEPENDS ${sir_proto_cpp_files})

if(DAWN_PYTHON) 
  # Genreate python proto files
  dawn_protobuf_generate(
    OUT_FILES sir_proto_python_files
    PROTOS SIR/SIR.proto SIR/statements.proto
    WDIR ${CMAKE_CURRENT_SOURCE_DIR}/proto
    PACKG SIR
    LANGUAGE python
  )

  add_custom_target(DawnSIRPythonProto ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${sir_proto_python_files} ${CMAKE_BINARY_DIR}/python/dawn
    DEPENDS ${sir_proto_python_files}
  )
  install(FILES ${sir_proto_python_files} DESTINATION ${DAWN_INSTALL_PYTHON_DIR}/SIR)
  install(FILES ${CMAKE_SOURCE_DIR}/python/SIR/__init__.py DESTINATION ${DAWN_INSTALL_PYTHON_DIR}/SIR)
endif(DAWN_PYTHON)

if(DAWN_JAVA)
  # Genreate python proto files
  dawn_protobuf_generate(
    OUT_FILES sir_proto_java_files
    OUT_DIRS sir_proto_java_dirs
    PROTOS SIR/SIR.proto SIR/statements.proto 
    WDIR ${CMAKE_CURRENT_SOURCE_DIR}/proto
    PACKG SIR
    LANGUAGE java
  )

  add_custom_target(DawnSIRJavaProto ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${sir_proto_java_dirs} ${CMAKE_BINARY_DIR}/java/dawn
    DEPENDS ${sir_proto_java_files}
  )

  install(DIRECTORY ${sir_proto_java_dirs} DESTINATION ${DAWN_INSTALL_JAVA_DIR}/dawn)
endif(DAWN_JAVA)
