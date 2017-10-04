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

find_package(Protobuf 3.0 NO_MODULE REQUIRED)

get_property(Protobuf_INCLUDE_DIRS 
  TARGET protobuf::libprotobuf 
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES
)

get_property(Protobuf_PROTOC_EXECUTABLE 
  TARGET protobuf::protoc 
  PROPERTY LOCATION
)

dawn_export_package(
  NAME Protobuf
  FOUND ${Protobuf_FOUND}
  EXECUTABLE "${Protobuf_PROTOC_EXECUTABLE}"
  INCLUDE_DIRS "${Protobuf_INCLUDE_DIRS}"
  LIBRARIES "protobuf::libprotobuf"
  VERSION "${Protobuf_VERSION}"
)
