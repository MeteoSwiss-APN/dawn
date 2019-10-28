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

find_package(Protobuf REQUIRED)

get_filename_component(protobuf_lib_dir ${Protobuf_LIBRARY} DIRECTORY)
yoda_enable_full_rpath("${protobuf_lib_dir}")

yoda_export_package(
  NAME Protobuf
  FOUND ${Protobuf_FOUND}
  EXECUTABLE "${Protobuf_PROTOC_EXECUTABLE}"
  INCLUDE_DIRS "${Protobuf_INCLUDE_DIR}"
  LIBRARIES "protobuf::libprotobuf"
  VERSION "${Protobuf_VERSION}"
  # https://stackoverflow.com/questions/34474091/protobuf-inline-not-in-headers
  DEFINITIONS "-DPROTOBUF_INLINE_NOT_IN_HEADERS=0"
)
