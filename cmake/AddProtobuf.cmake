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

find_package(Protobuf 3.4 NO_MODULE REQUIRED)

get_property(Protobuf_INCLUDE_DIRS 
  TARGET protobuf::libprotobuf 
  PROPERTY INTERFACE_INCLUDE_DIRECTORIES
)

get_property(Protobuf_PROTOC_EXECUTABLE 
  TARGET protobuf::protoc 
  PROPERTY LOCATION
)

# Get the <root> directory of protobuf by assuming protoc is installed in <root>/bin/protoc
get_filename_component(root_bin_dir ${Protobuf_PROTOC_EXECUTABLE} DIRECTORY)
get_filename_component(root_dir ${root_bin_dir} DIRECTORY)
get_filename_component(root_dir ${root_dir} ABSOLUTE)
set(DAWN_PROTOBUF_ROOT ${root_dir})

# set the path to the install-direcotry of protobuf
set(DAWN_PROTOBUF_RPATH_DIR ${DAWN_PROTOBUF_ROOT}/lib)

dawn_export_package(
  NAME Protobuf
  FOUND ${Protobuf_FOUND}
  EXECUTABLE "${Protobuf_PROTOC_EXECUTABLE}"
  INCLUDE_DIRS "${Protobuf_INCLUDE_DIRS}"
  LIBRARIES "protobuf::libprotobuf"
  VERSION "${Protobuf_VERSION}"
  # https://stackoverflow.com/questions/34474091/protobuf-inline-not-in-headers
  DEFINITIONS "-DPROTOBUF_INLINE_NOT_IN_HEADERS=0"
)
