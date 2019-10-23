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

# Get the <root> directory of protobuf by assuming protoc is installed in <root>/bin/protoc
get_filename_component(root_bin_dir ${Protobuf_PROTOC_EXECUTABLE} DIRECTORY)
get_filename_component(root_dir ${root_bin_dir} DIRECTORY)
get_filename_component(root_dir ${root_dir} ABSOLUTE)
set(DAWN_PROTOBUF_ROOT ${root_dir})

# set the path to the install-directory of protobuf
set(DAWN_PROTOBUF_RPATH_DIR ${DAWN_PROTOBUF_ROOT}/lib)

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
