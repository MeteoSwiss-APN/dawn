##===------------------------------------------------------------------------------*- CMake -*-===##
##                         _       _
##                        | |     | |
##                    __ _| |_ ___| | __ _ _ __   __ _ 
##                   / _` | __/ __| |/ _` | '_ \ / _` |
##                  | (_| | || (__| | (_| | | | | (_| |
##                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
##                    __/ |                       __/ |
##                   |___/                       |___/
##
##
##  This file is distributed under the MIT License (MIT). 
##  See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

include(msbuildSetDownloadDir)
include(msbuildFindPackage)

# Set the default download directory (define GTCLANG_ALL_DOWNLOAD_DIR)
msbuild_set_download_dir()

#
# Protobuf
#
set(protobuf_version "3.4.0")
set(protobuf_version_short "3.4")

msbuild_find_package(
  PACKAGE Protobuf
  PACKAGE_ARGS ${protobuf_version_short} NO_MODULE 
  REQUIRED_VARS Protobuf_DIR
  BUILD_VERSION ${protobuf_version}
  ADDITIONAL
    DOWNLOAD_DIR ${MSBUILD_DOWNLOAD_DIR}
    URL "https://github.com/google/protobuf/archive/v${protobuf_version}.tar.gz"
    URL_MD5 "1d077a7d4db3d75681f5c333f2de9b1a"
)

#dawn_report_result("Package summary" ${GTCLANG_ALL_PACKAGE_INFO})
