#!/usr/bin/python3
# -*- coding: utf-8 -*-
##===-----------------------------------------------------------------------------*- Python -*-===##
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

__dawn_versioninfo__ = (${VERSION_MAJOR}, ${VERSION_MINOR}, ${VERSION_PATCH})
__dawn_install_module__ = '${CMAKE_INSTALL_PREFIX}/python'
__dawn_install_dawnclib__ = '${CMAKE_INSTALL_FULL_LIBDIR}/$<TARGET_FILE_NAME:DawnC>'
