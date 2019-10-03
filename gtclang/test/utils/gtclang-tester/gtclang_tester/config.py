#!/usr/bin/python3
# -*- coding: utf-8 -*-
##===-----------------------------------------------------------------------------*- Python -*-===##
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

class Config(object):
    """ Global configuration options """

    # Path to gtclang
    gtclang = "gtclang"

    # Verbosity
    verbose = False

    # GridTools flags
    gridtools_flags = "-std=c++11"

    # C++ compiler
    cxx = "c++"

    # Generate reference files
    generate_reference = False

    # Don't show any progressbar
    no_progressbar = False