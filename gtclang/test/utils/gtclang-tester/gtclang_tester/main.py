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

from optparse import OptionParser

from .config import Config
from .error import report_fatal_error
from .parser import parse
from .runner import run


def main():
    parser = OptionParser("gtclang-tester.py [directories] [options]")

    parser.add_option("--gtclang", dest="gtclang",
                      help="path to the gtclang executable", metavar="PATH")
    parser.add_option("--cxx", dest="cxx",
                      help="path to the c++ compiler used to compile gridtools C++ code",
                      metavar="PATH")
    parser.add_option("--gridtools_flags", dest="gridtools_flags",
                      help="semicolon separated list of compile flags required to compile gridtools C++ code",
                      metavar="FLAGS")
    parser.add_option("--no-progressbar", dest="no_progressbar", action="store_true",
                      help="Don't show any progressbar")
    parser.add_option("-v", "--verbose", dest="verbose", action="store_true",
                      help="verbose logging")
    parser.add_option("-g", "--generate-reference", dest="generate_reference", action="store_true",
                      help="generate reference file for file commands")

    (options, args) = parser.parse_args()

    if options.generate_reference:
        Config.generate_reference = True

    if options.gtclang:
        Config.gtclang = options.gtclang

    if options.verbose:
        Config.verbose = True

    if options.gridtools_flags:
        Config.gridtools_flags = options.gridtools_flags

    if options.cxx:
        Config.cxx = options.cxx

    if options.no_progressbar:
        Config.no_progressbar = True

    if not args:
        report_fatal_error('no input directories given')

    tests = parse(args)
    return run(tests)
