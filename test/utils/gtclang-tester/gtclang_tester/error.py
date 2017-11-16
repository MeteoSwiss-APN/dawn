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

from sys import exit, stderr

from .config import Config
from .progressbar import TerminalController


def report_info(msg):
    """ Report an info message """
    if Config.verbose:
        print("[LOG] " + msg.rstrip())


def report_warning(msg):
    """ Issue a warning """

    term = TerminalController()
    if term.is_valid():
        print(
            term.BOLD + term.BLUE + "gtclang-tester:" + term.MAGENTA + " warning" + term.BLUE + (
                ": %s" % msg) + term.NORMAL, file=stderr)
    else:
        print("gtclang-terster: warning: %s" % msg, file=stderr)


def report_fatal_error(msg):
    """ Issue a fatal error and exit with non-zero exit status """

    term = TerminalController()
    if term.is_valid():
        print(term.BOLD + term.BLUE + "gtclang-tester:" + term.RED + " error" + term.BLUE + (
            ": %s" % msg) + term.NORMAL, file=stderr)
    else:
        print("gtclang-terster: error: %s" % msg, file=stderr)
    exit(1)
