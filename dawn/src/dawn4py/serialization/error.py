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


class Error(Exception):
    """ Thrown in case of an error. """

    pass


class ParseError(Error):
    """ Thrown in case of a parsing error. """

    pass


class SIRError(Error):
    """ Thrown in case of an invalid SIR configuration. """

    pass
