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

if __name__ != '__main__':
    try:
        __import__('pkg_resources').declare_namespace(__name__)
    except ImportError:
        __path__ = __import__('pkgutil').extend_path(__path__, __name__)

from .config import __dawn_versioninfo__

__versioninfo__ = __dawn_versioninfo__
__version__ = '.'.join(str(v) for v in __versioninfo__)

#
# Check python version
#
from sys import version_info as __dawn_python_version_info

if __dawn_python_version_info < (3, 4):
    from sys import version as __dawn_python_version

    raise Exception(
        "Dawn (%s) requires at least Python 3.4 (running on %s)" % (
            __version__, __dawn_python_version.replace('\n', ' ')))

#
# Import submodules
#
from .sir import *

__all__ = [
    # sir.py
    # TODO
]
