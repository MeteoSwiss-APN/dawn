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


"""General Python utilities."""

import re
from typing import List

import attr

NOTHING = object()


def camel_case_split(name: str) -> List[str]:
    """Split a CamelCase name in its components.

    From: https://stackoverflow.com/a/29920015
    """
    matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", name)
    result = [m.group(0) for m in matches]
    return result


def pythonize_name(name: str) -> str:
    words = camel_case_split(name)
    result = "_".join(words).lower()
    return result
