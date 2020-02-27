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

def convert_sir(sir):
    if isinstance(sir, int) or isinstance(sir, float):
        return
    for elem in sir:
        if isinstance(elem, str):
            key = elem
            new_key = pythonize_name(key)
            if new_key != key:
                sir[new_key] = sir.pop(key)
                key = new_key
            if isinstance(sir[key], dict):
                convert_sir(sir[key])
            elif isinstance(sir[key], list):
                for sub in sir[key]:
                    convert_sir(sub)
