#!/usr/bin/env python
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

import jinja2

from collections import namedtuple
import numbers
import os
import re
import subprocess


_CPP_TO_PYTHON_TYPE_MAPPING = {"std::string": str}


# Ref: https://stackoverflow.com/a/29920015
def _camel_case_split(name: str) -> list:
    matches = re.finditer(".+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)", name)
    result = [m.group(0) for m in matches]
    return result


def pythonize_name(cpp_name: str) -> str:
    words = _camel_case_split(cpp_name)
    result = "_".join(words).lower()
    return result


def pythonize_type(cpp_type: str) -> type:
    result = _CPP_TO_PYTHON_TYPE_MAPPING.get(cpp_type, None)
    result = result or eval(cpp_type)
    return result


def pythonize_value(cpp_value: str, as_type: type):
    if as_type is bool:
        result = False if cpp_value == "false" or cpp_value == "0" else True
    elif issubclass(type, numbers.Number):
        result = eval(cpp_value)
    elif cpp_value == "nullptr":
        result = None
    elif issubclass(as_type, str):
        result = f'"{cpp_value}"'
    else:
        result = as_type(cpp_value)
    return result


MemberInfo = namedtuple(
    "MemberInfo",
    ["py_name", "cpp_name", "py_type", "cpp_type", "py_default", "cpp_default", "const", "help"],
)


def extract_dawn_compiler_options() -> list:
    """Generate options_info for the Dawn compiler options struct."""

    options_info = []
    regex = re.compile(r"OPT\(([^,]+), ?(\w+)")
    DAWN_CPP_SRC_ROOT = os.path.join(
        os.path.dirname(__file__), os.pardir, os.pardir, "src", "dawn"
    )

    # Extract info from .cpp files
    for name in [
        os.path.join(DAWN_CPP_SRC_ROOT, "Compiler", "Options.inc"),
        os.path.join(DAWN_CPP_SRC_ROOT, "Optimizer", "OptimizerOptions.inc"),
    ]:
        options_cpp = []
        with open(name, "r") as f:
            for line in f:
                line = line.strip()
                if not (line.startswith("//") or line.startswith("#")):
                    if line.startswith("OPT("):
                        m = regex.match(line)
                        type_str, name_str = m.group(1), m.group(2)
                        line = re.sub(regex, f'{name_str} = ("{type_str}" ', line)
                        options_cpp.append(line)
                    elif line:
                        if options_cpp[-1].endswith('"'):
                            options_cpp[-1] += " + " + line
                        else:
                            options_cpp[-1] += line

        #  OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP)
        options_cpp = "\n".join(options_cpp)
        for old, new in [("false", "'false'"), ("true", "'true'")]:
            options_cpp = options_cpp.replace(old, new)

        defs = {}
        exec(options_cpp, defs)

        for key, value in defs.items():
            if not key.startswith("__"):
                py_type = pythonize_type(value[0])
                py_default = pythonize_value(value[1], as_type=py_type)
                options_info.append(
                    MemberInfo(
                        py_name=pythonize_name(key),
                        cpp_name=key,
                        py_type=py_type,
                        cpp_type=value[0],
                        py_default=py_default,
                        cpp_default=value[1],
                        const=False,
                        help=value[4],
                    )
                )

    return options_info


def make_struct_binding(
    python_name: str,
    cpp_name: str,
    members_info: list,
    *,
    generate_repr=True,
    indent_size=4,
    module_var="m",
) -> str:
    """Generate pybind11 definitions for a simple struct.

    `members_info` is a `list` of `MemberInfo` namedtuples.
    """

    indent = " " * indent_size

    # Generate class definitions
    source = f'py::class_<{cpp_name}>({module_var}, "{python_name}")\n'

    # Generate __init__ method
    cpp_arg_items = []
    cpp_value_items = []
    python_arg_items = []
    for info in members_info:
        if issubclass(info.py_type, numbers.Number):
            arg_format = "{type} {name}"
        else:
            arg_format = "const {type}& {name}"
        cpp_arg_items.append(arg_format.format(type=info.cpp_type, name=info.cpp_name))
        cpp_value_items.append(info.cpp_name)
        default = f'"{info.cpp_default}"' if info.py_type == str else info.cpp_default
        python_arg_items.append(f'py::arg("{info.py_name}") = {default}')

    cpp_init_args = ", ".join(cpp_arg_items)
    cpp_init_values = ", ".join(cpp_value_items)
    python_init_args = ", ".join(python_arg_items)
    lines = [
        f"{indent}.def(py::init(",
        f"{indent}{indent}[]({cpp_init_args}) {{",
        f"{indent}{indent}{indent}return {cpp_name}{{{cpp_init_values}}};",
        f"{indent}{indent}}}), {python_init_args})",
        "",
    ]
    source += "\n".join(lines)

    # Generate member definitions
    lines = []
    for info in members_info:
        line = f'{indent}.def_readwrite("{info.py_name}", &{cpp_name}::{info.cpp_name})'
        lines.append(line)
    source += "\n".join(lines)

    # Generate __repr__ method
    lines = []
    if generate_repr:
        lines.append(f'\n{indent}.def("__repr__",')
        lines.append(f"{indent + indent}[](const {cpp_name} &self) {{")

        lines.append(f"{indent + indent + indent}std::ostringstream ss;")
        ss_line = f"{indent + indent + indent}ss "
        header = ""
        for info in members_info:
            value_str = f"self.{info.cpp_name}"
            if info.py_type == str:
                value_str = f'"\\"" << {value_str} << "\\""'
            ss_line += f'{header} << "{info.py_name}=" << {value_str}'
            header = f' << ",\\n{indent}"'
        ss_line += ";"
        lines.append(ss_line)

        lines.append(
            f'{indent + indent + indent}return "{python_name}(\\n{indent}" + ss.str() + "\\n)";'
        )
        lines.append(f"{indent + indent}}})")
        source += "\n".join(lines)
    source += ";"

    return source


TEMPLATE_NAME = "_dawn4py.cpp.in"
OUTPUT_NAME = TEMPLATE_NAME.replace(".in", "")

if __name__ == "__main__":
    print("-> Generating pybind11 bindings for Dawn...\n")
    options_infos = extract_dawn_compiler_options()
    options_class_def = make_struct_binding("Options", "dawn::Options", options_infos)

    TEMPLATE_FILE = os.path.join(os.path.dirname(__file__), TEMPLATE_NAME)
    OUTPUT_FILE = os.path.join(os.path.dirname(__file__), OUTPUT_NAME)
    with open(TEMPLATE_FILE, "r") as f:
        template = jinja2.Template(f.read())
    source = template.render(options_class_def=options_class_def)
    with open(OUTPUT_FILE, "w") as f:
        f.write(source)

    try:
        cmd = f"clang-format -style=file -i {os.path.abspath(OUTPUT_FILE)}"
        print(f"-> Applying clang-format...  [{cmd}]\n")
        subprocess.check_call(list(cmd.split()), cwd=os.path.abspath(os.path.dirname(__file__)))
    except subprocess.CalledProcessError as e:
        print("Error found when trying to apply 'clang-format' to the generated file!")

    print("-> Done\n")
    print("pybind11 bindings for Dawn have been generated!")
    print(
        f"Check the output file at '{OUTPUT_FILE}' and if "
        "everything is fine, copy the file to its final destination at "
        "'${DAWN_ROOT}/dawn/src/dawn4py'"
    )
