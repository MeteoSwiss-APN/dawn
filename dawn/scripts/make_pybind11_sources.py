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

from collections import namedtuple
import numbers
import os
import re
import subprocess

THIS_DIR = os.path.dirname(__file__)
DAWN_CPP_SRC_ROOT = os.path.join(THIS_DIR, os.pardir, "src", "dawn")
DAWN4PY_MODULE_DIR = os.path.join(THIS_DIR, os.pardir, "src", "dawn4py")
TEMPLATE_FILE = os.path.join(THIS_DIR, "_dawn4py.cpp.in")
OUTPUT_FILE = os.path.join(DAWN4PY_MODULE_DIR, "_dawn4py.cpp")

_CPP_TO_PYTHON_TYPE_MAPPING = {"std::string": str}


# define OPT(TYPE, NAME, DEFAULT_VALUE, OPTION, OPTION_SHORT, HELP, VALUE_NAME, HAS_VALUE, F_GROUP)
opt_regexp = re.compile(
    r'(?!//)\s*OPT\(\s*(?P<type>[^,]+)\s*,\s*(?P<name>[^,]+)\s*,\s*(?P<default_value>[^,]+)\s*,\s*"(?P<option>[^\"]+)"\s*,\s*"(?P<option_short>[^\"]*)"\s*,\s*"(?P<help>[^\"]*)"\s*,\s*"(?P<value_name>[^\""]*)"\s*,\s*(?P<has_value>[^,]+)\s*,\s*(?P<f_group>[^,]+)\s*\)',
    re.M,
)


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


def extract_options_from_files(files: list) -> list:
    """Generate a list of MemberInfo for an options struct."""

    options_info = []
    for f in files:
        with open(f, mode="r") as fo:
            for m in opt_regexp.finditer("".join(fo.readlines())):
                default_value = m.group("default_value").strip('"')
                py_type = pythonize_type(m.group("type"))
                py_default = pythonize_value(default_value, as_type=py_type)
                options_info.append(
                    MemberInfo(
                        py_name=pythonize_name(m.group("name")),
                        cpp_name=m.group("name"),
                        py_type=py_type,
                        cpp_type=m.group("type"),
                        py_default=py_default,
                        cpp_default=default_value,
                        const=False,
                        help=m.group("help"),
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


def splice_into_string(string: str, original: str, replacement: str):
    pos_start = string.find(original)
    assert pos_start >= 0 and pos_start < len(string)
    return string[0:pos_start] + replacement + string[pos_start + len(original) :]


def get_enum_values(filename: str, enum_name: str):
    with open(filename, mode="r") as f:
        code = f.read()
        m = re.search(r"enum\s*(class)?\s+" + enum_name, code)
        assert m
        enum_start = m.start()
        start_pos = code[enum_start + len(enum_name) :].find("{") + enum_start + len(enum_name) + 1
        end_pos = code[start_pos:].find("}") + start_pos
        enum_values = code[start_pos:end_pos].strip().split("\n")
        assert len(enum_values) > 0
        return list(map(lambda x: x.split("//")[0].split(",")[0].strip(), enum_values))


def make_enum_binding(py_name: str, c_name: str, values: list):
    return (
        f'py::enum_<{c_name}>(m, "{py_name}")'
        + "".join(['.value("' + val + '", ' + c_name + "::" + val + ")" for val in values])
        + ".export_values();"
    )


if __name__ == "__main__":
    print("-> Generating pybind11 bindings for Dawn...\n")

    with open(TEMPLATE_FILE, mode="r") as f:
        code = f.read()
        for py_name, c_name, filename in (
            (
                "SIRSerializerFormat",
                "dawn::SIRSerializer::Format",
                os.path.join(DAWN_CPP_SRC_ROOT, "Serialization", "SIRSerializer.h"),
            ),
            (
                "IIRSerializerFormat",
                "dawn::IIRSerializer::Format",
                os.path.join(DAWN_CPP_SRC_ROOT, "Serialization", "IIRSerializer.h"),
            ),
            (
                "PassGroup",
                "dawn::PassGroup",
                os.path.join(DAWN_CPP_SRC_ROOT, "Optimizer", "Options.h"),
            ),
            (
                "CodeGenBackend",
                "dawn::codegen::Backend",
                os.path.join(DAWN_CPP_SRC_ROOT, "CodeGen", "Options.h"),
            ),
        ):
            values = []
            if "::" in c_name:
                values = get_enum_values(filename, c_name[c_name.rfind("::") + 2 :])
            if len(values) == 0:
                values = get_enum_values(filename, c_name)
            if len(values) == 0:
                raise ValueError(f"Could not parse enum values for {py_name} in {filename}")
            enum_str = make_enum_binding(py_name, c_name, values) + "\n"
            code = splice_into_string(code, "{{ " + py_name + " }}", enum_str)

        for py_name, c_name, filename in (
            (
                "CodeGenOptions",
                "dawn::codegen::Options",
                os.path.join(DAWN_CPP_SRC_ROOT, "CodeGen", "Options.inc"),
            ),
            (
                "Options",
                "dawn::Options",
                os.path.join(DAWN_CPP_SRC_ROOT, "Optimizer", "Options.inc"),
            ),
        ):
            struct_str = (
                make_struct_binding(py_name, c_name, extract_options_from_files([filename]))
                + "\n\n"
            )
            code = splice_into_string(code, "{{ " + py_name + " }}", struct_str)

        with open(OUTPUT_FILE, mode="w") as f:
            f.write(code)
    try:
        cmd = f"clang-format -style=file -i {os.path.abspath(OUTPUT_FILE)}"
        print(f"-> Applying clang-format...  [{cmd}]\n")
        subprocess.check_call(list(cmd.split()), cwd=os.path.abspath(os.path.dirname(__file__)))
    except subprocess.CalledProcessError as e:
        print("Error found when trying to apply 'clang-format' to the generated file!")

    print("-> Done\n")
    print("pybind11 bindings for Dawn have been regenerated!")
