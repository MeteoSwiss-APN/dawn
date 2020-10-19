#!/usr/bin/env python

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

import argparse
import os

import dawn4py
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as sir_utils

OUTPUT_NAME = "global_var_stencil"
OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"
OUTPUT_PATH = f"{OUTPUT_NAME}.cpp"


def main(args: argparse.Namespace):
    interval = sir_utils.make_interval(
        SIR.Interval.Start, SIR.Interval.End, 0, 0)
    
    body_ast = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("out", [0, 0, 0]),
                sir_utils.make_binary_operator(sir_utils.make_var_access_expr(
                    "dt", is_external=True), "*", sir_utils.make_field_access_expr("in", [1, 0, 0])),
                "="),
        ]
    )

    vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(
        body_ast, interval, SIR.VerticalRegion.Forward
    )

    globals = SIR.GlobalVariableMap()
    globals.map["dt"].double_value = 0.5

    sir = sir_utils.make_sir(
        OUTPUT_FILE,
        SIR.GridType.Value("Cartesian"),
        [
            sir_utils.make_stencil(
                OUTPUT_NAME,
                sir_utils.make_ast([vertical_region_stmt]),
                [
                    sir_utils.make_field(
                        "in", sir_utils.make_field_dimensions_cartesian()),
                    sir_utils.make_field(
                        "out", sir_utils.make_field_dimensions_cartesian()),
                ],
            )
        ],
        global_variables=globals
    )

    # print the SIR
    if args.verbose:
        sir_utils.pprint(sir)

    # compile
    code = dawn4py.compile(sir, backend=dawn4py.CodeGenBackend.CXXNaive)

    # write to file
    print(f"Writing generated code to '{OUTPUT_PATH}'")
    with open(OUTPUT_PATH, "w") as f:
        f.write(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a simple stencil with globals using Dawn compiler"
    )
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        default=False,
        help="Print the generated SIR",
    )
    main(parser.parse_args())
