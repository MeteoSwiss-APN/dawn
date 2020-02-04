#!/usr/bin/env python

##===-----------------------------------------------------------------------------*- Python -*-===##
# _
# | |
# __| | __ ___      ___ ___
# / _` |/ _` \ \ /\ / / '_  |
# | (_| | (_| |\ V  V /| | | |
# \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
##
##
# This file is distributed under the MIT License (MIT).
# See LICENSE.txt for details.
##
##===------------------------------------------------------------------------------------------===##

"""Sparse Dimensions stencil HIR generator

This program creates the HIR corresponding to an unstructured stencil including sparse dimensions
using the SIR serialization Python API."""

import argparse
import os

import dawn4py
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as sir_utils

OUTPUT_NAME = "sparse_dimensions"
OUTPUT_FILE = f"{OUTPUT_NAME}.cpp"
OUTPUT_PATH = os.path.join(os.path.dirname(
    __file__), "data", f"{OUTPUT_NAME}.cpp")


def main(args: argparse.Namespace):
    interval = sir_utils.make_interval(
        SIR.Interval.Start, SIR.Interval.End, 0, 0)

    # create the out = reduce(sparse_CE * in) statement
    body_ast = sir_utils.make_ast(
        [
            sir_utils.make_assignment_stmt(
                sir_utils.make_field_access_expr("out"),
                sir_utils.make_reduction_over_neighbor_expr(
                    "+",
                    sir_utils.make_binary_operator(
                        sir_utils.make_field_access_expr("sparse_CE"), "*", sir_utils.make_field_access_expr("in")),
                    sir_utils.make_literal_access_expr(
                        "1.0", SIR.BuiltinType.Float),
                    lhs_location=SIR.LocationType.Value('Cell'),
                    rhs_location=SIR.LocationType.Value('Edge')
                ),
                "=",
            )
        ]
    )

    vertical_region_stmt = sir_utils.make_vertical_region_decl_stmt(
        body_ast, interval, SIR.VerticalRegion.Forward)

    sir = sir_utils.make_sir(
        OUTPUT_FILE,
        SIR.GridType.Value("Unstructured"),
        [
            sir_utils.make_stencil(
                OUTPUT_NAME,
                sir_utils.make_ast([vertical_region_stmt]),
                [
                    sir_utils.make_field("in", sir_utils.make_field_dimensions_unstructured(
                        [SIR.LocationType.Value('Edge')], 1)),
                    sir_utils.make_field("sparse_CE", sir_utils.make_field_dimensions_unstructured(
                        [SIR.LocationType.Value('Cell'), SIR.LocationType.Value('Edge')], 1)),
                    sir_utils.make_field("out", sir_utils.make_field_dimensions_unstructured(
                        [SIR.LocationType.Value('Cell')], 1))
                ],
            ),
        ],
    )

    # print the SIR
    if args.verbose:
        sir_utils.pprint(sir)

    # compile
    code = dawn4py.compile(sir, backend="c++-naive-ico")

    # write to file
    print(f"Writing generated code to '{OUTPUT_PATH}'")
    with open(OUTPUT_PATH, "w") as f:
        f.write(code)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate a simple unstructured copy stencil using Dawn compiler")
    parser.add_argument(
        "-v", "--verbose", dest="verbose", action="store_true", default=False, help="Print the generated SIR",
    )
    main(parser.parse_args())
