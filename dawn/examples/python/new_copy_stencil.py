import os

import dawn4py
from dawn4py.serialization import SIR
from dawn4py.serialization import utils as ir_utils


interval = ir_utils.make_interval(SIR.Interval.Start, SIR.Interval.End, 0, 0)

# create the out = in[i+1] statement
body_ast = ir_utils.make_ast(
    [
        ir_utils.make_assignment_stmt(
            ir_utils.make_field_access_expr("out", [0, 0, 0]),
            ir_utils.make_field_access_expr("in", [1, 0, 0]),
            "=",
        )
    ]
)

vertical_region_stmt = ir_utils.make_vertical_region_decl_stmt(
    body_ast, interval, SIR.VerticalRegion.Forward
)


sir = ir_utils.make_sir(
    "copy_stencil.cpp",
    [
        ir_utils.make_stencil(
            "copy_stencil",
            ir_utils.make_ast([vertical_region_stmt]),
            [ir_utils.make_field("in"), ir_utils.make_field("out")],
        )
    ],
)

# print the SIR
ir_utils.pprint(sir)

# compile
code = dawn4py.generate(sir, dawn4py.Backend.GridTools)

# write to file
with open(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "new_copy_stencil.cpp"), "w"
) as f:
    f.write(code)
