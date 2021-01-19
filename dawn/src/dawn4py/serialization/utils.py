# -*- coding: utf-8 -*-
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


"""
Convenience functions to serialize/deserialize and print SIR and IIR objects.
"""


import textwrap

from enum import Enum
from collections.abc import Iterable
from typing import List, TypeVar, NewType

from google.protobuf import json_format

from .error import ParseError, SIRError
from .SIR.SIR_pb2 import *
from .SIR.statements_pb2 import *
from .SIR.enums_pb2 import *
from .. import utils

__all__ = [
    "make_sir",
    "make_stencil",
    "make_type",
    "make_field_dimensions_cartesian",
    "make_field_dimensions_unstructured",
    "make_field",
    "make_ast",
    "make_interval",
    "make_vertical_region",
    "make_stencil_call",
    "make_stmt",
    "make_block_stmt",
    "make_loop_stmt",
    "make_expr_stmt",
    "make_return_stmt",
    "make_var_decl_stmt",
    "make_vertical_region_decl_stmt",
    "make_stencil_call_decl_stmt",
    "make_boundary_condition_decl_stmt",
    "make_if_stmt",
    "make_expr",
    "make_unary_operator",
    "make_binary_operator",
    "make_assignment_expr",
    "make_assignment_stmt",
    "make_ternary_operator",
    "make_fun_call_expr",
    "make_stencil_fun_call_expr",
    "make_stencil_fun_arg_expr",
    "make_stencil_function_arg",
    "make_stencil_function",
    "make_var_access_expr",
    "make_field_access_expr",
    "make_literal_access_expr",
    "make_weights",
    "make_reduction_over_neighbor_expr",
    "to_bytes",
    "from_bytes",
    "to_json",
    "from_json",    
]

ExprType = TypeVar(
    "ExprType",
    Expr,
    UnaryOperator,
    BinaryOperator,
    AssignmentExpr,
    TernaryOperator,
    FunCallExpr,
    StencilFunCallExpr,
    StencilFunArgExpr,
    VarAccessExpr,
    FieldAccessExpr,
    LiteralAccessExpr,
    ReductionOverNeighborExpr,
)

StmtType = TypeVar(
    "StmtType",
    Stmt,
    BlockStmt,
    ExprStmt,
    ReturnStmt,
    VarDeclStmt,
    VerticalRegionDeclStmt,
    StencilCallDeclStmt,
    BoundaryConditionDeclStmt,
    IfStmt,
    LoopStmt,
)

# Can't pass SIR.enums_pb2.LocationType as argument because it doesn't contain the value
LocationTypeValue = NewType("LocationTypeValue", int)


def make_sir(
    filename: str,
    grid_type: GridType,
    stencils: List[Stencil],
    functions: List[StencilFunction] = [],
    global_variables: GlobalVariableMap = None,
) -> SIR:
    """ Create a SIR

    :param filename:          Source filename
    :param grid_type:         Grid type definition
    :param stencils:          list of stencils that compose the SIR
    :param functions:         list of functions used in the SIR
    :param global_variables:  global variable map used in the SIR
    """

    sir = SIR()
    sir.filename = filename
    sir.gridType = grid_type
    sir.stencils.extend(stencils)
    sir.stencil_functions.extend(functions)
    if global_variables:
        sir.global_variables.CopyFrom(global_variables)

    return sir


def make_stencil(name: str, ast: AST, fields: List[Field]) -> Stencil:
    """ Create a Stencil

    :param name:      Name of the stencil
    :param ast:       AST with stmts of the stencil
    :param fields:    list of input-output fields
    """

    stencil = Stencil()
    stencil.name = name
    stencil.ast.CopyFrom(ast)
    stencil.fields.extend(fields)

    return stencil


def make_type(builtin_type_or_name, is_const: bool = False, is_volatile: bool = False):
    """ Wrap a concrete expression (e.g VarAccessExpr) into an Expr object

    :param builtin_type_or_name: Either an Enum of BuiltinType or the name of the custom type in
                                 case it is not a builtin type).
    :type builtin_type_or_name:  Union[BuiltinType.TypeID, str, int]
    :param is_const:             Is the type const qualified?
    :param is_volatile:          Is the type volatile qualified?
    """
    t = Type()
    t.is_const = is_const
    t.is_volatile = is_volatile
    if isinstance(builtin_type_or_name, str):
        t.name = builtin_type_or_name
    elif isinstance(builtin_type_or_name, (type(BuiltinType.TypeID), int)):
        builtin_type = BuiltinType()
        builtin_type.type_id = builtin_type_or_name
        t.builtin_type.CopyFrom(builtin_type)
    else:
        raise TypeError(
            "expected 'builtin_type_or_name' to be either of type 'dawn.sir.BuiltinType.TypeID'"
            + "or 'str' (got {})".format(type(builtin_type_or_name))
        )
    return t


def make_field_dimensions_cartesian(mask: List[int] = None) -> FieldDimensions:
    """ Create FieldDimensions of cartesian type

    :param mask: mask to identify which cartesian dimensions are legal (default is [1, 1, 1])
    """

    if mask is None:
        mask = [1, 1, 1]
    assert len(mask) == 3

    horizontal_dim = CartesianDimension()
    horizontal_dim.mask_cart_i = mask[0]
    horizontal_dim.mask_cart_j = mask[1]

    dims = FieldDimensions()
    dims.cartesian_horizontal_dimension.CopyFrom(horizontal_dim)
    dims.mask_k = mask[2]
    return dims


def make_field_dimensions_unstructured(
    locations: List[LocationTypeValue], mask_k: int, include_center: bool = False
) -> FieldDimensions:
    """ Create FieldDimensions of unstructured type

    :locations:    a list of location types of the field. first entry is the dense part, additional entries are the (optional) sparse part
    :mask_k:       mask to identify if the vertical dimension is legal
    :sparse_part:  optional sparse part encoded by a neighbor chain
    """

    assert len(locations) >= 1
    dims = FieldDimensions()
    iter_space = UnstructuredIterationSpace()
    horizontal_dim = UnstructuredDimension()            
    iter_space.chain.extend(locations)
    iter_space.include_center = include_center
    horizontal_dim.iter_space.CopyFrom(iter_space)
    dims.unstructured_horizontal_dimension.CopyFrom(horizontal_dim)    
    dims.mask_k = mask_k
    return dims


def make_field_dimensions_vertical() -> FieldDimensions:
    """ Create Field dimension in the vertical only
    """
    dims = FieldDimensions()
    dims.mask_k = True
    return dims


def make_field(name: str, dimensions: FieldDimensions, is_temporary: bool = False) -> Field:
    """ Create a Field

    :param name:         Name of the field
    :param dimensions:   dimensions of the field (use make_field_dimensions_*)
    :param is_temporary: Is it a temporary field?
    """

    field = Field()
    field.name = name
    field.is_temporary = is_temporary
    field.field_dimensions.CopyFrom(dimensions)
    return field


def make_vertical_field(name: str, is_temporary: bool = False) -> Field:
    """ Create a vertical Field, i.e. a field with no horizontal dimensions

    :param name:         Name of the field
    :param is_temporary: Is it a temporary field?
    """
    return make_field(name, make_field_dimensions_vertical(), is_temporary)


def make_ast(root: List[StmtType]) -> AST:
    """ Create an AST

    :param root:    Root node of the AST (needs to be of type BlockStmt)
    """
    ast = AST()

    block_stmt = make_block_stmt(root)

    ast.root.CopyFrom(make_stmt(block_stmt))
    return ast


def make_interval(
    lower_level, upper_level, lower_offset: int = 0, upper_offset: int = 0
) -> Interval:
    """ Create an Interval

    Representation of a vertical interval, given by a lower and upper bound where a bound
    is represented by a level and an offset (`bound = level + offset`)

     The Interval has to satisfy the following invariants:
      - `lower_level >= Interval.Start`
      - `upper_level <= Interval.End`
      - `(lower_level + lower_offset) <= (upper_level + upper_offset)`

    :param lower_level:    Lower level integer between `[Interval.Start, Interval.End]`
    :param upper_level:    Lower level integer between `[Interval.Start, Interval.End]`
    :param lower_offset:   Lower offset
    :param upper_offset:   Upper offset
    """
    interval = Interval()

    if lower_level in (Interval.Start, Interval.End):
        interval.special_lower_level = lower_level
    else:
        interval.lower_level = lower_level

    if upper_level in (Interval.Start, Interval.End):
        interval.special_upper_level = upper_level
    else:
        interval.upper_level = upper_level

    interval.lower_offset = lower_offset
    interval.upper_offset = upper_offset
    return interval


def make_magic_num_interval(
    lower_level, upper_level, lower_offset: int = 0, upper_offset: int = 0
) -> Interval:
    """ Create an Interval

    Representation of a vertical interval, given by a lower and upper bound where a bound
    is represented by a level and an offset (`bound = level + offset`)

    """
    interval = Interval()

    interval.lower_level = lower_level
    interval.upper_level = upper_level

    interval.lower_offset = lower_offset
    interval.upper_offset = upper_offset
    return interval


def make_vertical_region(
    ast: AST,
    interval: Interval,
    loop_order: VerticalRegion.LoopOrder,
    i_range: Interval = None,
    j_range: Interval = None,
) -> VerticalRegion:
    """ Create a VerticalRegion

    :param ast:         Syntax tree of the body of the vertical region
    :param interval:    Vertical interval
    :param loop_order:  Vertical loop order of execution
    """
    vr = VerticalRegion()
    vr.ast.CopyFrom(ast)
    vr.interval.CopyFrom(interval)
    vr.loop_order = loop_order
    if i_range is not None:
        vr.i_range.CopyFrom(i_range)
    if j_range is not None:
        vr.j_range.CopyFrom(j_range)
    return vr


def make_stencil_call(callee: str, arguments: List[str]) -> StencilCall:
    """ Create a StencilCall

    :param callee:      Name of the called stencil (i.e callee)
    :param arguments:   Fields passed as arguments during the stencil call
    """
    call = StencilCall()
    call.callee = callee
    call.arguments.extend(arguments)
    return call


def make_stmt(stmt: StmtType):
    """ Wrap a concrete statement (e.g ExprStmt) into an Stmt object

    :param stmt: Statement to wrap
    :return: Statement wrapped into Stmt
    """
    if isinstance(stmt, Stmt):
        return stmt
    wrapped_stmt = Stmt()

    if isinstance(stmt, BlockStmt):
        wrapped_stmt.block_stmt.CopyFrom(stmt)
    elif isinstance(stmt, ExprStmt):
        wrapped_stmt.expr_stmt.CopyFrom(stmt)
    elif isinstance(stmt, ReturnStmt):
        wrapped_stmt.return_stmt.CopyFrom(stmt)
    elif isinstance(stmt, VarDeclStmt):
        wrapped_stmt.var_decl_stmt.CopyFrom(stmt)
    elif isinstance(stmt, VerticalRegionDeclStmt):
        wrapped_stmt.vertical_region_decl_stmt.CopyFrom(stmt)
    elif isinstance(stmt, StencilCallDeclStmt):
        wrapped_stmt.stencil_call_decl_stmt.CopyFrom(stmt)
    elif isinstance(stmt, BoundaryConditionDeclStmt):
        wrapped_stmt.boundary_condition_decl_stmt.CopyFrom(stmt)
    elif isinstance(stmt, IfStmt):
        wrapped_stmt.if_stmt.CopyFrom(stmt)
    elif isinstance(stmt, LoopStmt):
        wrapped_stmt.loop_stmt.CopyFrom(stmt)
    else:
        raise SIRError("cannot create Stmt from type {}".format(type(stmt)))
    return wrapped_stmt


def make_block_stmt(statements: List[StmtType]) -> BlockStmt:
    """ Create an UnaryOperator

    :param statements: List of statements that compose the block
    """
    stmt = BlockStmt()
    if isinstance(statements, Iterable):
        stmt.statements.extend([make_stmt(s)
                                for s in statements if not isinstance(s, Field)])
    else:
        stmt.statements.extend([make_stmt(statements)])
    return stmt


def make_loop_stmt(block: List[StmtType],  chain: List[LocationTypeValue], include_center: bool = False) -> LoopStmt:
    """ Create an For Loop

    :param block: List of statements that compose the body of the loop
    """
    stmt = LoopStmt()
    stmt.statements.CopyFrom(make_stmt(make_block_stmt(block)))
    loop_descriptor_chain = LoopDescriptorChain()
    iter_space = UnstructuredIterationSpace()
    iter_space.chain.extend(chain)
    iter_space.include_center = include_center
    loop_descriptor_chain.iter_space.CopyFrom(iter_space)
    stmt.loop_descriptor.loop_descriptor_chain.CopyFrom(loop_descriptor_chain)

    return stmt


def make_expr_stmt(expr: ExprType) -> ExprStmt:
    """ Create an ExprStmt

    :param expr:      Expression.
    """
    stmt = ExprStmt()
    stmt.expr.CopyFrom(make_expr(expr))
    return stmt


def make_return_stmt(expr: ExprType) -> ReturnStmt:
    """ Create an ReturnStmt

    :param expr:      Expression to return.
    """
    stmt = ReturnStmt()
    stmt.expr.CopyFrom(make_expr(expr))
    return stmt


def make_var_decl_stmt(
    type: Type, name: str, dimension: int = 0, op: str = "=", init_list=None
) -> VarDeclStmt:
    """ Create an ReturnStmt

    :param type:        Type of the variable.
    :param name:        Name of the variable.
    :param dimension:   Dimension of the array or 0 for variables.
    :param op:          Operation used for initialization.
    :param init_list:   List of expression used for array initialization or just 1 element for
                        variable initialization.
    """
    stmt = VarDeclStmt()
    stmt.type.CopyFrom(type)
    stmt.name = name
    stmt.dimension = dimension
    stmt.op = op
    if init_list:
        if isinstance(init_list, Iterable):
            stmt.init_list.extend([make_expr(expr) for expr in init_list])
        else:
            stmt.init_list.extend([make_expr(init_list)])

    return stmt


def make_stencil_call_decl_stmt(stencil_call: StencilCall) -> StencilCallDeclStmt:
    """ Create a StencilCallDeclStmt

    :param stencil_call:   Stencil call.
    """
    stmt = StencilCallDeclStmt()
    stmt.stencil_call.CopyFrom(stencil_call)
    return stmt


def make_vertical_region_decl_stmt(vertical_region: VerticalRegion) -> VerticalRegionDeclStmt:
    """ Create a VerticalRegionDeclStmt

    :param vertical_region:   Vertical region.
    """
    stmt = VerticalRegionDeclStmt()
    stmt.vertical_region.CopyFrom(vertical_region)
    return stmt


def make_vertical_region_decl_stmt(
    ast: AST,
    interval: Interval,
    loop_order: VerticalRegion.LoopOrder,
    IRange: Interval = None,
    JRange: Interval = None,
) -> VerticalRegionDeclStmt:
    """ Create a VerticalRegionDeclStmt

    :param vertical_region:   Vertical region.
    """
    stmt = VerticalRegionDeclStmt()
    stmt.vertical_region.CopyFrom(make_vertical_region(
        ast, interval, loop_order, IRange, JRange))
    return stmt


def make_boundary_condition_decl_stmt(
    functor: str, fields: List[str]
) -> BoundaryConditionDeclStmt:
    """ Create a BoundaryConditionDeclStmt

    :param functor:  Identifier of the boundary condition functor.
    :param fields:   List of field arguments to apply the functor to.
    """
    stmt = BoundaryConditionDeclStmt()
    stmt.functor = functor
    stmt.fields.extend(fields)
    return stmt


def make_if_stmt(cond_part: StmtType, then_part: StmtType, else_part: StmtType = None) -> IfStmt:
    """ Create an ReturnStmt

    :param cond_part:   Condition part.
    :param then_part:   Then part.
    :param else_part:   Else part.
    """
    stmt = IfStmt()
    stmt.cond_part.CopyFrom(make_stmt(cond_part))
    stmt.then_part.CopyFrom(make_stmt(then_part))
    if else_part:
        stmt.else_part.CopyFrom(make_stmt(else_part))
    return stmt


def make_expr(expr: ExprType):
    """ Wrap a concrete expression (e.g VarAccessExpr) into an Expr object

    :param expr: Expression to wrap
    :return: Expression wrapped into Expr
    """
    if isinstance(expr, Expr):
        return expr
    wrapped_expr = Expr()

    if isinstance(expr, UnaryOperator):
        wrapped_expr.unary_operator.CopyFrom(expr)
    elif isinstance(expr, BinaryOperator):
        wrapped_expr.binary_operator.CopyFrom(expr)
    elif isinstance(expr, AssignmentExpr):
        wrapped_expr.assignment_expr.CopyFrom(expr)
    elif isinstance(expr, TernaryOperator):
        wrapped_expr.ternary_operator.CopyFrom(expr)
    elif isinstance(expr, FunCallExpr):
        wrapped_expr.fun_call_expr.CopyFrom(expr)
    elif isinstance(expr, StencilFunCallExpr):
        wrapped_expr.stencil_fun_call_expr.CopyFrom(expr)
    elif isinstance(expr, StencilFunArgExpr):
        wrapped_expr.stencil_fun_arg_expr.CopyFrom(expr)
    elif isinstance(expr, VarAccessExpr):
        wrapped_expr.var_access_expr.CopyFrom(expr)
    elif isinstance(expr, FieldAccessExpr):
        wrapped_expr.field_access_expr.CopyFrom(expr)
    elif isinstance(expr, LiteralAccessExpr):
        wrapped_expr.literal_access_expr.CopyFrom(expr)
    elif isinstance(expr, ReductionOverNeighborExpr):
        wrapped_expr.reduction_over_neighbor_expr.CopyFrom(expr)
    else:
        raise SIRError("cannot create Expr from type {}".format(type(expr)))
    return wrapped_expr


def make_unary_operator(op: str, operand: ExprType) -> UnaryOperator:
    """ Create an UnaryOperator

    :param op:      Operation (e.g "+" or "-").
    :param operand:  Expression to apply the operation.
    """
    expr = UnaryOperator()
    expr.op = op
    expr.operand.CopyFrom(make_expr(operand))
    return expr


def make_binary_operator(left: ExprType, op: str, right: ExprType) -> BinaryOperator:
    """ Create a BinaryOperator

    :param left:    Left-hand side.
    :param op:      Operation (e.g "+" or "-").
    :param right:   Right-hand side.
    """
    expr = BinaryOperator()
    expr.op = op
    expr.left.CopyFrom(make_expr(left))
    expr.right.CopyFrom(make_expr(right))
    return expr


def make_assignment_stmt(left: ExprType, right: ExprType, op: str = "=") -> ExprStmt:
    """ Create an AssignmentStmt

    :param left:    Left-hand side.
    :param right:   Right-hand side.
    :param op:      Operation (e.g "=" or "+=").
    """
    return make_expr_stmt(make_assignment_expr(left, right, op))


def make_assignment_expr(left: ExprType, right: ExprType, op: str = "=") -> AssignmentExpr:
    """ Create an AssignmentExpr

    :param left:    Left-hand side.
    :param right:   Right-hand side.
    :param op:      Operation (e.g "=" or "+=").
    """
    expr = AssignmentExpr()
    expr.op = op
    expr.left.CopyFrom(make_expr(left))
    expr.right.CopyFrom(make_expr(right))
    return expr


def make_ternary_operator(cond: ExprType, left: ExprType, right: ExprType) -> TernaryOperator:
    """ Create a TernaryOperator

    :param cond:    Condition.
    :param left:    Left-hand side.
    :param right:   Right-hand side.
    """
    expr = TernaryOperator()
    expr.cond.CopyFrom(make_expr(cond))
    expr.left.CopyFrom(make_expr(left))
    expr.right.CopyFrom(make_expr(right))
    return expr


def make_fun_call_expr(callee: str, arguments: List[ExprType]) -> FunCallExpr:
    """ Create a FunCallExpr

    :param callee:        Identifier of the function (i.e callee).
    :param arguments:     List of arguments.
    """
    expr = FunCallExpr()
    expr.callee = callee
    expr.arguments.extend([make_expr(arg) for arg in arguments])
    return expr


def make_stencil_function_arg(arg):
    result = StencilFunctionArg()
    if isinstance(arg, Field):
        result.field_value.CopyFrom(arg)
    else:
        raise SIRError("arg type not implemented in dawn4py")
    return result


def make_stencil_function(
    name: str, asts: List[AST], intervals: List[Interval], arguments: List[StencilFunctionArg]
):
    result = StencilFunction()
    result.name = name
    result.asts.extend(asts)
    result.intervals.extend(intervals)
    result.arguments.extend(arguments)
    return result


def make_stencil_fun_call_expr(callee: str, arguments: List[ExprType]) -> StencilFunCallExpr:
    """ Create a StencilFunCallExpr

    :param callee:        Identifier of the function (i.e callee).
    :param arguments:     List of arguments.
    """
    expr = StencilFunCallExpr()
    expr.callee = callee
    expr.arguments.extend([make_expr(arg) for arg in arguments])
    return expr


def make_unstructured_offset(has_offset: bool = False) -> UnstructuredOffset:
    unstructured_offset = UnstructuredOffset()
    unstructured_offset.has_offset = has_offset
    return unstructured_offset


def make_stencil_fun_arg_expr(
    direction: Dimension.Direction, offset: int = 0, argument_index: int = -1
) -> StencilFunArgExpr:
    """ Create a StencilFunArgExpr

    :param direction:       Direction of the argument.
    :param offset:          Offset to the dimension.
    :param argument_index:  Index of the argument of the stencil function in the outer scope.
                            If unused, the value *has* to be set to -1.
    """
    dim = Dimension()
    dim.direction = direction

    expr = StencilFunArgExpr()
    expr.dimension.CopyFrom(dim)
    expr.offset = offset
    expr.argument_index = argument_index
    return expr


def make_unstructured_field_access_expr(
    name: str, horizontal_offset: UnstructuredOffset = None, vertical_shift: int = 0, vertical_indirection: str = None
) -> FieldAccessExpr:
    expr = FieldAccessExpr()
    expr.name = name
    if horizontal_offset is None:
        expr.unstructured_offset.CopyFrom(make_unstructured_offset(False))
    else:
        expr.unstructured_offset.CopyFrom(horizontal_offset)
    expr.vertical_shift = vertical_shift
    if vertical_indirection is not None:
        expr.vertical_indirection = vertical_indirection
    return expr


def make_field_access_expr(
    name: str,
    offset=None,
    argument_map: List[int] = [-1, -1, -1],
    argument_offset: List[int] = [0, 0, 0],
    negate_offset: bool = False,
) -> FieldAccessExpr:
    """ Create a FieldAccessExpr.

    :param name:            Name of the field.
    :param offset:          Static offset.
    :param argument_map:    Mapping of the directional and offset arguments of the stencil function.
    :param argument_offset: Offset to the directional and offset arguments.
    :param negate_offset:   Negate the offset in the end.
    """
    assert offset is None or isinstance(offset, list)

    expr = FieldAccessExpr()
    expr.name = name
    if offset is None:
        expr.zero_offset.SetInParent()

    elif len(offset) == 3:
        assert all(isinstance(x, int) for x in offset)

        expr.cartesian_offset.i_offset = offset[0]
        expr.cartesian_offset.j_offset = offset[1]

        expr.vertical_shift = offset[2]

    elif len(offset) == 2:
        assert isinstance(offset[0], bool)
        assert isinstance(offset[1], int)
        assert argument_map == [-1, -1, -1]
        assert argument_offset == [0, 0, 0]
        assert negate_offset == False

        expr.unstructured_offset.has_offset = offset[0]
        expr.vertical_shift = offset[1]

    else:
        assert False
    expr.argument_map.extend(argument_map)
    expr.argument_offset.extend(argument_offset)
    expr.negate_offset = negate_offset
    return expr


def make_var_access_expr(
    name: str, index: ExprType = None, is_external: bool = False
) -> VarAccessExpr:
    """ Create a VarAccessExpr.

    :param name:        Name of the variable.
    :param index:       Is it an array access (i.e var[2])?.
    :param is_external: Is this an access to a external variable (e.g a global)?
    """
    expr = VarAccessExpr()
    expr.name = name
    expr.is_external = is_external
    if index:
        expr.index.CopyFrom(make_expr(index))
    return expr


def make_literal_access_expr(value: str, type: BuiltinType.TypeID) -> LiteralAccessExpr:
    """ Create a LiteralAccessExpr.

    :param value:   Value of the literal (e.g "1.123123").
    :param type:    Builtin type id of the literal.
    """
    builtin_type = BuiltinType()
    if builtin_type.type_id != type:
        builtin_type.type_id = type

    expr = LiteralAccessExpr()
    expr.value = value
    expr.type.CopyFrom(builtin_type)
    return expr


def make_weights(weights) -> List[Expr]:
    """ Create a weights vector

    :param weights:         List of weights expressed with python primitive types
    """
    assert len(weights) != 0
    proto_weights = []
    for weight in weights:
        proto_weight = Expr()
        proto_weight.CopyFrom(make_expr(weight))
        proto_weights.append(proto_weight)

    return proto_weights


def make_reduction_over_neighbor_expr(
    op: str,
    rhs: ExprType,
    init: ExprType,
    chain: List[LocationTypeValue],
    weights: List[ExprType] = None,
    include_center: bool = False
) -> ReductionOverNeighborExpr:
    """ Create a ReductionOverNeighborExpr

    :param op:              Reduction operation performed for each neighbor
    :param rhs:             Operation to be performed for each neighbor before reducing
    :param init:            Initial value for reduction operation
    :param chain:           Neighbor chain definining the neighbors to reduce from and
                            the location type to reduce to (first element)
    :param weights:         Weights on neighbors (required to be of equal type)
    """
    expr = ReductionOverNeighborExpr()
    iterSpace = UnstructuredIterationSpace()
    expr.op = op
    expr.rhs.CopyFrom(make_expr(rhs))
    expr.init.CopyFrom(make_expr(init))
    iterSpace.chain.extend(chain)
    if weights is not None and len(weights) != 0:
        expr.weights.extend([make_expr(weight) for weight in weights])
    iterSpace.include_center = include_center
    expr.iter_space.CopyFrom(iterSpace)
    return expr


def to_bytes(msg):
    """ Converts protobuf message to JSON format.

    :param msg: The protocol buffers message instance to serialize.
    :returns: A byte string containing the serialized protocol buffer message.
    """
    return msg.SerializeToString()


def from_bytes(byte_string: bytes, message_type):
    """ Parses a serialized representation of a protocol message and returns the parsed message.

    :param text: Text JSON representation of the message.
    :param message_type: The *type* of message to parse.
    :returns: The parsed message.
    :raises ParseError: Failed to parse message
    """
    try:
        msg = message_type.FromString(byte_string)
    except Exception as e:
        raise ParseError(str(e))
    return msg


def to_json(msg):
    """ Converts protobuf message to JSON format.

    :param msg: The protocol buffers message instance to serialize.
    :returns: A string containing the JSON formatted protocol buffer message.
    """
    return json_format.MessageToJson(msg)


def from_json(text: str, message_type):
    """ Parses a JSON representation of a protocol message and returns the parsed message.

    :param text: Text JSON representation of the message.
    :param message_type: The *type* of message to parse.
    :returns: The parsed message.
    :raises ParseError: Failed to parse JSON
    """
    msg = message_type()
    try:
        json_format.Parse(text, msg)
    except json_format.ParseError as e:
        raise ParseError(str(e))
    return msg
