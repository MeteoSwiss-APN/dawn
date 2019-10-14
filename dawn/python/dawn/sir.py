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

from collections import Iterable
from sys import path as sys_path
from typing import List, TypeVar

from dawn.config import __dawn_install_protobuf_module__

from dawn.error import ParseError, SIRError

sys_path.insert(1, __dawn_install_protobuf_module__)

#
# Export all SIR classes
#
from SIR.SIR_pb2 import *
from SIR.statements_pb2 import *
from google.protobuf import json_format

ExprType = TypeVar('Expr',
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

StmtType = TypeVar('Stmt',
                   Stmt,
                   BlockStmt,
                   ExprStmt,
                   ReturnStmt,
                   VarDeclStmt,
                   VerticalRegionDeclStmt,
                   StencilCallDeclStmt,
                   BoundaryConditionDeclStmt,
                   IfStmt
                   )


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
            "expected 'builtin_type_or_name' to be either of type 'dawn.sir.BuiltinType.TypeID'" +
            "or 'str' (got {})".format(type(builtin_type_or_name)))
    return t


def make_ast(root: List[StmtType]) -> AST:
    """ Create an AST

    :param root:    Root node of the AST (needs to be of type BlockStmt)
    """
    ast = AST()

    block_stmt = make_block_stmt(root)

    ast.root.CopyFrom(make_stmt(block_stmt))
    return ast


def make_field(name: str, is_temporary: bool = False, dimensions: List[int] = [1, 1, 1]) -> Field:
    """ Create a Field

    :param name:         Name of the field
    :param is_temporary: Is it a temporary field?
    :param dimensions:   mask list to identify dimensions contained by the field
    """
    field = Field()
    field.name = name
    field.is_temporary = is_temporary
    field.field_dimensions.extend(dimensions)
    return field


def make_interval(lower_level, upper_level, lower_offset: int = 0,
                  upper_offset: int = 0) -> Interval:
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


def make_sir(filename: str, stencils: List[Stencil], functions: List[StencilFunction] = [],
             global_variables: GlobalVariableMap = None) -> SIR:
    """ Create a SIR

    :param filename:          Source filename
    :param stencils:          list of stencils that compose the SIR
    :param functions:         list of functions used in the SIR
    :param global_variables:  global variable map used in the SIR
    """

    sir = SIR()
    sir.filename = filename
    sir.stencils.extend(stencils)
    sir.stencil_functions.extend(functions)
    if global_variables:
        sir.global_variables.CopyFrom(global_variables)

    return sir


def make_stencil_call(callee: str, arguments: List[str]) -> StencilCall:
    """ Create a StencilCall

    :param callee:      Name of the called stencil (i.e callee)
    :param arguments:   Fields passed as arguments during the stencil call
    """
    call = StencilCall()
    call.callee = callee
    call.arguments.extend(arguments)
    return call


def make_vertical_region(ast: AST, interval: Interval,
                         loop_order: VerticalRegion.LoopOrder) -> VerticalRegion:
    """ Create a VerticalRegion

    :param ast:         Syntax tree of the body of the vertical region
    :param interval:    Vertical interval
    :param loop_order:  Vertical loop order of execution
    """
    vr = VerticalRegion()
    vr.ast.CopyFrom(ast)
    vr.interval.CopyFrom(interval)
    vr.loop_order = loop_order
    return vr


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
        wrapped_stmt.var_decl_stmt.CopyFrom(stmt)
    elif isinstance(stmt, BoundaryConditionDeclStmt):
        wrapped_stmt.var_decl_stmt.CopyFrom(stmt)
    elif isinstance(stmt, IfStmt):
        wrapped_stmt.if_stmt.CopyFrom(stmt)
    else:
        raise SIRError("cannot create Stmt from type {}".format(type(stmt)))
    return wrapped_stmt


def make_block_stmt(statements: List[StmtType]) -> BlockStmt:
    """ Create an UnaryOperator

    :param statements: List of statements that compose the block
    """
    stmt = BlockStmt()
    if isinstance(statements, Iterable):
        stmt.statements.extend([make_stmt(s) for s in statements])
    else:
        stmt.statements.extend([make_stmt(statements)])
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


def make_var_decl_stmt(type: Type, name: str, dimension: int = 0, op: str = "=",
                       init_list=None) -> VarDeclStmt:
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


def make_vertical_region_decl_stmt(ast: AST, interval: Interval,
                                   loop_order: VerticalRegion.LoopOrder) -> VerticalRegionDeclStmt:
    """ Create a VerticalRegionDeclStmt

    :param vertical_region:   Vertical region.
    """
    stmt = VerticalRegionDeclStmt()
    stmt.vertical_region.CopyFrom(make_vertical_region(ast, interval, loop_order))
    return stmt

 
def make_boundary_condition_decl_stmt(functor: str,
                                      fields: List[str]) -> BoundaryConditionDeclStmt:
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


def make_stencil_fun_call_expr(callee: str, arguments: List[ExprType]) -> StencilFunCallExpr:
    """ Create a StencilFunCallExpr

    :param callee:        Identifier of the function (i.e callee).
    :param arguments:     List of arguments.
    """
    expr = StencilFunCallExpr()
    expr.callee = callee
    expr.arguments.extend([make_expr(arg) for arg in arguments])
    return expr


def make_stencil_fun_arg_expr(direction: Dimension.Direction, offset: int = 0,
                              argument_index: int = -1) -> StencilFunArgExpr:
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


def make_field_access_expr(name: str, offset: List[int] = [0, 0, 0],
                           argument_map: List[int] = [-1, -1, -1],
                           argument_offset: List[int] = [0, 0, 0],
                           negate_offset: bool = False) -> FieldAccessExpr:
    """ Create a FieldAccessExpr.

    :param name:            Name of the field.
    :param offset:          Static offset.
    :param argument_map:    Mapping of the directional and offset arguments of the stencil function.
    :param argument_offset: Offset to the directional and offset arguments.
    :param negate_offset:   Negate the offset in the end.
    """
    expr = FieldAccessExpr()
    expr.name = name
    expr.offset.extend(offset)
    expr.argument_map.extend(argument_map)
    expr.argument_offset.extend(argument_offset)
    expr.negate_offset = negate_offset
    return expr


def make_var_access_expr(name: str, index: ExprType = None,
                         is_external: bool = False) -> VarAccessExpr:
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
    builtin_type.type_id = type

    expr = LiteralAccessExpr()
    expr.value = value
    expr.type.CopyFrom(builtin_type)
    return expr

def make_reduction_over_neighbor_expr(op: str, rhs: ExprType, init: ExprType) -> ReductionOverNeighborExpr:
    """ Create a ReductionOverNeighborExpr

    :param op:          Reduction operation performed for each neighbor
    :param rhs:         Operation to be performed for each neighbor before reducing
    :param init:        Initial value for reduction operation
    """
    expr = ReductionOverNeighborExpr()
    expr.op = op
    expr.rhs.CopyFrom(make_expr(rhs))
    expr.init.CopyFrom(make_expr(init))
    return expr




__all__ = [
    # SIR
    'SIR',
    'Stencil',
    'StencilFunction',
    'GlobalVariableMap',
    'SourceLocation',
    'AST',
    'make_ast',
    'Field',
    'make_field',
    'Interval',
    'make_interval',
    'BuiltinType',
    'SourceLocation',
    'Type',
    'make_type',
    'Dimension',
    'VerticalRegion',
    'make_vertical_region',
    'StencilCall',
    'make_stencil_call',
    'make_stencil',
    'make_sir',

    # Stmt
    'Stmt',
    'make_stmt',
    'BlockStmt',
    'make_block_stmt',
    'ExprStmt',
    'make_expr_stmt',
    'ReturnStmt',
    'make_return_stmt',
    'VarDeclStmt',
    'make_var_decl_stmt',
    'VerticalRegionDeclStmt',
    'make_vertical_region_decl_stmt',
    'StencilCallDeclStmt',
    'make_stencil_call_decl_stmt',
    'BoundaryConditionDeclStmt',
    'make_boundary_condition_decl_stmt',
    'IfStmt',
    'make_if_stmt',

    # Expr
    'Expr',
    'make_expr',
    'UnaryOperator',
    'make_unary_operator',
    'BinaryOperator',
    'make_binary_operator',
    'AssignmentExpr',
    'make_assignment_expr',
    'make_assignment_stmt',
    'TernaryOperator',
    'make_ternary_operator',
    'FunCallExpr',
    'make_fun_call_expr',
    'StencilFunCallExpr',
    'make_stencil_fun_call_expr',
    'StencilFunArgExpr',
    'make_stencil_fun_arg_expr',
    'VarAccessExpr',
    'make_var_access_expr',
    'FieldAccessExpr',
    'make_field_access_expr',
    'LiteralAccessExpr',
    'make_literal_access_expr',
    'ReductionOverNeighborExpr',
    'make_reduction_over_neighbor_expr',

    # Convenience functions
    'to_json',
    'from_json',
]
