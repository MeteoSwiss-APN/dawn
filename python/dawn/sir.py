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
from .SIR_pb2 import *
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


def makeType(builtin_type_or_name, is_const: bool = False, is_volatile: bool = False):
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


def makeAST(root: StmtType) -> AST:
    """ Create an AST

    :param root:    Root node of the AST (needs to be of type BlockStmt)
    """
    ast = AST()
    if isinstance(root, BlockStmt) or (
        isinstance(root, Stmt) and root.WhichOneof("stmt") == "block_stmt"):
        ast.root.CopyFrom(makeStmt(root))
    else:
        raise SIRError("root statement of an AST needs to be a BlockStmt")
    return ast


def makeField(name: str, is_temporary: bool = False) -> Field:
    """ Create a Field

    :param name:         Name of the field
    :param is_temporary: Is it a temporary field?
    """
    field = Field()
    field.name = name
    field.is_temporary = is_temporary
    return field


def makeInterval(lower_level, upper_level, lower_offset: int = 0,
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


def makeStencilCall(callee: str, arguments: List[Field]) -> StencilCall:
    """ Create a StencilCall

    :param callee:      Name of the called stencil (i.e callee)
    :param arguments:   Fields passed as arguments during the stencil call
    """
    call = StencilCall()
    call.callee = callee
    if isinstance(arguments, Iterable):
        call.arguments.extend(
            [makeField(arg) if isinstance(arg, str) else arg for arg in arguments])
    else:
        call.arguments.extend([makeField(arguments) if isinstance(arguments, str) else arguments])
    return call


def makeVerticalRegion(ast: AST, interval: Interval,
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


def makeExpr(expr: ExprType):
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
        wrapped_expr.var_access_expr.CopyFrom(expr)
    elif isinstance(expr, LiteralAccessExpr):
        wrapped_expr.literal_access_expr.CopyFrom(expr)
    else:
        raise SIRError("cannot create Expr from type {}".format(type(expr)))
    return wrapped_expr


def makeStmt(stmt: StmtType):
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
        wrapped_stmt.var_decl_stmt.CopyFrom(stmt)
    elif isinstance(stmt, StencilCallDeclStmt):
        wrapped_stmt.var_decl_stmt.CopyFrom(stmt)
    elif isinstance(stmt, BoundaryConditionDeclStmt):
        wrapped_stmt.var_decl_stmt.CopyFrom(stmt)
    elif isinstance(stmt, IfStmt):
        wrapped_stmt.if_stmt.CopyFrom(stmt)
    else:
        raise SIRError("cannot create Stmt from type {}".format(type(stmt)))
    return wrapped_stmt


def makeBlockStmt(statements: List[StmtType]) -> BlockStmt:
    """ Create an UnaryOperator

    :param op:      Operation (e.g "+" or "-").
    :param operand:  Expression to apply the operation.
    """
    stmt = BlockStmt()
    if isinstance(statements, Iterable):
        stmt.statements.extend([makeStmt(s) for s in statements])
    else:
        stmt.statements.extend([makeStmt(statements)])
    return stmt


def makeExprStmt(expr: ExprType) -> ExprStmt:
    """ Create an ExprStmt

    :param expr:      Expression.
    """
    stmt = ExprStmt()
    stmt.expr.CopyFrom(makeExpr(expr))
    return stmt


def makeReturnStmt(expr: ExprType) -> ReturnStmt:
    """ Create an ReturnStmt

    :param expr:      Expression to return.
    """
    stmt = ReturnStmt()
    stmt.expr.CopyFrom(makeExpr(expr))
    return stmt


def makeVarDeclStmt(type: Type, name: str, dimension: int = 0, op: str = "=",
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
            stmt.init_list.extend([makeExpr(expr) for expr in init_list])
        else:
            stmt.init_list.extend([makeExpr(init_list)])

    return stmt


def makeStencilCallDeclStmt(stencil_call: StencilCall) -> StencilCallDeclStmt:
    """ Create a StencilCallDeclStmt

    :param stencil_call:   Stencil call.
    """
    stmt = StencilCallDeclStmt()
    stmt.stencil_call.CopyFrom(stencil_call)
    return stmt


def makeVerticalRegionDeclStmt(vertical_region: VerticalRegion) -> VerticalRegionDeclStmt:
    """ Create a VerticalRegionDeclStmt

    :param vertical_region:   Vertical region.
    """
    stmt = VerticalRegionDeclStmt()
    stmt.vertical_region.CopyFrom(vertical_region)
    return stmt


def makeBoundaryConditionDeclStmt(functor: str,
                                  fields: List[Field]) -> BoundaryConditionDeclStmt:
    """ Create a BoundaryConditionDeclStmt

    :param functor:  Identifier of the boundary condition functor.
    :param fields:   List of field arguments to apply the functor to.
    """
    stmt = BoundaryConditionDeclStmt()
    stmt.functor = functor
    if isinstance(fields, Iterable):
        stmt.fields.extend(
            [makeField(field) if isinstance(field, str) else field for field in fields])
    else:
        stmt.fields.extend([makeField(fields) if isinstance(fields, str) else fields])
    return stmt


def makeIfStmt(cond_part: StmtType, then_part: StmtType, else_part: StmtType = None) -> IfStmt:
    """ Create an ReturnStmt

    :param cond_part:   Condition part.
    :param then_part:   Then part.
    :param else_part:   Else part.
    """
    stmt = IfStmt()
    stmt.expr.CopyFrom(makeStmt(cond_part))
    stmt.expr.CopyFrom(makeStmt(then_part))
    if else_part:
        stmt.expr.CopyFrom(makeStmt(else_part))
    return stmt


def makeUnaryOperator(op: str, operand: ExprType) -> UnaryOperator:
    """ Create an UnaryOperator

    :param op:      Operation (e.g "+" or "-").
    :param operand:  Expression to apply the operation.
    """
    expr = UnaryOperator()
    expr.op = op
    expr.operand.CopyFrom(makeExpr(operand))
    return expr


def makeBinaryOperator(left: ExprType, op: str, right: ExprType) -> BinaryOperator:
    """ Create a BinaryOperator

    :param left:    Left-hand side.
    :param op:      Operation (e.g "+" or "-").
    :param right:   Right-hand side.
    """
    expr = BinaryOperator()
    expr.op = op
    expr.left.CopyFrom(makeExpr(left))
    expr.right.CopyFrom(makeExpr(right))
    return expr


def makeAssignmentExpr(left: ExprType, right: ExprType, op: str = "=") -> AssignmentExpr:
    """ Create an AssignmentExpr

    :param left:    Left-hand side.
    :param right:   Right-hand side.
    :param op:      Operation (e.g "=" or "+=").
    """
    expr = AssignmentExpr()
    expr.op = op
    expr.left.CopyFrom(makeExpr(left))
    expr.right.CopyFrom(makeExpr(right))
    return expr


def makeTernaryOperator(cond: ExprType, left: ExprType, right: ExprType) -> TernaryOperator:
    """ Create a TernaryOperator

    :param cond:    Condition.
    :param left:    Left-hand side.
    :param right:   Right-hand side.
    """
    expr = TernaryOperator()
    expr.cond.CopyFrom(makeExpr(cond))
    expr.left.CopyFrom(makeExpr(left))
    expr.right.CopyFrom(makeExpr(right))
    return expr


def makeFunCallExpr(callee: str, arguments: List[ExprType]) -> FunCallExpr:
    """ Create a FunCallExpr

    :param callee:        Identifier of the function (i.e callee).
    :param arguments:     List of arguments.
    """
    expr = FunCallExpr()
    expr.callee = callee
    expr.arguments.extend([makeExpr(arg) for arg in arguments])
    return expr


def makeStencilFunCallExpr(callee: str, arguments: List[ExprType]) -> StencilFunCallExpr:
    """ Create a StencilFunCallExpr

    :param callee:        Identifier of the function (i.e callee).
    :param arguments:     List of arguments.
    """
    expr = StencilFunCallExpr()
    expr.callee = callee
    expr.arguments.extend([makeExpr(arg) for arg in arguments])
    return expr


def makeStencilFunArgExpr(direction: Dimension.Direction, offset: int = 0,
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


def makeFieldAccessExpr(name: str, offset: List[int] = [0, 0, 0],
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


def makeVarAccessExpr(name: str, index: ExprType = None,
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
        expr.index.CopyFrom(makeExpr(index))
    return expr


def makeLiteralAccessExpr(value: str, type: BuiltinType.TypeID) -> LiteralAccessExpr:
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


__all__ = [
    # SIR
    'SIR',
    'Stencil',
    'StencilFunction',
    'GlobalVariableMap',
    'SourceLocation',
    'AST',
    'makeAST',
    'Field',
    'makeField',
    'Interval',
    'makeInterval',
    'BuiltinType',
    'SourceLocation',
    'Type',
    'makeType',
    'Dimension',
    'VerticalRegion',
    'makeVerticalRegion',
    'StencilCall',
    'makeStencilCall',

    # Stmt
    'Stmt',
    'makeStmt',
    'BlockStmt',
    'makeBlockStmt',
    'ExprStmt',
    'makeExprStmt',
    'ReturnStmt',
    'makeReturnStmt',
    'VarDeclStmt',
    'makeVarDeclStmt',
    'VerticalRegionDeclStmt',
    'makeVerticalRegionDeclStmt',
    'StencilCallDeclStmt',
    'makeStencilCallDeclStmt',
    'BoundaryConditionDeclStmt',
    'makeBoundaryConditionDeclStmt',
    'IfStmt',
    'makeIfStmt',

    # Expr
    'Expr',
    'makeExpr',
    'UnaryOperator',
    'makeUnaryOperator',
    'BinaryOperator',
    'makeBinaryOperator',
    'AssignmentExpr',
    'makeAssignmentExpr',
    'TernaryOperator',
    'makeTernaryOperator',
    'FunCallExpr',
    'makeFunCallExpr',
    'StencilFunCallExpr',
    'makeStencilFunCallExpr',
    'StencilFunArgExpr',
    'makeStencilFunArgExpr',
    'VarAccessExpr',
    'makeVarAccessExpr',
    'FieldAccessExpr',
    'makeFieldAccessExpr',
    'LiteralAccessExpr',
    'makeLiteralAccessExpr',

    # Convenience functions
    'to_json',
    'from_json',
]
