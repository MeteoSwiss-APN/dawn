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


import textwrap

from enum import Enum
from collections import Iterable
from typing import List, TypeVar

from google.protobuf import json_format

from .error import ParseError, SIRError
from .SIR.SIR_pb2 import *
from .SIR.statements_pb2 import *
from .. import utils

__all__ = [
    "make_ast",
    "make_field",
    "make_interval",
    "make_type",
    "make_vertical_region",
    "make_stencil_call",
    "make_stencil",
    "make_sir",
    "make_stmt",
    "make_block_stmt",
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
    "make_var_access_expr",
    "make_field_access_expr",
    "make_literal_access_expr",
    "make_reduction_over_neighbor_expr",
    "to_bytes",
    "from_bytes",
    "to_json",
    "from_json",
    "SIRPrinter",
    "pprint",
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
)


class LocationType(Enum):
    Cell = 0
    Edge = 1
    Vertex = 2


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


def make_ast(root: List[StmtType]) -> AST:
    """ Create an AST

    :param root:    Root node of the AST (needs to be of type BlockStmt)
    """
    ast = AST()

    block_stmt = make_block_stmt(root)

    ast.root.CopyFrom(make_stmt(block_stmt))
    return ast


def make_field(
    name: str,
    is_temporary: bool = False,
    dimensions: List[int] = [1, 1, 1],
    location_type: LocationType = LocationType.Cell,
) -> Field:
    """ Create a Field

    :param name:         Name of the field
    :param is_temporary: Is it a temporary field?
    :param dimensions:   mask list to identify dimensions contained by the field
    """
    field = Field()
    field.name = name
    field.is_temporary = is_temporary
    field.field_dimensions.extend(dimensions)
    if location_type == LocationType.Cell:
        field.location_type = Field.Cell
    elif location_type == LocationType.Edge:
        field.location_type = Field.Edge
    elif location_type == LocationType.Vertex:
        field.location_type = Field.Vertex
    return field


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


def make_sir(
    filename: str,
    stencils: List[Stencil],
    functions: List[StencilFunction] = [],
    global_variables: GlobalVariableMap = None,
) -> SIR:
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


def make_vertical_region(
    ast: AST, interval: Interval, loop_order: VerticalRegion.LoopOrder
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
    ast: AST, interval: Interval, loop_order: VerticalRegion.LoopOrder
) -> VerticalRegionDeclStmt:
    """ Create a VerticalRegionDeclStmt

    :param vertical_region:   Vertical region.
    """
    stmt = VerticalRegionDeclStmt()
    stmt.vertical_region.CopyFrom(make_vertical_region(ast, interval, loop_order))
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

        expr.vertical_offset = offset[2]

    elif len(offset) == 2:
        assert isinstance(offset[0], bool)
        assert isinstance(offset[1], int)
        assert argument_map == [-1, -1, -1]
        assert argument_offset == [0, 0, 0]
        assert negate_offset == False

        expr.unstructured_offset.has_offset = offset[0]
        expr.vertical_offset = offset[1]

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
    builtin_type.type_id = type

    expr = LiteralAccessExpr()
    expr.value = value
    expr.type.CopyFrom(builtin_type)
    return expr


def make_reduction_over_neighbor_expr(
    op: str, rhs: ExprType, init: ExprType
) -> ReductionOverNeighborExpr:
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


class SIRPrinter:
    def __init__(self, indent_size=2):
        self._indent = 0
        self.indent_size = indent_size
        self.wrapper = textwrap.TextWrapper(
            initial_indent=" " * self._indent, width=120, subsequent_indent=" " * self._indent
        )

    @classmethod
    def apply(cls, node, *, indent_size=2):
        cls(indent_size=indent_size).visit(node)

    def visit(self, node):
        name = type(node).__name__.split(".")[-1]
        method_name = "visit_" + utils.pythonize_name(name)
        method = getattr(self, method_name, None)
        if not method:
            raise ValueError(f"SIR node not recognized ({node})")

        return method(node)

    def visit_builtin_type(self, builtin_type):
        if builtin_type.type_id == 0:
            raise ValueError("Builtin type not supported")
        elif builtin_type.type_id == 1:
            return "auto"
        elif builtin_type.type_id == 2:
            return "bool"
        elif builtin_type.type_id == 3:
            return "int"
        elif builtin_type.type_id == 4:
            return "float"
        raise ValueError("Builtin type not supported")

    def visit_unary_operator(self, expr):
        return expr.op + " " + self.visit_expr(expr.operand)

    def visit_binary_operator(self, expr):
        return (
            "("
            + self.visit_expr(expr.left)
            + " "
            + expr.op
            + " "
            + self.visit_expr(expr.right)
            + ")"
        )

    def visit_assignment_expr(self, expr):
        return self.visit_expr(expr.left) + " " + expr.op + " " + self.visit_expr(expr.right)

    def visit_ternary_operator(self, expr):
        return (
            "("
            + self.visit_expr(expr.cond)
            + " ? "
            + self.visit_expr(expr.left)
            + " : "
            + self.visit_expr(expr.right)
            + ")"
        )

    def visit_var_access_expr(self, expr):
        return expr.name  # + self.visit_expr(expr.index)

    def visit_field_access_expr(self, expr):
        str_ = expr.name + "["
        if expr.WhichOneof("horizontal_offset") == "cartesian_offset":
            str_ += str(expr.cartesian_offset.i_offset) + ","
            str_ += str(expr.cartesian_offset.j_offset)
        elif expr.WhichOneof("horizontal_offset") == "unstructured_offset":
            str_ += (
                "<has_horizontal_offset>"
                if expr.unstructured_offset.has_offset
                else "<no_horizontal_offset>"
            )
        elif expr.WhichOneof("horizontal_offset") == "zero_offset":
            str_ += "<no_horizontal_offset>"
        else:
            raise ValueError("Unknown offset")
        str_ += "," + str(expr.vertical_offset)
        str_ += "]"
        return str_

    def visit_literal_access_expr(self, expr):
        return expr.value

    # call to external function, like math::sqrt
    def visit_fun_call_expr(self, expr):
        return expr.callee + "(" + ",".join(self.visit_expr(x) for x in expr.arguments) + ")"

    def visit_reduction_over_neighbor_expr(self, expr):
        return (
            "reduce("
            + expr.op
            + ", init="
            + self.visit_expr(expr.init)
            + ", rhs="
            + self.visit_expr(expr.rhs)
            + ")"
        )

    def visit_expr(self, expr):
        if expr.WhichOneof("expr") == "unary_operator":
            return self.visit_unary_operator(expr.unary_operator)
        elif expr.WhichOneof("expr") == "binary_operator":
            return self.visit_binary_operator(expr.binary_operator)
        elif expr.WhichOneof("expr") == "assignment_expr":
            return self.visit_assignment_expr(expr.assignment_expr)
        elif expr.WhichOneof("expr") == "ternary_operator":
            return self.visit_ternary_operator(expr.ternary_operator)
        elif expr.WhichOneof("expr") == "fun_call_expr":
            return self.visit_fun_call_expr(expr.fun_call_expr)
        elif expr.WhichOneof("expr") == "stencil_fun_call_expr":
            raise ValueError("non supported expression")
        elif expr.WhichOneof("expr") == "stencil_fun_arg_expr":
            raise ValueError("non supported expression")
        elif expr.WhichOneof("expr") == "var_access_expr":
            return self.visit_var_access_expr(expr.var_access_expr)
        elif expr.WhichOneof("expr") == "field_access_expr":
            return self.visit_field_access_expr(expr.field_access_expr)
        elif expr.WhichOneof("expr") == "literal_access_expr":
            return self.visit_literal_access_expr(expr.literal_access_expr)
        elif expr.WhichOneof("expr") == "reduction_over_neighbor_expr":
            return self.visit_reduction_over_neighbor_expr(expr.reduction_over_neighbor_expr)
        else:
            raise ValueError("Unknown expression")

    def visit_var_decl_stmt(self, var_decl):
        str_ = ""
        if var_decl.type.WhichOneof("type") == "name":
            str_ += var_decl.type.name
        elif var_decl.type.WhichOneof("type") == "builtin_type":
            str_ += self.visit_builtin_type(var_decl.type.builtin_type)
        else:
            raise ValueError("Unknown type ", var_decl.type.WhichOneof("type"))
        str_ += " " + var_decl.name

        if var_decl.dimension != 0:
            str_ += "[" + str(var_decl.dimension) + "]"

        str_ += var_decl.op

        for expr in var_decl.init_list:
            str_ += self.visit_expr(expr)

        print(self.wrapper.fill(str_))

    def visit_expr_stmt(self, stmt):
        print(self.wrapper.fill(self.visit_expr(stmt.expr)))

    def visit_if_stmt(self, stmt):
        cond = stmt.cond_part
        if cond.WhichOneof("stmt") != "expr_stmt":
            raise ValueError("Not expected stmt")

        print(self.wrapper.fill("if(" + self.visit_expr(cond.expr_stmt.expr) + ")"))
        self.visit_body_stmt(stmt.then_part)
        self.visit_body_stmt(stmt.else_part)

    def visit_block_stmt(self, stmt):
        print(self.wrapper.fill("{"))
        self._indent += self.indent_size
        self.wrapper.initial_indent = " " * self._indent

        for each in stmt.statements:
            self.visit_body_stmt(each)
        self._indent -= self.indent_size
        self.wrapper.initial_indent = " " * self._indent

        print(self.wrapper.fill("}"))

    def visit_body_stmt(self, stmt):
        if stmt.WhichOneof("stmt") == "var_decl_stmt":
            self.visit_var_decl_stmt(stmt.var_decl_stmt)
        elif stmt.WhichOneof("stmt") == "expr_stmt":
            self.visit_expr_stmt(stmt.expr_stmt)
        elif stmt.WhichOneof("stmt") == "if_stmt":
            self.visit_if_stmt(stmt.if_stmt)
        elif stmt.WhichOneof("stmt") == "block_stmt":
            self.visit_block_stmt(stmt.block_stmt)
        else:
            raise ValueError("Stmt not supported :" + stmt.WhichOneof("stmt"))

    def visit_vertical_region(self, vertical_region):
        str_ = "vertical_region("
        interval = vertical_region.interval
        if interval.WhichOneof("LowerLevel") == "special_lower_level":
            if interval.special_lower_level == 0:
                str_ += "kstart"
            else:
                str_ += "kend"
        elif interval.WhichOneof("LowerLevel") == "lower_level":
            str_ += str(interval.lower_level)
        if interval.lower_offset != 0:
            str_ += "+" + str(interval.lower_offset)
        str_ += ","
        if interval.WhichOneof("UpperLevel") == "special_upper_level":
            if interval.special_upper_level == 0:
                str_ += "kstart"
            else:
                str_ += "kend"
        elif interval.WhichOneof("UpperLevel") == "upper_level":
            str_ += str(interval.upper_level)
        if interval.upper_offset != 0:
            if interval.upper_offset > 0:
                str_ += "+" + str(interval.upper_offset)
            else:
                str_ += "-" + str(-interval.upper_offset)
        str_ += ")"
        print(self.wrapper.fill(str_))

        self._indent += self.indent_size
        self.wrapper.initial_indent = " " * self._indent

        for stmt in vertical_region.ast.root.block_stmt.statements:
            self.visit_body_stmt(stmt)

        self._indent -= self.indent_size
        self.wrapper.initial_indent = " " * self._indent

    def visit_stencil(self, stencil):
        print(self.wrapper.fill("stencil " + stencil.name))
        self._indent += self.indent_size
        self.wrapper.initial_indent = " " * self._indent

        self.visit_fields(stencil.fields)

        block = stencil.ast.root.block_stmt
        for stmt in block.statements:
            if stmt.WhichOneof("stmt") == "vertical_region_decl_stmt":
                self.visit_vertical_region(stmt.vertical_region_decl_stmt.vertical_region)

        self._indent -= self.indent_size
        self.wrapper.initial_indent = " " * self._indent

    def visit_global_variables(self, global_variables):
        if not global_variables.IsInitialized():
            return

        print(self.wrapper.fill("globals"))

        self._indent += self.indent_size
        self.wrapper.initial_indent = " " * self._indent
        for name, value in global_variables.map.items():
            str_ = ""
        if value.WhichOneof("Value") == "integer_value":
            str_ += "integer " + name
            if value.is_constexpr:
                str_ += "= " + value.integer_value

        if value.WhichOneof("Value") == "double_value":
            str_ += "double " + name
            if value.is_constexpr:
                str_ += "= " + value.double_value

        if value.WhichOneof("Value") == "boolean_value":
            str_ += "bool " + name
            if value.is_constexpr:
                str_ += "= " + value.boolean_value

        if value.WhichOneof("Value") == "string_value":
            str_ += "string " + name
            if value.is_constexpr:
                str_ += "= " + value.string_value

        print(self.wrapper.fill(str_))
        self._indent -= self.indent_size
        self.wrapper.initial_indent = " " * self._indent

    def visit_fields(self, fields):
        str_ = "field "
        for field in fields:
            str_ += field.name
            dims = ["i", "j", "k"]
            dims_ = []
            for i in range(0, 3):
                if field.field_dimensions[i] != -1:
                    dims_.append(dims[i])
            str_ += str(dims_)
            str_ += ","
        print(self.wrapper.fill(str_))

    def visit_sir(self, sir):
        for stencil in sir.stencils:
            self.visit_stencil(stencil)


pprint = SIRPrinter.apply
