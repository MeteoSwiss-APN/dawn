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

from os import path
from sys import path as sys_path

sys_path.insert(1, path.join(path.dirname(path.realpath(__file__)), ".."))

import unittest

from dawn.error import SIRError
from dawn.sir import *


class ExprTestBase(unittest.TestCase):
    def lit(self):
        """ Create a dummy literal expression """
        return make_literal_access_expr("1.235", BuiltinType.Float)

    def var(self, index=None):
        """ Create a dummy variable expression """
        return make_var_access_expr("var", index)


class StmtTestBase(ExprTestBase):
    def lit_stmt(self):
        """ Create a dummy literal expression statement """
        return make_expr_stmt(self.lit())

    def var_stmt(self, index=None):
        """ Create a dummy variable expression statement """
        return make_expr_stmt(self.var(index))


class ASTTestBase(StmtTestBase):
    def ast(self):
        """ Create a dummy AST """
        return make_ast(make_block_stmt(self.var_stmt()))


class VerticalRegionTestBase(ASTTestBase):
    def vertical_region(self):
        """ Create a dummy vertical region """
        return make_vertical_region(make_interval(Interval.Start, Interval.End),
                                  VerticalRegion.Forward)


class TestJSON(unittest.TestCase):
    def test_serialization(self):
        ref_expr = make_literal_access_expr("1.235", BuiltinType.Float)

        json = to_json(ref_expr)
        expr = from_json(json, LiteralAccessExpr)
        self.assertEqual(expr, ref_expr)


class Testmake_expr(unittest.TestCase):
    def test_make_expr(self):
        expr = make_literal_access_expr("1.235", BuiltinType.Float)

        wrapped_expr = make_expr(expr)
        self.assertEqual(wrapped_expr.literal_access_expr, expr)

        # Should return itself
        self.assertEqual(make_expr(wrapped_expr), wrapped_expr)

        # Invalid type, should throw exception
        with self.assertRaises(SIRError):
            make_expr("foo")


class Testmake_stmt(unittest.TestCase):
    def test_make_stmt(self):
        stmt = make_expr_stmt(make_literal_access_expr("1.235", BuiltinType.Float))

        wrapped_stmt = make_stmt(stmt)
        self.assertEqual(wrapped_stmt.expr_stmt, stmt)

        # Should return itself
        self.assertEqual(make_stmt(wrapped_stmt), wrapped_stmt)

        # Invalid type, should throw exception
        with self.assertRaises(SIRError):
            make_stmt("foo")


class Testmake_type(unittest.TestCase):
    def test_make_custom_type(self):
        t = make_type("foo")
        self.assertEqual(t.name, "foo")

    def test_make_custom_builtin_type(self):
        t = make_type(BuiltinType.Integer)
        builtin_type2 = BuiltinType()
        builtin_type2.type_id = BuiltinType.Integer
        self.assertEqual(t.builtin_type, builtin_type2)


class TestMakeField(unittest.TestCase):
    def test_make_field(self):
        field = make_field("foo")
        self.assertEqual(field.name, "foo")

    def test_make_field_temporary(self):
        field = make_field("foo", True)
        self.assertEqual(field.name, "foo")
        self.assertEqual(field.is_temporary, True)


class TestMakeInterval(unittest.TestCase):
    def test_make_interval_start_end(self):
        interval = make_interval(Interval.Start, Interval.End)
        self.assertEqual(interval.special_lower_level, Interval.Start)
        self.assertEqual(interval.special_upper_level, Interval.End)
        self.assertEqual(interval.lower_offset, 0)
        self.assertEqual(interval.upper_offset, 0)

    def test_make_interval_start_plus_1_end_minuse_1(self):
        interval = make_interval(Interval.Start, Interval.End, 1, -1)
        self.assertEqual(interval.special_lower_level, Interval.Start)
        self.assertEqual(interval.special_upper_level, Interval.End)
        self.assertEqual(interval.lower_offset, 1)
        self.assertEqual(interval.upper_offset, -1)

    def test_make_interval_11_end(self):
        interval = make_interval(11, Interval.End)
        self.assertEqual(interval.lower_level, 11)
        self.assertEqual(interval.special_upper_level, Interval.End)
        self.assertEqual(interval.lower_offset, 0)
        self.assertEqual(interval.upper_offset, 0)

    def test_make_interval_11_22(self):
        interval = make_interval(11, 22, 5, -5)
        self.assertEqual(interval.lower_level, 11)
        self.assertEqual(interval.upper_level, 22)
        self.assertEqual(interval.lower_offset, 5)
        self.assertEqual(interval.upper_offset, -5)


class TestMakeStencilCall(unittest.TestCase):
    def test_make_stencil_call(self):
        call = make_stencil_call("foo", ["a", "b"])
        self.assertEqual(call.callee, "foo")
        self.assertEqual(call.arguments[0], "a")
        self.assertEqual(call.arguments[1], "b")

    def test_make_stencil_call_1_arg(self):
        call = make_stencil_call("foo", "a")
        self.assertEqual(call.callee, "foo")
        self.assertEqual(call.arguments[0], "a")

    def test_make_stencil_call_str_args(self):
        call = make_stencil_call("foo", ["a", "b", "c"])
        self.assertEqual(call.callee, "foo")
        self.assertEqual(call.arguments[0], "a")
        self.assertEqual(call.arguments[1], "b")
        self.assertEqual(call.arguments[2], "c")


class TestMakeVerticalRegion(ASTTestBase):
    def test_make_vertical_region(self):
        vr = make_vertical_region(make_interval(Interval.Start, Interval.End),
                                VerticalRegion.Backward)
        self.assertEqual(vr.interval, make_interval(Interval.Start, Interval.End))
        self.assertEqual(vr.loop_order, VerticalRegion.Backward)
        stmt = make_vertical_region_decl_stmt(self.ast(), make_interval(Interval.Start, Interval.End), VerticalRegion.Backward)
        self.assertEqual(stmt.ast, self.ast())
        self.assertEqual(stmt.vertical_region, vr)


class TestStmt(VerticalRegionTestBase):
    def block_stmt(self):
        stmt1 = make_block_stmt([self.var_stmt(), self.lit_stmt()])
        self.assertEqual(stmt1.statements[0], make_stmt(self.var_stmt()))
        self.assertEqual(stmt1.statements[1], make_stmt(self.lit_stmt()))

        stmt2 = make_block_stmt(self.var_stmt())
        self.assertEqual(stmt2.statements[0], make_stmt(self.var_stmt()))

    def test_expr_stmt(self):
        stmt = make_expr_stmt(self.var())
        self.assertEqual(stmt.expr, make_expr(self.var()))

    def test_return_stmt(self):
        stmt = make_return_stmt(self.var())
        self.assertEqual(stmt.expr, make_expr(self.var()))

    def test_var_decl_stmt(self):
        stmt = make_var_decl_stmt(make_type(BuiltinType.Float), "var")
        self.assertEqual(stmt.type, make_type(BuiltinType.Float))
        self.assertEqual(stmt.name, "var")

    def test_boundary_condition_decl_stmt(self):
        stmt = make_boundary_condition_decl_stmt("foo", ["a", "b"])
        self.assertEqual(stmt.functor, "foo")
        self.assertEqual(stmt.fields[0], "a")
        self.assertEqual(stmt.fields[1], "b")


class TestExpr(ExprTestBase):
    def test_unary_operator(self):
        expr = make_unary_operator("+", self.lit())
        self.assertEqual(expr.op, "+")
        self.assertEqual(expr.operand, make_expr(self.lit()))

    def test_binary_operator(self):
        expr = make_binary_operator(self.var(), "+", self.lit())
        self.assertEqual(expr.op, "+")
        self.assertEqual(expr.left, make_expr(self.var()))
        self.assertEqual(expr.right, make_expr(self.lit()))

    def test_assignment_expr(self):
        expr = make_assignment_expr(self.var(), self.lit(), "+=")
        self.assertEqual(expr.left, make_expr(self.var()))
        self.assertEqual(expr.right, make_expr(self.lit()))
        self.assertEqual(expr.op, "+=")

    def test_ternary_operator(self):
        expr = make_ternary_operator(self.lit(), self.var(), self.var(self.lit()))
        self.assertEqual(expr.cond, make_expr(self.lit()))
        self.assertEqual(expr.left, make_expr(self.var()))
        self.assertEqual(expr.right, make_expr(self.var(self.lit())))

    def test_fun_call_expr(self):
        expr = make_fun_call_expr("fun", [self.var(), self.lit()])
        self.assertEqual(expr.callee, "fun")
        self.assertEqual(expr.arguments[0], make_expr(self.var()))
        self.assertEqual(expr.arguments[1], make_expr(self.lit()))

    def test_stencil_fun_call_expr(self):
        expr = make_stencil_fun_call_expr("fun", [self.var(), self.lit()])
        self.assertEqual(expr.callee, "fun")
        self.assertEqual(expr.arguments[0], make_expr(self.var()))
        self.assertEqual(expr.arguments[1], make_expr(self.lit()))

    def test_stencil_fun_arg_expr(self):
        arg1 = make_stencil_fun_arg_expr(Dimension.I)
        self.assertEqual(arg1.dimension.direction, Dimension.I)

    def test_field_access_expr(self):
        field1 = make_field_access_expr("a")
        self.assertEqual(field1.name, "a")
        self.assertEqual(field1.offset, [0, 0, 0])
        self.assertEqual(field1.argument_map, [-1, -1, -1])
        self.assertEqual(field1.argument_offset, [0, 0, 0])
        self.assertEqual(field1.negate_offset, False)

        field2 = make_field_access_expr("a", [0, 0, -1])
        self.assertEqual(field2.name, "a")
        self.assertEqual(field2.offset, [0, 0, -1])
        self.assertEqual(field2.argument_map, [-1, -1, -1])
        self.assertEqual(field2.argument_offset, [0, 0, 0])
        self.assertEqual(field2.negate_offset, False)

        field2 = make_field_access_expr("a", [0, 2, -1], negate_offset=True,
                                     argument_offset=[1, -2, 3], argument_map=[0, 1, 2])
        self.assertEqual(field2.name, "a")
        self.assertEqual(field2.offset, [0, 2, -1])
        self.assertEqual(field2.argument_map, [0, 1, 2])
        self.assertEqual(field2.argument_offset, [1, -2, 3])
        self.assertEqual(field2.negate_offset, True)

    def test_var_access_expr(self):
        expr = make_var_access_expr("var")
        self.assertEqual(expr.name, "var")
        self.assertEqual(expr.is_external, False)

        array_expr = make_var_access_expr("var", self.lit())
        self.assertEqual(array_expr.name, "var")
        self.assertEqual(array_expr.index, make_expr(self.lit()))
        self.assertEqual(array_expr.is_external, False)

    def test_literal_access_expr(self):
        expr = make_literal_access_expr("1.235", BuiltinType.Float)
        self.assertEqual(expr.value, "1.235")
        self.assertEqual(expr.type.type_id, BuiltinType.Float)


class TestAST(StmtTestBase):
    def test_make_ast(self):
        ast = make_ast([self.var_stmt()])
        self.assertEqual(ast.root, make_stmt(make_block_stmt(self.var_stmt())))

class TestSIR(unittest.TestCase):
    def test_filename(self):
        sir = SIR()
        sir.filename = "foo"
        self.assertEqual(sir.filename, "foo")


class TestStencil(unittest.TestCase):
    def test_name(self):
        stencil = Stencil()
        stencil.name = "foo"
        self.assertEqual(stencil.name, "foo")


class TestStencilFunction(unittest.TestCase):
    def test_filename(self):
        stencil_function = StencilFunction()
        stencil_function.name = "foo"
        self.assertEqual(stencil_function.name, "foo")


if __name__ == "__main__":
    unittest.main()
