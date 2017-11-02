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

import unittest

from dawn.error import Error
from dawn.sir import *


class TestJSON(unittest.TestCase):
    def test_serialization(self):
        ref_expr = makeLiteralAccessExpr("1.235", BuiltinType.Float)

        json = to_json(ref_expr)
        expr = from_json(json, LiteralAccessExpr)
        self.assertEqual(expr, ref_expr)


class TestMakeExpr(unittest.TestCase):
    def test_make_expr(self):
        expr = makeLiteralAccessExpr("1.235", BuiltinType.Float)

        wrapped_expr = makeExpr(expr)
        self.assertEqual(wrapped_expr.literal_access_expr, expr)

        # Should return itself
        self.assertEqual(makeExpr(wrapped_expr), wrapped_expr)

        # Invalid type, should throw exception
        with self.assertRaises(Error):
            makeExpr("foo")


class TestMakeStmt(unittest.TestCase):
    def test_make_stmt(self):
        stmt = makeExprStmt(makeLiteralAccessExpr("1.235", BuiltinType.Float))

        wrapped_stmt = makeStmt(stmt)
        self.assertEqual(wrapped_stmt.expr_stmt, stmt)

        # Should return itself
        self.assertEqual(makeStmt(wrapped_stmt), wrapped_stmt)

        # Invalid type, should throw exception
        with self.assertRaises(Error):
            makeStmt("foo")


class TestMakeType(unittest.TestCase):
    def test_make_custom_type(self):
        t = makeType("foo")
        self.assertEqual(t.name, "foo")

    def test_make_custom_builtin_type(self):
        t = makeType(BuiltinType.Integer)
        builtin_type2 = BuiltinType()
        builtin_type2.type_id = BuiltinType.Integer
        self.assertEqual(t.builtin_type, builtin_type2)


class ExprTestBase(unittest.TestCase):
    def lit(self):
        """ Create a dummy literal expression """
        return makeLiteralAccessExpr("1.235", BuiltinType.Float)

    def var(self, index=None):
        """ Create a dummy variable expression """
        return makeVarAccessExpr("var", index)


class TestStmt(ExprTestBase):
    def lit_stmt(self):
        """ Create a dummy literal expression statement """
        return makeExprStmt(self.lit())

    def var_stmt(self, index=None):
        """ Create a dummy variable expression statement """
        return makeExprStmt(self.var(index))

    def block_stmt(self):
        stmt1 = makeBlockStmt([self.var_stmt(), self.lit_stmt()])
        self.assertEqual(stmt1.statements[0], makeStmt(self.var_stmt()))
        self.assertEqual(stmt1.statements[1], makeStmt(self.lit_stmt()))

        stmt2 = makeBlockStmt(self.var_stmt())
        self.assertEqual(stmt2.statements[0], makeStmt(self.var_stmt()))

    def test_expr_stmt(self):
        stmt = makeExprStmt(self.var())
        self.assertEqual(stmt.expr, makeExpr(self.var()))

    def test_return_stmt(self):
        stmt = makeReturnStmt(self.var())
        self.assertEqual(stmt.expr, makeExpr(self.var()))

    def test_var_decl_stmt(self):
        stmt = makeVarDeclStmt(makeType(BuiltinType.Float), "var")
        self.assertEqual(stmt.type, makeType(BuiltinType.Float))
        self.assertEqual(stmt.name, "var")


class TestExpr(ExprTestBase):
    def test_unary_operator(self):
        expr = makeUnaryOperator("+", self.lit())
        self.assertEqual(expr.op, "+")
        self.assertEqual(expr.operand, makeExpr(self.lit()))

    def test_binary_operator(self):
        expr = makeBinaryOperator(self.var(), "+", self.lit())
        self.assertEqual(expr.op, "+")
        self.assertEqual(expr.left, makeExpr(self.var()))
        self.assertEqual(expr.right, makeExpr(self.lit()))

    def test_assignment_expr(self):
        expr = makeAssignmentExpr(self.var(), self.lit())
        self.assertEqual(expr.left, makeExpr(self.var()))
        self.assertEqual(expr.right, makeExpr(self.lit()))

    def test_ternary_operator(self):
        expr = makeTernaryOperator(self.lit(), self.var(), self.var(self.lit()))
        self.assertEqual(expr.cond, makeExpr(self.lit()))
        self.assertEqual(expr.left, makeExpr(self.var()))
        self.assertEqual(expr.right, makeExpr(self.var(self.lit())))

    def test_fun_call_expr(self):
        expr = makeFunCallExpr("fun", [self.var(), self.lit()])
        self.assertEqual(expr.callee, "fun")
        self.assertEqual(expr.arguments[0], makeExpr(self.var()))
        self.assertEqual(expr.arguments[1], makeExpr(self.lit()))

    def test_stencil_fun_call_expr(self):
        expr = makeStencilFunCallExpr("fun", [self.var(), self.lit()])
        self.assertEqual(expr.callee, "fun")
        self.assertEqual(expr.arguments[0], makeExpr(self.var()))
        self.assertEqual(expr.arguments[1], makeExpr(self.lit()))

    def test_stencil_fun_arg_expr(self):
        arg1 = makeStencilFunArgExpr(Dimension.I)
        self.assertEqual(arg1.dimension.dimension, Dimension.I)

    def test_field_access_expr(self):
        field1 = makeFieldAccessExpr("a")
        self.assertEqual(field1.name, "a")
        self.assertEqual(field1.offset, [0, 0, 0])
        self.assertEqual(field1.argument_map, [-1, -1, -1])
        self.assertEqual(field1.argument_offset, [0, 0, 0])
        self.assertEqual(field1.negate_offset, False)

        field2 = makeFieldAccessExpr("a", [0, 0, -1])
        self.assertEqual(field2.name, "a")
        self.assertEqual(field2.offset, [0, 0, -1])
        self.assertEqual(field2.argument_map, [-1, -1, -1])
        self.assertEqual(field2.argument_offset, [0, 0, 0])
        self.assertEqual(field2.negate_offset, False)

        field2 = makeFieldAccessExpr("a", [0, 2, -1], negate_offset=True,
                                     argument_offset=[1, -2, 3], argument_map=[0, 1, 2])
        self.assertEqual(field2.name, "a")
        self.assertEqual(field2.offset, [0, 2, -1])
        self.assertEqual(field2.argument_map, [0, 1, 2])
        self.assertEqual(field2.argument_offset, [1, -2, 3])
        self.assertEqual(field2.negate_offset, True)

    def test_var_access_expr(self):
        expr = makeVarAccessExpr("var")
        self.assertEqual(expr.name, "var")
        self.assertEqual(expr.is_external, False)

        array_expr = makeVarAccessExpr("var", self.lit())
        self.assertEqual(array_expr.name, "var")
        self.assertEqual(array_expr.index, makeExpr(self.lit()))
        self.assertEqual(array_expr.is_external, False)

    def test_literal_access_expr(self):
        expr = makeLiteralAccessExpr("1.235", BuiltinType.Float)
        self.assertEqual(expr.value, "1.235")
        self.assertEqual(expr.type.type_id, BuiltinType.Float)


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
