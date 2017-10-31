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

from dawn.sir import *


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


class TestJSON(unittest.TestCase):
    def test_serialization(self):
        ref_expr = makeLiteralAccessExpr("1.235", BuiltinType.Float)

        json = to_json(ref_expr)
        expr = from_json(json, LiteralAccessExpr)
        self.assertEqual(expr, ref_expr)


class TestExpr(unittest.TestCase):
    def test_literal_access_expr(self):
        expr = makeLiteralAccessExpr("1.235", BuiltinType.Float)
        self.assertEqual(expr.value, "1.235")
        self.assertEqual(expr.type.type_id, BuiltinType.Float)


if __name__ == "__main__":
    unittest.main()
