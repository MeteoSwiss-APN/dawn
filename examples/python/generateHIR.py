import textwrap
import sys
import argparse
import ctypes
import os.path
from ctypes import *

from config import __dawn_install_module__,__dawn_install_dawnclib__ 
sys.path.insert(1, __dawn_install_module__)

from dawn.error import SIRError
from dawn.sir import *
from dawn.config import *

import SIR

dawn = CDLL(__dawn_install_dawnclib__)

class deserializer:
  def __init__(self):
    self.indent_=0
    self.T_ = textwrap.TextWrapper(initial_indent=' '*self.indent_, width=120,subsequent_indent=' '*self.indent_)
 
  def visitBuiltinType(self, builtin_type):
    if builtin_type.type_id == 0:
      raise ValueError('Builtin type not supported')
    elif builtin_type.type_id == 1:
      return "auto"
    elif builtin_type.type_id == 2:
      return "bool"
    elif builtin_type.type_id == 3:
      return "int"
    elif builtin_type.type_id == 4:
      return "float"
    raise ValueError('Builtin type not supported')

  def visitUnaryOperator(self, expr):
    return expr.op+" " +self.visitExpr(expr.operand)
  def visitBinaryOperator(self, expr):
    return self.visitExpr(expr.left) + " " + expr.op + " " + self.visitExpr(expr.right) 
  def visitAssignmentExpr(self, expr):
    return self.visitExpr(expr.left) + " " + expr.op + " " + self.visitExpr(expr.right)
  def visitTernaryOperator(self, expr):
    return "(" + self.visitExpr(expr.cond) + " ? " + self.visitExpr(expr.left) + " : " + self.visitExpr(expr.right) + ")"
  def visitVarAccessExpr(self, expr):
    return expr.name  #+ self.visitExpr(expr.index)
  def visitFieldAccessExpr(self, expr):
    str_ = expr.name +"["
    str_ += ",".join(str(x) for x in expr.offset)
    str_ += "]"
    return str_ 
  def visitLiteralAccessExpr(self, expr):
    return expr.value
  # call to external function, like math::sqrt
  def visitFunCallExpr(self, expr):
    return expr.callee + "("+",".join(self.visitExpr(x) for x in expr.arguments)+")"
  def visitExpr(self,expr):
    if expr.WhichOneof("expr") == "unary_operator":
      return self.visitUnaryOperator(expr.unary_operator)
    elif expr.WhichOneof("expr") == "binary_operator":
      return self.visitBinaryOperator(expr.binary_operator)
    elif expr.WhichOneof("expr") == "assignment_expr":
      return self.visitAssignmentExpr(expr.assignment_expr)
    elif expr.WhichOneof("expr") == "ternary_operator":
      return self.visitTernaryOperator(expr.ternary_operator)
    elif expr.WhichOneof("expr") == "fun_call_expr":
      return self.visitFunCallExpr(expr.fun_call_expr)
    elif expr.WhichOneof("expr") == "stencil_fun_call_expr":
      raise ValueError("non supported expression")
    elif expr.WhichOneof("expr") == "stencil_fun_arg_expr":
      raise ValueError("non supported expression")
    elif expr.WhichOneof("expr") == "var_access_expr":
      return self.visitVarAccessExpr(expr.var_access_expr)
    elif expr.WhichOneof("expr") == "field_access_expr":
      return self.visitFieldAccessExpr(expr.field_access_expr)
    elif expr.WhichOneof("expr") == "literal_access_expr":
      return self.visitLiteralAccessExpr(expr.literal_access_expr)
    else:
      raise ValueError("Unknown expression")

  def visitVarDeclStmt(self, var_decl):
    str_=""
    if var_decl.type.WhichOneof("type")=="name":
      str_+=var_decl.type.name
    elif var_decl.type.WhichOneof("type")=="builtin_type":
      str_ += self.visitBuiltinType(var_decl.type.builtin_type)
    else:
      raise ValueError("Unknown type ", var_decl.type.WhichOneof("type"))
    str_ += " " + var_decl.name

    if var_decl.dimension != 0:
      str_ += "["+str(var_decl.dimension)+"]"

    str_ += var_decl.op
 
    for expr in var_decl.init_list:
      str_ += self.visitExpr(expr)

    print(self.T_.fill(str_))
  def visitExprStmt(self,stmt):
    print(self.T_.fill(self.visitExpr(stmt.expr)))
  def visitIfStmt(self, stmt):
    cond = stmt.cond_part
    if cond.WhichOneof("stmt") != "expr_stmt":
      raise ValueError("Not expected stmt")

    print(self.T_.fill("if("+self.visitExpr(cond.expr_stmt.expr)+")"))
    self.visitBodyStmt(stmt.then_part)
    self.visitBodyStmt(stmt.else_part)

  def visitBlockStmt(self, stmt):
    print(self.T_.fill("{"))
    self.indent_+=2
    self.T_.initial_indent=' '*self.indent_

    for each in stmt.statements:
      self.visitBodyStmt(each)
    self.indent_-=2
    self.T_.initial_indent=' '*self.indent_

    print(self.T_.fill("}"))

  def visitBodyStmt(self, stmt):
    if stmt.WhichOneof("stmt") == "var_decl_stmt":
      self.visitVarDeclStmt(stmt.var_decl_stmt)
    elif stmt.WhichOneof("stmt") == "expr_stmt":
      self.visitExprStmt(stmt.expr_stmt)
    elif stmt.WhichOneof("stmt") == "if_stmt":
      self.visitIfStmt(stmt.if_stmt)
    elif stmt.WhichOneof("stmt") == "block_stmt":
      self.visitBlockStmt(stmt.block_stmt)
    else:
      raise ValueError("Stmt not supported :" + stmt.WhichOneof("stmt"))
  def visitVerticalRegion(self, vertical_region):
    str_="vertical_region("
    interval = vertical_region.interval
    if (interval.WhichOneof("LowerLevel") == 'special_lower_level'):
      if interval.special_lower_level == 0:
        str_ += "kstart"
      else:
        str_ += "kend"
    elif  (interval.WhichOneof("LowerLevel") == 'lower_level'):
      str_ += str(interval.lower_level)
    str_ += ","
    if (interval.WhichOneof("UpperLevel") == 'special_upper_level'):
      if interval.special_upper_level == 0:
        str_ += "kstart"
      else:
        str_ += "kend"
    elif  (interval.WhichOneof("UpperLevel") == 'upper_level'):
      str_ += str(interval.upper_level)
    str_ += ")"
    print(self.T_.fill(str_))

    self.indent_+=2
    self.T_.initial_indent=' '*self.indent_
 
    for stmt in vertical_region.ast.root.block_stmt.statements:
      self.visitBodyStmt(stmt)

    self.indent_-=2
    self.T_.initial_indent=' '*self.indent_
 
  def visitStencil(self,stencil):
    print(self.T_.fill("stencil "+stencil.name))
    self.indent_+=2
    self.T_.initial_indent=' '*self.indent_
 
    self.visitFields(stencil.fields)
 
    block = stencil.ast.root.block_stmt
    for stmt in block.statements:
      if stmt.WhichOneof("stmt") == "vertical_region_decl_stmt":
        self.visitVerticalRegion(stmt.vertical_region_decl_stmt.vertical_region)
   
    self.indent_-=2
    self.T_.initial_indent=' '*self.indent_
  
  def visitGlobalVariables(self, global_variables):
    if not global_variables.IsInitialized():
      return
  
    print(self.T_.fill("globals"))

    self.indent_+=2
    self.T_.initial_indent=' '*self.indent_
    for name, value in hir.global_variables.map.iteritems():
      str_=""
    if value.WhichOneof("Value") == "integer_value":
      str_ += "integer "+name
      if(value.is_constexpr):
        str_ += "= "+ value.integer_value
  
    if value.WhichOneof("Value") == "double_value":
      str_ += "double "+name
      if(value.is_constexpr):
        str_ += "= "+ value.double_value
  
    if value.WhichOneof("Value") == "boolean_value":
      str_ += "bool "+name
      if(value.is_constexpr):
        str_ += "= "+ value.boolean_value
  
    if value.WhichOneof("Value") == "string_value":
      str_ += "string "+name
      if(value.is_constexpr):
        str_ += "= "+ value.string_value
  
    print(self.T_.fill(str_))
    self.indent_-=2
    self.T_.initial_indent=' '*self.indent_

      
  def visitFields(self,fields): 
    str_="field "
    for field in fields:
      str_ += field.name
      dims = ["i","j","k"]
      dims_=[]
      for i in range(0,3):
        if field.field_dimensions[i] != -1:
          dims_.append(dims[i])
      str_ += str(dims_)
      str_ += ","
    print(self.T_.fill(str_))


def createAssignmentExpr(aBodyBlockStmt):
  exprStmt = SIR_pb2.ExprStmt()

  exprStmt.loc.CopyFrom(loc)

  expr = SIR_pb2.Expr()

  assExpr = SIR_pb2.AssignmentExpr()

  assExpr.op = "="

  createLHS(assExpr)
  createRHS(assExpr)
  assExpr.loc.CopyFrom(loc)
  expr.assignment_expr.CopyFrom(assExpr)
  exprStmt.expr.CopyFrom(expr)
  aBodyBlockStmt.expr_stmt.CopyFrom(exprStmt)

def createRHS(assignExpr):
  rightExpr = SIR_pb2.Expr()

  rightField = SIR_pb2.FieldAccessExpr()

  rightField.name = "out"
  rightField.offset.append(0)
  rightField.offset.append(0)
  rightField.offset.append(0)
  rightField.loc.CopyFrom(loc)

  rightExpr.field_access_expr.CopyFrom(rightField)

  assignExpr.right.CopyFrom(rightExpr)

def createLHS(assignExpr):

  leftExpr = SIR_pb2.Expr()

  leftField = SIR_pb2.FieldAccessExpr()

  leftField.name = "in"
  leftField.offset.append(0)
  leftField.offset.append(0)
  leftField.offset.append(0)
  leftField.loc.CopyFrom(loc)

  leftExpr.field_access_expr.CopyFrom(leftField)

  assignExpr.left.CopyFrom(leftExpr)

def createBodyAST(verticalRegion):

  bodyAST = SIR_pb2.AST()

  bodyRootStmt = SIR_pb2.Stmt()
  bodyBlockStmt = SIR_pb2.BlockStmt()
  aBodyBlockStmt = bodyBlockStmt.statements.add()

  createAssignmentExpr(aBodyBlockStmt)

  bodyRootStmt.block_stmt.CopyFrom(bodyBlockStmt)

  bodyAST.root.CopyFrom(bodyRootStmt)


  verticalRegionStmt = SIR_pb2.VerticalRegionDeclStmt()

  verticalRegion.ast.CopyFrom(bodyAST)


def createVerticalRegionStmt(aBlockStmt):
  verticalRegionStmt = SIR_pb2.VerticalRegionDeclStmt()

  verticalRegionStmt.loc.CopyFrom(loc)

  verticalRegion = SIR_pb2.VerticalRegion()

  verticalRegion.loc.CopyFrom(loc)

  interval = SIR_pb2.Interval()
  interval.special_lower_level = SIR_pb2.Interval.Start
  interval.special_upper_level = SIR_pb2.Interval.End
  interval.lower_offset = 0
  interval.upper_offset = 0

  verticalRegion.interval.CopyFrom(interval)

  verticalRegion.loop_order = SIR_pb2.VerticalRegion.Forward

  createBodyAST(verticalRegion)

  verticalRegionStmt.vertical_region.CopyFrom(verticalRegion)

  aBlockStmt.vertical_region_decl_stmt.CopyFrom(verticalRegionStmt)

def createStencilAST(stencil):
  AST = SIR_pb2.AST()
  rootStmt = SIR_pb2.Stmt()

  blockStmt = SIR_pb2.BlockStmt()
  aBlockStmt = blockStmt.statements.add()

  createVerticalRegionStmt(aBlockStmt)

  rootStmt.block_stmt.CopyFrom(blockStmt)

  AST.root.CopyFrom(rootStmt)

  stencil.ast.CopyFrom(AST)


def createStencil(hir):

  stencil = hir.stencils.add()
  stencil.name = "mystencil"
  loc = SIR_pb2.SourceLocation()
  loc.Line = 111
  loc.Column = 111
  stencil.loc.CopyFrom(loc)

  input = stencil.fields.add()
  output = stencil.fields.add()

  input.name = "in"
  loc = SIR_pb2.SourceLocation()
  loc.Line = 111
  loc.Column = 111

  input.loc.CopyFrom(loc)
  input.is_temporary = False
  input.field_dimensions.append(1)
  input.field_dimensions.append(1)
  input.field_dimensions.append(1)

  output.name = "out"
  output.loc.CopyFrom(loc)
  output.is_temporary = False
  output.field_dimensions.append(1)
  output.field_dimensions.append(1)
  output.field_dimensions.append(1)

  createStencilAST(stencil)


hir = SIR.SIR_pb2.HIR()

loc = SIR_pb2.SourceLocation()
loc.Line = 111
loc.Column = 111

createStencil(hir)

T = textwrap.TextWrapper(initial_indent=' '*1, width=120,subsequent_indent=' '*1)
des = deserializer()

#des.visitGlobalVariables(hir.global_variables)

for stencil in hir.stencils:
  des.visitStencil(stencil)

hirstr = hir.SerializeToString()

options = dawn.dawnOptionsCreate()
backend = dawn.dawnOptionsEntryCreateString("cuda".encode('utf-8'))
dawn.dawnOptionsSet(options, "Backend".encode('utf-8'), backend)

tu = dawn.dawnCompile(hirstr, len(hirstr), options)
stencilname = "mystencil"
b_stencilName = stencilname.encode('utf-8')
code = dawn.dawnTranslationUnitGetStencil(tu, b_stencilName)

f = open("copy_example.cpp","w")
f.write(ctypes.c_char_p(code).value.decode("utf-8"))

f.close()
