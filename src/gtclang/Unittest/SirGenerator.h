//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _
//                        | |     | |
//                    __ _| |_ ___| | __ _ _ __   __ _
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GTCLANG_UNITTEST_SIRGENERATOR
#define GTCLANG_UNITTEST_SIRGENERATOR
#include "dawn/SIR/ASTExpr.h"
#include "dawn/SIR/ASTStmt.h"
#include "dawn/SIR/ASTVisitor.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Array.h"
#include "dawn/Support/Casting.h"
#include "dawn/Support/Unreachable.h"
#include <string>
#include <vector>

namespace gtclang {

/// @brief parsing unittest-strings to automatically write simple stencils
/// @ingroup unittest
/// The class is filled with parse
/// To generate the stencil, we call FileWriter with this class as an argument
class ParsedString {
public:
  ParsedString() = default;
  ParsedString(ParsedString&&) = default;
  ParsedString(const ParsedString&) = default;
  ParsedString(const std::string& functionCall) : functionCall_(functionCall) {}

  const std::vector<std::string>& getFields() const { return fields_; }
  const std::vector<std::string>& getVariables() const { return variables_; }
  const std::string& getCall() const { return functionCall_; }

  template <typename... Args>
  void argumentParsing(ParsedString& p, const std::shared_ptr<dawn::Expr>& argument,
                       Args&&... args) {
    argumentParsingImpl(p, argument);
    argumentParsing(p, std::forward<Args>(args)...);
  }
  void argumentParsing(ParsedString& p) {}

  void argumentParsingImpl(ParsedString& p, const std::shared_ptr<dawn::Expr>& argument);

  void dump();

private:
  void addField(std::string& field) { fields_.push_back(field); }
  void addVariable(std::string& variable) { variables_.push_back(variable); }

  std::vector<std::string> fields_;
  std::vector<std::string> variables_;
  std::string functionCall_;
};

/// @brief parses a string describing an operation with its respective variables
/// @ingroup unittest
/// @param[in] Function call as a string (e.g "a = b + c")
/// @param[in] Declaration of Variables as Fields or variable accesses [dawn::field("a"),
/// dawn::var("b")]
/// @param[out] An object containing all the information to autogenerate the corresponding stencil
/// to a File
template <typename... Args>
ParsedString parse(const std::string& functionCall, Args&&... args) {
  ParsedString parsed(functionCall);
  parsed.argumentParsing(parsed, std::forward<Args>(args)...);
  return parsed;
}

/// @brief simplification for generating SIR in memory
/// @ingroup unittest
/// This group of statements allows for a simplyfied notation to generate in-memory SIRs for testing
/// puropses. It can be used to describe simple operations or blocks of operations in a human
/// readable way like assign(var("a"), binop(var("b"),"+",var("c")))
/// @{
std::shared_ptr<dawn::BlockStmt> block(const std::vector<std::shared_ptr<dawn::Stmt>>& statements);
std::shared_ptr<dawn::ExprStmt> expr(const std::shared_ptr<dawn::Expr>& expr);
std::shared_ptr<dawn::ReturnStmt> ret(const std::shared_ptr<dawn::Expr>& expr);
std::shared_ptr<dawn::VarDeclStmt> vardec(const std::string& type, const std::string& name,
                                          const std::shared_ptr<dawn::Expr> &init, const char* op = "=");
std::shared_ptr<dawn::VarDeclStmt> vecdec(const std::string& type, const std::string& name,
                                          std::vector<std::shared_ptr<dawn::Expr>> initList,
                                          int dimension = 0, const char* op = "=");
std::shared_ptr<dawn::VerticalRegionDeclStmt>
vrdec(const std::shared_ptr<dawn::sir::VerticalRegion>& verticalRegion);
std::shared_ptr<dawn::StencilCallDeclStmt>
scdec(const std::shared_ptr<dawn::sir::StencilCall>& stencilCall);
std::shared_ptr<dawn::BoundaryConditionDeclStmt> bcdec(const std::string& callee);
std::shared_ptr<dawn::IfStmt> ifst(const std::shared_ptr<dawn::Stmt>& condExpr,
                                   const std::shared_ptr<dawn::Stmt>& thenStmt,
                                   const std::shared_ptr<dawn::Stmt>& elseStmt = nullptr);
std::shared_ptr<dawn::UnaryOperator> unop(const std::shared_ptr<dawn::Expr>& operand,
                                          const char* op);
std::shared_ptr<dawn::BinaryOperator> binop(const std::shared_ptr<dawn::Expr>& left, const char* op,
                                            const std::shared_ptr<dawn::Expr>& right);
std::shared_ptr<dawn::AssignmentExpr> assign(const std::shared_ptr<dawn::Expr>& left,
                                             const std::shared_ptr<dawn::Expr>& right, const char *op="=");
std::shared_ptr<dawn::TernaryOperator> ternop(const std::shared_ptr<dawn::Expr>& cond,
                                              const std::shared_ptr<dawn::Expr>& left,
                                              const std::shared_ptr<dawn::Expr>& right);
std::shared_ptr<dawn::FunCallExpr> fcall(const std::string& callee);
std::shared_ptr<dawn::StencilFunCallExpr> sfcall(const std::string& calee);
std::shared_ptr<dawn::StencilFunArgExpr> sfarg(int direction, int offset, int argumentIndex);
std::shared_ptr<dawn::VarAccessExpr> var(const std::string& name,
                                         std::shared_ptr<dawn::Expr> index = nullptr);
std::shared_ptr<dawn::FieldAccessExpr>
field(const std::string& name, dawn::Array3i offset = dawn::Array3i{{0, 0, 0}},
      dawn::Array3i argumentMap = dawn::Array3i{{-1, -1, -1}},
      dawn::Array3i argumentOffset = dawn::Array3i{{0, 0, 0}}, bool negateOffset = false);
std::shared_ptr<dawn::LiteralAccessExpr> lit(const std::string& value,
                                             dawn::BuiltinTypeID builtinType = dawn::BuiltinTypeID::Integer);
/// @}

/// @brief Simple Visitor that prints Statements with their respective Types
/// @ingroup unittest
class PrintAllExpressionTypes : public dawn::ASTVisitorForwarding {
public:
  virtual void visit(const std::shared_ptr<dawn::BlockStmt>& stmt);
  virtual void visit(const std::shared_ptr<dawn::ExprStmt>& stmt);
  virtual void visit(const std::shared_ptr<dawn::ReturnStmt>& stmt);
  virtual void visit(const std::shared_ptr<dawn::VarDeclStmt>& stmt);
  virtual void visit(const std::shared_ptr<dawn::VerticalRegionDeclStmt>& stmt);
  virtual void visit(const std::shared_ptr<dawn::StencilCallDeclStmt>& stmt);
  virtual void visit(const std::shared_ptr<dawn::BoundaryConditionDeclStmt>& stmt);
  virtual void visit(const std::shared_ptr<dawn::IfStmt>& stmt);
  virtual void visit(const std::shared_ptr<dawn::UnaryOperator>& expr);
  virtual void visit(const std::shared_ptr<dawn::BinaryOperator>& expr);
  virtual void visit(const std::shared_ptr<dawn::AssignmentExpr>& expr);
  virtual void visit(const std::shared_ptr<dawn::TernaryOperator>& expr);
  virtual void visit(const std::shared_ptr<dawn::FunCallExpr>& expr);
  virtual void visit(const std::shared_ptr<dawn::StencilFunCallExpr>& expr);
  virtual void visit(const std::shared_ptr<dawn::StencilFunArgExpr>& expr);
  virtual void visit(const std::shared_ptr<dawn::VarAccessExpr>& expr);
  virtual void visit(const std::shared_ptr<dawn::FieldAccessExpr>& expr);
  virtual void visit(const std::shared_ptr<dawn::LiteralAccessExpr>& expr);
};

/// @brief helper to finds all the field to register for auto-generating stencil codes
class FieldFinder : public dawn::ASTVisitorForwarding {
public:
  virtual void visit(const std::shared_ptr<dawn::FieldAccessExpr>& expr);

  virtual void visit(const std::shared_ptr<dawn::VerticalRegionDeclStmt>& stmt);

  const std::vector<std::shared_ptr<dawn::sir::Field>>& getFields() const { return allFields_; }

private:
  std::vector<std::shared_ptr<dawn::sir::Field>> allFields_;
};

class BlockWriter {
public:
  template <typename... Args>
  void recursiveBlock(const std::shared_ptr<dawn::Stmt>& statement, Args&&... args) {
    storage_.push_back(statement);
    recursiveBlock(std::forward<Args>(args)...);
  }

  template <typename... Args>
  void recursiveBlock(const std::shared_ptr<dawn::Expr>& expression, Args&&... args) {
    recursiveBlock(std::make_shared<dawn::ExprStmt>(expression), std::forward<Args>(args)...);
  }

  void recursiveBlock() {}

  template <typename... Args>
  const std::vector<std::shared_ptr<dawn::Stmt>>&
  createVec(const std::shared_ptr<dawn::Stmt>& statement, Args&&... args) {
    recursiveBlock(statement, std::forward<Args>(args)...);
    return storage_;
  }

private:
  std::vector<std::shared_ptr<dawn::Stmt>> storage_;
};

template <typename... Args>
std::shared_ptr<dawn::BlockStmt>
blockMultiple(Args&&... args) {
  BlockWriter bw;
  auto vec = bw.createVec(std::forward<Args>(args)...);
  return std::make_shared<dawn::BlockStmt>(vec);
}

dawn::Type stringToType(const std::string& typestring);

} // namespace gtclang

#endif
