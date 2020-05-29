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

#ifndef GTCLANG_FRONTEND_CLANGASTEXPRRESOLVER_H
#define GTCLANG_FRONTEND_CLANGASTEXPRRESOLVER_H

#include "dawn/SIR/ASTFwd.h"
#include "dawn/Support/SourceLocation.h"
#include "dawn/Support/Type.h"
#include "gtclang/Frontend/Diagnostics.h"
#include "clang/AST/ASTFwd.h"
#include <memory>
#include <string>
#include <vector>

namespace gtclang {

class GTClangContext;
class StencilParser;
class FunctionResolver;

/// @brief Resolve Clang AST Expr, Stmts and Decls and convert them into SIR AST nodes
/// @ingroup frontend
class ClangASTExprResolver {
  GTClangContext* context_;
  StencilParser* parser_;

  // State variables
  clang::CStyleCastExpr* currentCStyleCastExpr_;
  std::shared_ptr<FunctionResolver> functionResolver_;

public:
  ClangASTExprResolver(GTClangContext* context, StencilParser* parser);

  /// @brief Parse `LHS = ...` where LHS is a reference to a variable declaration
  std::shared_ptr<dawn::sir::Stmt> resolveExpr(clang::BinaryOperator* expr);

  /// @brief Parse `LHS = ...` where LHS is a storage
  std::shared_ptr<dawn::sir::Stmt> resolveExpr(clang::CXXOperatorCallExpr* expr);

  /// @brief Parse call to a stencil function (e.g `avg(u)`)
  std::shared_ptr<dawn::sir::Stmt> resolveExpr(clang::CXXConstructExpr* expr);

  /// @brief Parse call to a stencil function (e.g `avg(i+1)`)
  std::shared_ptr<dawn::sir::Stmt> resolveExpr(clang::CXXFunctionalCastExpr* expr);

  /// @brief Parse a single literal
  /// @{
  std::shared_ptr<dawn::sir::Stmt> resolveExpr(clang::FloatingLiteral* expr);
  std::shared_ptr<dawn::sir::Stmt> resolveExpr(clang::IntegerLiteral* expr);
  std::shared_ptr<dawn::sir::Stmt> resolveExpr(clang::CXXBoolLiteralExpr* expr);
  /// @}

  /// @brief Parse `var;` where var is an unused result
  std::shared_ptr<dawn::sir::Stmt> resolveExpr(clang::DeclRefExpr* expr);

  /// @brief Parse `var` where var is a member access needed for parsing of `if(var) { ... }` where
  /// var is a global variable
  std::shared_ptr<dawn::sir::Stmt> resolveExpr(clang::MemberExpr* expr);

  /// @brief Parse `+/-var;` where var is an unused result
  std::shared_ptr<dawn::sir::Stmt> resolveExpr(clang::UnaryOperator* expr);

  /// @brief Parse `return ...`
  std::shared_ptr<dawn::sir::Stmt> resolveStmt(clang::ReturnStmt* stmt);

  /// @brief Parse `LHS = ...` where LHS is a variable declaration
  std::shared_ptr<dawn::sir::Stmt> resolveDecl(clang::VarDecl* decl);

  /// @brief Get the stencil parser
  const StencilParser* getParser() const { return parser_; }
  StencilParser* getParser() { return parser_; }

  /// @brief Get the context
  const GTClangContext* getContext() const { return context_; }
  GTClangContext* getContext() { return context_; }

  /// @brief Report a daignostics
  clang::DiagnosticBuilder reportDiagnostic(clang::SourceLocation loc, Diagnostics::DiagKind kind);

  dawn::SourceLocation getSourceLocation(clang::Stmt* expr) const;
  dawn::SourceLocation getSourceLocation(clang::Decl* decl) const;

private:
  //===----------------------------------------------------------------------------------------===//
  //     Internal expression resolver
  //===----------------------------------------------------------------------------------------===//
  std::shared_ptr<dawn::sir::Expr> resolve(clang::Expr* expr);
  std::shared_ptr<dawn::sir::Expr> resolve(clang::ArraySubscriptExpr* expr);
  std::shared_ptr<dawn::sir::Expr> resolve(clang::BinaryOperator* expr);
  std::shared_ptr<dawn::sir::Expr> resolve(clang::CallExpr* expr);
  std::shared_ptr<dawn::sir::Expr> resolve(clang::CStyleCastExpr* expr);
  std::shared_ptr<dawn::sir::Expr> resolve(clang::CXXBoolLiteralExpr* expr);
  std::shared_ptr<dawn::sir::Expr> resolve(clang::CXXOperatorCallExpr* expr);
  std::shared_ptr<dawn::sir::Expr> resolve(clang::CXXConstructExpr* expr);
  std::shared_ptr<dawn::sir::Expr> resolve(clang::CXXMemberCallExpr* expr);
  std::shared_ptr<dawn::sir::Expr> resolve(clang::CXXFunctionalCastExpr* expr);
  std::shared_ptr<dawn::sir::Expr> resolve(clang::ConditionalOperator* expr);
  std::shared_ptr<dawn::sir::Expr> resolve(clang::DeclRefExpr* expr);
  std::shared_ptr<dawn::sir::Expr> resolve(clang::FloatingLiteral* expr);
  std::shared_ptr<dawn::sir::Expr> resolve(clang::IntegerLiteral* expr);
  std::shared_ptr<dawn::sir::Expr> resolve(clang::MemberExpr* expr);
  std::shared_ptr<dawn::sir::Expr> resolve(clang::ParenExpr* expr);
  std::shared_ptr<dawn::sir::Expr> resolve(clang::UnaryOperator* expr);

  std::shared_ptr<dawn::sir::Expr> resolveAssignmentOp(clang::Expr* expr);

  dawn::BuiltinTypeID resolveBuiltinType(clang::Expr* expr);

  void resetInternals();
};

} // namespace gtclang

#endif
