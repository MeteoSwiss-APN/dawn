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

#ifndef GTCLANG_FRONTEND_CLANGASTSTMTRESOLVER_H
#define GTCLANG_FRONTEND_CLANGASTSTMTRESOLVER_H

#include "dawn/SIR/ASTFwd.h"
#include "clang/AST/ASTFwd.h"
#include "llvm/ADT/ArrayRef.h"
#include <memory>
#include <vector>

namespace gtclang {

class GTClangContext;
class StencilParser;
class ClangASTExprResolver;

/// @brief Resolve Clang AST Stmts and convert them to SIR AST nodes
/// @ingroup frontend
class ClangASTStmtResolver {
public:
  /// @brief Kind of stencil
  ///
  /// We need to distinguish between parsing the AST of a vertical region or stencil function body
  /// (`StencilBody`) and the description AST of the whole stencil (`StencilDesc`). In the latter
  /// case we need to be able to parse VerticalRegions (given as `CXXForRangeStmt`) and calls to
  /// other stencils while in the former such statements are disallowed.
  enum ASTKind {
    AK_Unknown = 0,
    AK_StencilDesc, ///< Stencil description AST allows parsing of vertical region and stencil calls
    AK_StencilBody  ///< Stencil body ASTs
  };

  ClangASTStmtResolver(GTClangContext* context, StencilParser* parser);

  /// @brief Resolve a single statment into (possibly multiple) SIR Stmts
  llvm::ArrayRef<std::shared_ptr<dawn::Stmt>> resolveStmt(clang::Stmt* stmt, ASTKind kind);

  std::vector<std::shared_ptr<dawn::Stmt>>& getStatements();
  const std::vector<std::shared_ptr<dawn::Stmt>>& getStatements() const;

private:
  //===----------------------------------------------------------------------------------------===//
  //     Internal statment resolver

  void resolve(clang::Stmt* stmt);
  void resolve(clang::BinaryOperator* expr);
  void resolve(clang::CXXOperatorCallExpr* expr);
  void resolve(clang::CXXConstructExpr* expr);
  void resolve(clang::CXXFunctionalCastExpr* expr);
  void resolve(clang::CXXForRangeStmt* expr);
  void resolve(clang::DeclRefExpr* expr);
  void resolve(clang::UnaryOperator* expr);
  void resolve(clang::DeclStmt* stmt);
  void resolve(clang::IfStmt* stmt);
  void resolve(clang::NullStmt* stmt);
  void resolve(clang::ReturnStmt* stmt);

  void resetInternals();

  /// @brief Used internally to parse the body of an If-statement
  ClangASTStmtResolver(const std::shared_ptr<ClangASTExprResolver>& resolver);

private:
  std::shared_ptr<ClangASTExprResolver> clangASTExprResolver_;

  // State variables
  std::vector<std::shared_ptr<dawn::Stmt>> statements_;
  ASTKind AstKind_;
};

} // namespace gtclang

#endif
