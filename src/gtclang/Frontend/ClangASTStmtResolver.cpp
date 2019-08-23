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

#include "gtclang/Frontend/ClangASTStmtResolver.h"
#include "dawn/SIR/AST.h"
#include "gtclang/Frontend/ClangASTExprResolver.h"
#include "gtclang/Frontend/StencilParser.h"
#include "gtclang/Support/ASTUtils.h"
#include "gtclang/Support/ClangCompat/SourceLocation.h"
#include "clang/AST/AST.h"

namespace gtclang {

ClangASTStmtResolver::ClangASTStmtResolver(GTClangContext* context, StencilParser* parser)
    : clangASTExprResolver_(std::make_shared<ClangASTExprResolver>(context, parser)),
      AstKind_(AK_Unknown) {}

ClangASTStmtResolver::ClangASTStmtResolver(const std::shared_ptr<ClangASTExprResolver>& resolver)
    : clangASTExprResolver_(resolver), AstKind_(AK_Unknown) {}

llvm::ArrayRef<std::shared_ptr<dawn::Stmt>> ClangASTStmtResolver::resolveStmt(clang::Stmt* stmt,
                                                                              ASTKind kind) {
  resetInternals();
  AstKind_ = kind;
  resolve(stmt);
  return llvm::ArrayRef<std::shared_ptr<dawn::Stmt>>(statements_);
}

std::vector<std::shared_ptr<dawn::Stmt>>& ClangASTStmtResolver::getStatements() {
  return statements_;
}

const std::vector<std::shared_ptr<dawn::Stmt>>& ClangASTStmtResolver::getStatements() const {
  return statements_;
}

//===------------------------------------------------------------------------------------------===//
//     Internal statment resolver

void ClangASTStmtResolver::resolve(clang::Stmt* stmt) {
  using namespace clang;
  // skip implicit nodes
  stmt = skipAllImplicitNodes(stmt);

  if(BinaryOperator* s = dyn_cast<BinaryOperator>(stmt))
    resolve(s);
  else if(CXXOperatorCallExpr* s = dyn_cast<CXXOperatorCallExpr>(stmt))
    resolve(s);
  else if(CXXConstructExpr* s = dyn_cast<CXXConstructExpr>(stmt))
    resolve(s);
  else if(CXXFunctionalCastExpr* s = dyn_cast<CXXFunctionalCastExpr>(stmt))
    resolve(s);
  else if(CXXForRangeStmt* s = dyn_cast<CXXForRangeStmt>(stmt)) {
    if(AstKind_ == AK_StencilDesc)
      resolve(s);
    else
      clangASTExprResolver_->getParser()->reportDiagnostic(
          clang_compat::getBeginLoc(*stmt), Diagnostics::err_do_method_nested_vertical_region);
  } else if(DeclStmt* s = dyn_cast<DeclStmt>(stmt))
    resolve(s);
  else if(DeclRefExpr* s = dyn_cast<DeclRefExpr>(stmt))
    resolve(s);
  else if(ReturnStmt* s = dyn_cast<ReturnStmt>(stmt))
    resolve(s);
  else if(IfStmt* s = dyn_cast<IfStmt>(stmt))
    resolve(s);
  else if(NullStmt* s = dyn_cast<NullStmt>(stmt))
    resolve(s);
  else if(UnaryOperator* s = dyn_cast<UnaryOperator>(stmt))
    resolve(s);
  else {
    stmt->dumpColor();
    DAWN_ASSERT_MSG(0, "unresolved statement");
  }
}

void ClangASTStmtResolver::resolve(clang::BinaryOperator* expr) {
  // LHS is a variable (e.g `a = ...`)
  statements_.emplace_back(clangASTExprResolver_->resolveExpr(expr));
}

void ClangASTStmtResolver::resolve(clang::CXXOperatorCallExpr* expr) {
  // LHS is a storage (e.g `u = ..` or `u(i, j) = ...`)
  statements_.emplace_back(clangASTExprResolver_->resolveExpr(expr));
}

void ClangASTStmtResolver::resolve(clang::CXXConstructExpr* expr) {
  if(AstKind_ == AK_StencilBody) {
    // Call to a stencil function (e.g `avg(u)`)
    statements_.emplace_back(clangASTExprResolver_->resolveExpr(expr));
  } else {
    // Call to a another stencil
    statements_.emplace_back(clangASTExprResolver_->getParser()->parseStencilCall(expr));
  }
}

void ClangASTStmtResolver::resolve(clang::CXXFunctionalCastExpr* expr) {
  // Call to a stencil function (e.g `avg(i+1)`)
  statements_.emplace_back(clangASTExprResolver_->resolveExpr(expr));
}

void ClangASTStmtResolver::resolve(clang::CXXForRangeStmt* expr) {
  // Parse a vertial region (i.e `for( ... ) { ... }`)
  DAWN_ASSERT(AstKind_ == AK_StencilDesc);
  statements_.emplace_back(clangASTExprResolver_->getParser()->parseVerticalRegion(expr));
}

void ClangASTStmtResolver::resolve(clang::DeclRefExpr* expr) {
  // Access to a local variable `var;` where `var` is an unused result
  statements_.emplace_back(clangASTExprResolver_->resolveExpr(expr));
}

void ClangASTStmtResolver::resolve(clang::UnaryOperator* expr) {
  // Access to a local variable `+/-var;` where `var` is an unused result
  statements_.emplace_back(clangASTExprResolver_->resolveExpr(expr));
}

void ClangASTStmtResolver::resolve(clang::DeclStmt* stmt) {
  DAWN_ASSERT_MSG(stmt->isSingleDecl(), "only single declarations are currently supported");

  // LHS is a variable declaration (e.g `double a = ...`)
  statements_.emplace_back(
      clangASTExprResolver_->resolveDecl(clang::dyn_cast<clang::VarDecl>(stmt->getSingleDecl())));
}

void ClangASTStmtResolver::resolve(clang::ReturnStmt* stmt) {
  // Return from a stencil function (e.g `return u(i+1)`)
  statements_.emplace_back(clangASTExprResolver_->resolveStmt(stmt));
}

void ClangASTStmtResolver::resolve(clang::IfStmt* stmt) {
  using namespace clang;
  std::shared_ptr<dawn::Stmt> condStmt = nullptr;

  // We currently don't support expression with variable decls in the condition
  if(stmt->getConditionVariable())
    clangASTExprResolver_->getParser()->reportDiagnostic(
        clang_compat::getBeginLoc(*stmt->getConditionVariable()),
        Diagnostics::DiagKind::err_do_method_invalid_expr_if_cond)
        << stmt->getConditionVariable()->getSourceRange();

  // Parse `cond` in `if(cond)` (Note that we currently don't allow variable declarations (which
  // would ofcourse be valid C++ code) in the condition)
  Expr* clangCond = skipAllImplicitNodes(stmt->getCond());

  if(BinaryOperator* s = dyn_cast<BinaryOperator>(clangCond))
    condStmt = clangASTExprResolver_->resolveExpr(s);
  else if(CXXOperatorCallExpr* s = dyn_cast<CXXOperatorCallExpr>(clangCond))
    condStmt = clangASTExprResolver_->resolveExpr(s);
  else if(CXXConstructExpr* s = dyn_cast<CXXConstructExpr>(clangCond))
    condStmt = clangASTExprResolver_->resolveExpr(s);
  else if(CXXFunctionalCastExpr* s = dyn_cast<CXXFunctionalCastExpr>(clangCond))
    condStmt = clangASTExprResolver_->resolveExpr(s);
  else if(DeclRefExpr* s = dyn_cast<DeclRefExpr>(clangCond))
    condStmt = clangASTExprResolver_->resolveExpr(s);
  else if(FloatingLiteral* s = dyn_cast<FloatingLiteral>(clangCond))
    condStmt = clangASTExprResolver_->resolveExpr(s);
  else if(IntegerLiteral* s = dyn_cast<IntegerLiteral>(clangCond))
    condStmt = clangASTExprResolver_->resolveExpr(s);
  else if(CXXBoolLiteralExpr* s = dyn_cast<CXXBoolLiteralExpr>(clangCond))
    condStmt = clangASTExprResolver_->resolveExpr(s);
  else if(MemberExpr* s = dyn_cast<MemberExpr>(clangCond))
    condStmt = clangASTExprResolver_->resolveExpr(s);
  else {
    clangASTExprResolver_->getParser()->reportDiagnostic(
        clang_compat::getBeginLoc(*clangCond),
        Diagnostics::DiagKind::err_do_method_invalid_expr_if_cond)
        << clangCond->getSourceRange();
  }

  auto parseBody = [&](clang::Stmt* clangStmt) -> std::shared_ptr<dawn::BlockStmt> {
    if(!clangStmt)
      return nullptr;

    auto blockStmt =
        std::make_shared<dawn::BlockStmt>(clangASTExprResolver_->getSourceLocation(clangStmt));
    ClangASTStmtResolver resolver(clangASTExprResolver_);

    if(CompoundStmt* compound = dyn_cast<CompoundStmt>(clangStmt)) {
      for(Stmt* s : compound->body()) {
        blockStmt->insert_back(resolver.resolveStmt(s, AstKind_));
      }
    } else {
      blockStmt->insert_back(resolver.resolveStmt(clangStmt, AstKind_));
    }

    return blockStmt;
  };

  statements_.emplace_back(std::make_shared<dawn::IfStmt>(
      condStmt, parseBody(stmt->getThen()), parseBody(stmt->getElse()),
      clangASTExprResolver_->getSourceLocation(stmt)));
}

void ClangASTStmtResolver::resolve(clang::NullStmt* stmt) {}

void ClangASTStmtResolver::resetInternals() { statements_.clear(); }

} // namespace gtclang
