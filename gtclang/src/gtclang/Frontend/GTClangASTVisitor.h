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

#ifndef GTCLANG_FRONTEND_GTCLANGASTVISITOR_H
#define GTCLANG_FRONTEND_GTCLANGASTVISITOR_H

#include "gtclang/Driver/Options.h"
#include "gtclang/Frontend/GlobalVariableParser.h"
#include "gtclang/Frontend/StencilParser.h"
#include "clang/AST/RecursiveASTVisitor.h"

namespace clang {
class Rewriter;
}

namespace gtclang {

class GTClangContext;

/// @brief Do preorder depth-first traversal on the entire Clang AST and visit each node
class GTClangASTVisitor : public clang::RecursiveASTVisitor<GTClangASTVisitor> {
public:
  explicit GTClangASTVisitor(GTClangContext* context);

  /// @name Visitor implementations
  /// @{
  bool VisitCXXRecordDecl(clang::CXXRecordDecl* recordDecl);
  bool VisitDecl(clang::Decl* Decl);
  /// @}

  /// @brief Get the StencilParser
  const StencilParser& getStencilParser() const { return stencilParser_; }

  /// @brief Get the GlobalVariableParser
  const GlobalVariableParser& getGlobalVariableParser() const { return globalVariableParser_; }

  std::unordered_set<clang::VarDecl*>& getIntervalDecls() { return intervalDecls_; }
  const std::unordered_set<clang::VarDecl*>& getIntervalDecls() const { return intervalDecls_; }

private:
  GTClangContext* context_;

  GlobalVariableParser globalVariableParser_;
  StencilParser stencilParser_;
  std::unordered_set<clang::VarDecl*> intervalDecls_;
};

} // namespace gtclang

#endif
