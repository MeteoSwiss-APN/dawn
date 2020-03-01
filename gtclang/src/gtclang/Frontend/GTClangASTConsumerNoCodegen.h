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

#ifndef GTCLANG_GTCLANGASTCONSUMERNOCODEGEN
#define GTCLANG_GTCLANGASTCONSUMERNOCODEGEN

#include "gtclang/Driver/Options.h"
#include "gtclang/Frontend/GTClangASTVisitor.h"
#include "clang/AST/ASTConsumer.h"
#include <memory>

namespace gtclang {

class GTClangContext;
class GTClangASTVisitor;
class GTClangASTAction;

/// @brief Implementation to read ASTs produced by the Clang parser and convert them into SIR
/// @ingroup frontend
class GTClangASTConsumerNoCodegen : public clang::ASTConsumer {
public:
  GTClangASTConsumerNoCodegen(GTClangContext* context, const std::string& file,
                              gtclang::GTClangASTAction* parentAction);

  /// @brief This method translating the AST to SIR and generates code
  virtual void HandleTranslationUnit(clang::ASTContext& ASTContext) override;

private:
  GTClangContext* context_;
  std::string file_;

  std::unique_ptr<GTClangASTVisitor> visitor_;
  GTClangASTAction* parentAction_;
};

} // namespace gtclang

#endif
