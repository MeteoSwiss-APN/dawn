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

#ifndef GTCLANG_FRONTEND_GTCLANGASTACTION_H
#define GTCLANG_FRONTEND_GTCLANGASTACTION_H

#include "clang/Frontend/FrontendAction.h"

#include "gtclang/Frontend/GTClangContext.h"
#include <iostream>

namespace gtclang {

class GTClangContext;

/// @brief Frontend action for the GTClang tool
/// @ingroup frontend
class GTClangASTAction : public clang::ASTFrontendAction {
  GTClangContext* context_;

public:
  GTClangASTAction(GTClangContext* context);
  virtual ~GTClangASTAction(){}

  /// @brief Create the AST consumer to read the AST
  virtual std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& compiler,
                                                                llvm::StringRef file) override;

  /// @brief catch the SIR from the ASTConsumer to use it afterwards
  void catchSIR(std::shared_ptr<dawn::SIR> sir);

  std::shared_ptr<dawn::SIR> getSIR() const;

private:
  std::shared_ptr<dawn::SIR> sir_;
};

} // namespace gtclang

#endif
