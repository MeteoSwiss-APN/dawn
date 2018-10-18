//===--------------------------------------------------------------------------------*- C++ -*-===//
//                          _
//                         | |
//                       __| | __ ___      ___ ___
//                      / _` |/ _` \ \ /\ / / '_  |
//                     | (_| | (_| |\ V  V /| | | |
//                      \__,_|\__,_| \_/\_/ |_| |_| - Compiler Toolchain
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef DAWN_SIR_STATEMENT_H
#define DAWN_SIR_STATEMENT_H

#include "dawn/SIR/SIR.h"
#include <memory>
#include <vector>

namespace dawn {

/// @brief Wrapper for AST statements of the SIR which contains additional information
/// @ingroup optimizer
struct Statement {
  Statement(const std::shared_ptr<Stmt>& stmt,
            const std::shared_ptr<std::vector<sir::StencilCall*>>& stackTrace)
      : ASTStmt(stmt), StackTrace(stackTrace) {}

  /// SIR AST statement
  std::shared_ptr<Stmt> ASTStmt;

  /// Stack trace of inlined stencil calls of this statment (might be `NULL`)
  std::shared_ptr<std::vector<sir::StencilCall*>> StackTrace;

  std::shared_ptr<Statement> clone() {
    std::shared_ptr<std::vector<sir::StencilCall*>> clonedStackTrace;
    if(StackTrace) {
      for(const auto call : *StackTrace) {
        clonedStackTrace->emplace_back(call->clone().get());
      }
    } else {
      clonedStackTrace = nullptr;
    }
    std::shared_ptr<Statement> retval =
        std::make_shared<Statement>(ASTStmt->clone(), clonedStackTrace);
    return retval;
  }
};

} // namespace dawn

#endif
