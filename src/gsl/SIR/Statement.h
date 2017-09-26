//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GSL_SIR_STATEMENT_H
#define GSL_SIR_STATEMENT_H

#include "gsl/SIR/SIR.h"
#include <memory>
#include <vector>

namespace gsl {

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
};

} // namespace gsl

#endif
