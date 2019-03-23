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
#ifndef DAWN_IIR_CONTROLFLOWDESCRIPTOR_H
#define DAWN_IIR_CONTROLFLOWDESCRIPTOR_H

#include "dawn/SIR/Statement.h"

namespace dawn {
namespace iir {
class ControlFlowDescriptor {
public:
  using StmtContainer = std::vector<std::shared_ptr<Statement>>;
  using StmtIterator = std::vector<std::shared_ptr<Statement>>::iterator;
  using StmtConstIterator = std::vector<std::shared_ptr<Statement>>::const_iterator;

private:
  /// The control flow statements. These are built from the StencilDescAst of the sir::Stencil
  StmtContainer controlFlowStatements_;

public:
  /// @brief Get the stencil description AST
  const StmtContainer& getStatements() const { return controlFlowStatements_; }

  StmtConstIterator eraseStmt(StmtConstIterator it) { return controlFlowStatements_.erase(it); }
  void insertStmt(const std::shared_ptr<Statement>& stmt) {
    controlFlowStatements_.push_back(stmt);
  }

  //  // TODO do not have a non const
  //  /// @brief Get the stencil description AST
  //  std::vector<std::shared_ptr<Statement>>& getStatements() { return controlFlowStatements_; }

  void insertStmt(std::shared_ptr<Statement>&& statment);
};
}
}

#endif
