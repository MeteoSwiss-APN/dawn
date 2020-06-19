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
#pragma once

#include "dawn/IIR/ASTStmt.h"
#include <set>

namespace dawn {
namespace iir {
class StencilMetaInformation;

class ControlFlowDescriptor {
public:
  using StmtContainer = std::vector<std::shared_ptr<iir::Stmt>>;
  using StmtIterator = std::vector<std::shared_ptr<iir::Stmt>>::iterator;
  using StmtConstIterator = std::vector<std::shared_ptr<iir::Stmt>>::const_iterator;

private:
  /// The control flow statements. These are built from the StencilDescAst of the sir::Stencil
  StmtContainer controlFlowStatements_;

public:
  // deep clone
  ControlFlowDescriptor clone() const;

  /// @brief Get the stencil description AST
  const StmtContainer& getStatements() const { return controlFlowStatements_; }
  StmtContainer& getStatements() { return controlFlowStatements_; }

  StmtConstIterator eraseStmt(StmtConstIterator it) { return controlFlowStatements_.erase(it); }

  template <typename TStmt>
  void insertStmt(TStmt&& stmt) {
    controlFlowStatements_.push_back(std::forward<TStmt>(stmt));
  }

  void removeStencilCalls(const std::set<int>& emptyStencilIDs,
                          iir::StencilMetaInformation& metadata);
};
} // namespace iir
} // namespace dawn
