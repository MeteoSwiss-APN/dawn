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

#include "dawn/IIR/StatementAccessesPair.h"
#include "dawn/IIR/Accesses.h"
#include "dawn/IIR/StencilFunctionInstantiation.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/SIR/ASTStringifier.h"
#include "dawn/SIR/Statement.h"
#include "dawn/Support/Printing.h"
#include <sstream>

namespace dawn {
namespace iir {

const std::vector<std::unique_ptr<StatementAccessesPair>>&
BlockStatements::getBlockStatements() const {
  return blockStatements_;
}

bool BlockStatements::hasBlockStatements() const { return !blockStatements_.empty(); }

BlockStatements BlockStatements::clone() const {
  BlockStatements blockStatements;

  for(const auto& stmt : blockStatements_) {
    blockStatements.insert(stmt->clone());
  }
  return blockStatements;
}

void BlockStatements::insert(std::unique_ptr<StatementAccessesPair>&& stmt) {
  std::cout << "inser " << blockStatements_.size() << std::endl;
  std::cout << "T" << static_cast<std::shared_ptr<Stmt>>(stmt->getStatement()->ASTStmt)
            << std::endl;
  blockStatements_.push_back(std::move(stmt));
}

} // namespace iir
} // namespace dawn
