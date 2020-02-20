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

#include "PassPreprocessScalarIf.h"
#include "dawn/IIR/ASTExpr.h"
#include "dawn/IIR/ASTStmt.h"
#include "dawn/IIR/DoMethod.h"
#include "dawn/IIR/StencilInstantiation.h"
#include "dawn/IIR/StencilMetaInformation.h"

namespace dawn {
namespace {
bool isIfConditionScalar(const iir::IfStmt& ifStmt) {
  // TODO implement
}

std::vector<std::shared_ptr<iir::Stmt>> computeReplacements(const iir::IfStmt& ifStmt) {
  // TODO implement
}

} // namespace

bool PassPreprocessScalarIf::run(
    const std::shared_ptr<iir::StencilInstantiation>& stencilInstantiation) {

  for(const auto& doMethod : iterateIIROver<iir::DoMethod>(*stencilInstantiation->getIIR())) {
    auto& block = doMethod->getAST();

    for(auto stmtIt = block.getStatements().begin(); stmtIt != block.getStatements().end();) {
      if((*stmtIt)->getKind() == ast::Stmt::Kind::IfStmt) {
        if(isIfConditionScalar(*std::dynamic_pointer_cast<iir::IfStmt>(*stmtIt))) {
          const std::vector<std::shared_ptr<iir::Stmt>> replacements =
              computeReplacements(*std::dynamic_pointer_cast<iir::IfStmt>(*stmtIt));
          stmtIt = block.erase(stmtIt);
          stmtIt = block.insert(stmtIt, replacements.begin(), replacements.end());
          std::advance(stmtIt, replacements.size());
          continue;
        }
      }
      ++stmtIt;
    }
  }

  return true;
}

} // namespace dawn
