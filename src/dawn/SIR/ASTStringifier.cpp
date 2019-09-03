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

#include "dawn/SIR/ASTStringifier.h"
#include "dawn/SIR/ASTStmt.h"
#include "dawn/SIR/SIR.h"
#include "dawn/Support/Printing.h"
#include "dawn/Support/StringUtil.h"
#include <sstream>

namespace dawn {
namespace sir {

StringVisitor::StringVisitor(int initialIndent, bool newLines)
    : ast::StringVisitor<SIRASTData>(initialIndent, newLines) {}

void StringVisitor::visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) {
  if(scopeDepth_ == 0)
    ss_ << std::string(curIndent_, ' ');

  ss_ << "vertical-region : ";
  ss_ << *stmt->verticalRegion_->VerticalInterval.get();
  ss_ << " ["
      << (stmt->verticalRegion_->LoopOrder == VerticalRegion::LK_Forward ? "forward" : "backward")
      << "]\n";
  ss_ << ASTStringifier::toString(*stmt->getAST(), curIndent_);
}

} // namespace sir
} // namespace dawn
