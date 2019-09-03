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

#ifndef DAWN_SIR_ASTSTRINGIFER_H
#define DAWN_SIR_ASTSTRINGIFER_H

#include "dawn/AST/ASTStringifier.h"
#include "dawn/SIR/ASTFwd.h"
#include "dawn/SIR/ASTVisitor.h"

namespace dawn {
namespace sir {

class StringVisitor : public ast::StringVisitor<SIRASTData>, public ASTVisitor {
public:
  StringVisitor(int initialIndent, bool newLines);
  void visit(const std::shared_ptr<VerticalRegionDeclStmt>& stmt) override;
};

using ASTStringifier = ast::ASTStringifier<SIRASTData, StringVisitor>;

extern inline std::ostream& operator<<(std::ostream& os, const AST& ast) {
  return ast::operator<<<SIRASTData, StringVisitor>(os, ast);
}
extern inline std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Stmt>& expr) {
  return ast::operator<<<SIRASTData, StringVisitor>(os, expr);
}
extern inline std::ostream& operator<<(std::ostream& os, const std::shared_ptr<Expr>& stmt) {
  return ast::operator<<<SIRASTData, StringVisitor>(os, stmt);
}
} // namespace sir
} // namespace dawn

#endif
