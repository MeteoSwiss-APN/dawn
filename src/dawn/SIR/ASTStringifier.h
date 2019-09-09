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

namespace dawn {
namespace sir {
//
// TODO refactor_AST: this is TEMPORARY, should be changed in the future to template specialization
//
using ASTStringifier = ast::ASTStringifier;
extern inline std::ostream& operator<<(std::ostream& os, const AST& ast) {
  return ast::operator<<(os, ast);
}
extern inline std::ostream& operator<<(std::ostream& os, const std::shared_ptr<sir::Stmt>& expr) {
  return ast::operator<<(os, expr);
}
extern inline std::ostream& operator<<(std::ostream& os, const std::shared_ptr<sir::Expr>& stmt) {
  return ast::operator<<(os, stmt);
}
} // namespace sir
} // namespace dawn

#endif
