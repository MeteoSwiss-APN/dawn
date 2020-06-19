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

#include "dawn/AST/AST.h"
#include "dawn/SIR/ASTStmt.h"

namespace dawn {
namespace sir {
inline std::shared_ptr<ast::AST> makeAST() {
  return std::make_shared<ast::AST>(std::make_unique<SIRStmtData>());
}
//
// TODO refactor_AST: this is TEMPORARY, will be removed in the future
//
using AST = ast::AST;
} // namespace sir
} // namespace dawn
