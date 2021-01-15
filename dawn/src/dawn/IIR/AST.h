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
#include "dawn/IIR/ASTStmt.h"

namespace dawn {
namespace iir {
inline std::shared_ptr<ast::AST> makeAST() {
  return std::make_shared<ast::AST>(std::make_unique<IIRStmtData>());
}
} // namespace iir
} // namespace dawn
