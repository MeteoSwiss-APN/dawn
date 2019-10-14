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

#include "dawn/SIR/ASTStmt.h"
#include <memory>

namespace dawn {
namespace sir {
std::unique_ptr<ast::StmtData> SIRStmtData::clone() const {
  return std::make_unique<SIRStmtData>(*this);
}
bool SIRStmtData::equals(ast::StmtData const* other) const {
  return getDataType() == other->getDataType();
}
} // namespace sir
} // namespace dawn
