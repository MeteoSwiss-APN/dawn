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

#include "dawn/IIR/ASTStmt.h"
#include <memory>

namespace dawn {
namespace iir {
std::unique_ptr<ast::StmtData> IIRStmtData::clone() const {
  return make_unique<IIRStmtData>(*this);
}
} // namespace iir
} // namespace dawn
