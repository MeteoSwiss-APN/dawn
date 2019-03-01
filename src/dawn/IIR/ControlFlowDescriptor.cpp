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
#include "ControlFlowDescriptor.h"

namespace dawn {
namespace iir {

void ControlFlowDescriptor::insertStmt(std::shared_ptr<Statement>&& statment) {
  controlFlowStatements_.push_back(statment);
}

} // namespace iir
} // namespace dawn
