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

#include "dawn/Compiler/DiagnosticsEngine.h"

namespace dawn {

void DiagnosticsEngine::report(const DiagnosticsMessage& diag) { queue_.push_back(diag); }

void DiagnosticsEngine::report(DiagnosticsMessage&& diag) { queue_.push_back(std::move(diag)); }

void DiagnosticsEngine::report(const DiagnosticsBuilder& diagBuilder) {
  report(diagBuilder.getMessage(filename_));
}

} // namespace dawn
