//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "gsl/Compiler/DiagnosticsEngine.h"

namespace gsl {

void DiagnosticsEngine::report(const DiagnosticsMessage& diag) { queue_.push_back(diag); }

void DiagnosticsEngine::report(DiagnosticsMessage&& diag) { queue_.push_back(std::move(diag)); }

void DiagnosticsEngine::report(const DiagnosticsBuilder& diagBuilder) {
  report(diagBuilder.getMessage(filename_));
}

} // namespace gsl
