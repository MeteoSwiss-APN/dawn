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

#include "gsl/Compiler/DiagnosticsQueue.h"
#include "gsl/Support/STLExtras.h"

namespace gsl {

DiagnosticsQueue::DiagnosticsQueue() : numErrors_(0), numWarnings_(0) {}

void DiagnosticsQueue::push_back(const DiagnosticsMessage& msg) {
  numErrors_ += msg.getDiagKind() == DiagnosticsKind::Error;
  numWarnings_ += msg.getDiagKind() == DiagnosticsKind::Warning;
  queue_.push_back(make_unique<DiagnosticsMessage>(msg));
}

void DiagnosticsQueue::push_back(DiagnosticsMessage&& msg) {
  numErrors_ += msg.getDiagKind() == DiagnosticsKind::Error;
  numWarnings_ += msg.getDiagKind() == DiagnosticsKind::Warning;
  queue_.push_back(make_unique<DiagnosticsMessage>(std::move(msg)));
}

void DiagnosticsQueue::clear() {
  queue_.clear();
  numErrors_ = 0;
  numWarnings_ = 0;
}

} // namespace gsl
