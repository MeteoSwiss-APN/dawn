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

#include "dawn/Compiler/DiagnosticsQueue.h"
#include "dawn/Support/STLExtras.h"

namespace dawn {

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

} // namespace dawn
