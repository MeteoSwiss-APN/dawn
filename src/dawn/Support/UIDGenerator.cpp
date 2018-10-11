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

#include "dawn/Support/UIDGenerator.h"

namespace dawn {

/* Null, because instance will be initialized on demand. */
UIDGenerator* UIDGenerator::instance_ = 0;

UIDGenerator* UIDGenerator::getInstance() {
  if(instance_ == 0) {
    instance_ = new UIDGenerator();
  }

  return instance_;
}

} // namespace dawn
