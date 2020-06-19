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

#include "gtclang/Support/SmallVector.h"

namespace gtclang {

void SmallVectorBase::grow_pod(void* FirstEl, size_t MinSizeInBytes, size_t TSize) {
  size_t CurSizeBytes = size_in_bytes();
  size_t NewCapacityInBytes = 2 * capacity_in_bytes() + TSize; // Always grow.
  if(NewCapacityInBytes < MinSizeInBytes)
    NewCapacityInBytes = MinSizeInBytes;

  void* NewElts;
  if(BeginX == FirstEl) {
    NewElts = malloc(NewCapacityInBytes);

    // Copy the elements over.  No need to run dtors on PODs.
    memcpy(NewElts, this->BeginX, CurSizeBytes);
  } else {

    // If this wasn't grown from the inline copy, grow the allocated space.
    NewElts = realloc(this->BeginX, NewCapacityInBytes);
  }
  DAWN_ASSERT_MSG(NewElts, "Out of memory");

  this->EndX = (char*)NewElts + CurSizeBytes;
  this->BeginX = NewElts;
  this->CapacityX = (char*)this->BeginX + NewCapacityInBytes;
}

} // namespace gtclang
