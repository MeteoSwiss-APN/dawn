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

#include "LocationType.h"

#include "dawn/Support/Assert.h"

namespace dawn {
namespace ast {

struct UnstructuredIterationSpace {
  NeighborChain Chain;
  bool IncludeCenter = false;

  UnstructuredIterationSpace(std::vector<LocationType>&& chain) : Chain(chain) {}
  UnstructuredIterationSpace(std::vector<LocationType>&& chain, bool includeCenter)
      : Chain(chain), IncludeCenter(includeCenter) {
    if(includeCenter)
      DAWN_ASSERT_MSG(chain.front() == chain.back(),
                      "including center is only allowed if the end "
                      "location is the same as the starting location");
  }
  operator std::vector<LocationType>() const { return Chain; }

  bool chainIsValid() const {
    for(int chainIdx = 0; chainIdx < Chain.size() - 1; chainIdx++) {
      if(Chain[chainIdx] == Chain[chainIdx + 1]) {
        return false;
      }
    }
    return true;
  }

  bool operator==(const UnstructuredIterationSpace& other) const {
    return Chain == other.Chain && IncludeCenter == other.IncludeCenter;
  }
};

} // namespace ast
} // namespace dawn