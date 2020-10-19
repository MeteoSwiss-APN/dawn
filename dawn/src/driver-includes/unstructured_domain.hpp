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

#include <map>

#include "unstructured_interface.hpp"

namespace dawn {

enum class UnstructuredIterationSpace { LateralBoundary = 0, Nudging, Interior, Halo, End };

class unstructured_domain {
  using KeyType = std::tuple<::dawn::LocationType, UnstructuredIterationSpace, int>;
  std::map<KeyType, int> iterationSpaceToIndex_;

public:
  int operator()(KeyType&& key) const { return iterationSpaceToIndex_.at(key); }
  void set_splitter_index(KeyType&& key, int index) { iterationSpaceToIndex_[key] = index; }
};

} // namespace dawn