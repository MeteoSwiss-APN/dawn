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

enum class UnstructuredSubdomain { LateralBoundary = 0, Nudging, Interior, Halo, End };

class unstructured_domain {
  using KeyType = std::tuple<::dawn::LocationType, UnstructuredSubdomain, int>;
  std::map<KeyType, int> subdomainToIndex_;

public:
  int operator()(KeyType&& key) const { return subdomainToIndex_.at(key); }
  void set_splitter_index(KeyType&& key, int index) { subdomainToIndex_[key] = index; }
};

} // namespace dawn