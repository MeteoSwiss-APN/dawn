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
#include <unordered_map>

namespace dawn {
namespace support {

template <typename Key, typename Value>
std::map<Key, Value> orderMap(const std::unordered_map<Key, Value>& umap) {
  std::map<Key, Value> m;
  for(const auto& f : umap)
    m.insert(f);

  return m;
}

} // namespace support
} // namespace dawn
