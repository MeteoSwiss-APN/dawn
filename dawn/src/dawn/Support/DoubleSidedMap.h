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

#include "dawn/Support/Assert.h"
#include <unordered_map>

namespace dawn {

template <typename Key1, typename Key2>
class DoubleSidedMap {
  std::unordered_map<Key1, Key2> directMap_;
  std::unordered_map<Key2, Key1> reverseMap_;

public:
  /// Add a new element to the map. This function asserts that neither key1 nor key2 is already
  /// present. That is, one to one relation (bijective mapping) is maintained
  void add(const Key1& key1, const Key2& key2) {
    DAWN_ASSERT(!directHas(key1));
    DAWN_ASSERT(!reverseHas(key2));

    directMap_.emplace(key1, key2);
    reverseMap_.emplace(key2, key1);
  }

  void directEraseKey(const Key1& key1) {
    const auto& key2 = directAt(key1);
    reverseMap_.erase(key2); // order matters as key2 references elem of directMap_
    directMap_.erase(key1);
  }

  void reverseEraseKey(const Key2& key2) {
    const auto& key1 = reverseAt(key2); // order matters as key1 references elem of reverseMap_
    directMap_.erase(key1);
    reverseMap_.erase(key2);
  }

  const Key2& directAt(const Key1& key1) const { return directMap_.at(key1); }
  const Key1& reverseAt(const Key2& key2) const { return reverseMap_.at(key2); }

  bool directHas(const Key1& key1) const { return directMap_.count(key1); }
  bool reverseHas(const Key2& key2) const { return reverseMap_.count(key2); }

  const std::unordered_map<Key1, Key2>& getDirectMap() const { return directMap_; }
  const std::unordered_map<Key2, Key1>& getReverseMap() const { return reverseMap_; }
};
} // namespace dawn
