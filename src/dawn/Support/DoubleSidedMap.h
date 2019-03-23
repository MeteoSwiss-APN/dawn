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

#ifndef DAWN_SUPPORT_DOUBLESIDEDMAP_H
#define DAWN_SUPPORT_DOUBLESIDEDMAP_H

#include "dawn/Support/Assert.h"
#include <unordered_map>

namespace dawn {

template <typename Key1, typename Key2>
class DoubleSidedMap {
  std::unordered_map<Key1, Key2> directMap_;
  std::unordered_map<Key2, Key1> reverseMap_;

public:
  void emplace(const Key1& key1, const Key2& key2) {
    // TODO is assert enough, it is name to accessID not replicate entries protected in gtclang ?
    // TODO the emplace can not check for existance, should be a set method instead
    //    DAWN_ASSERT(!directMap_.count(key1));
    //    DAWN_ASSERT(!reverseMap_.count(key2));

    directMap_.emplace(key1, key2);
    reverseMap_.emplace(key2, key1);
  }
  void directEraseKey(const Key1& key1) {
    const auto& key2 = directAt(key1);
    erase(key1, key2);
  }

  void reverseEraseKey(const Key2& key2) {
    const auto& key1 = reverseAt(key2);
    erase(key1, key2);
  }

  const Key2& directAt(const Key1& key1) const { return directMap_.at(key1); }
  const Key1& reverseAt(const Key2& key2) const { return reverseMap_.at(key2); }

  bool directHas(const Key1& key1) const { return directMap_.count(key1); }
  bool reverseHas(const Key2& key2) const { return reverseMap_.count(key2); }

  const std::unordered_map<Key1, Key2>& getDirectMap() const { return directMap_; }
  const std::unordered_map<Key2, Key1>& getReverseMap() const { return reverseMap_; }

private:
  void erase(const Key1& key1, const Key2& key2) {
    directMap_.erase(key1);
    reverseMap_.erase(key2);
  }
};
} // namespace dawn
#endif
