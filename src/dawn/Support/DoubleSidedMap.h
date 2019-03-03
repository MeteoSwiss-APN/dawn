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
    DAWN_ASSERT(!directMap_.count(key1));
    DAWN_ASSERT(!reverseMap_.count(key2));

    directMap_.emplace(key1, key2);
    reverseMap_.emplace(key2, key1);
  }
  void directEraseKey(const Key1& key) { eraseKeyG(directMap_, reverseMap_, key); }
  void reverseEraseKey(const Key2& key) { eraseKeyG(reverseMap_, directMap_, key); }

  const Key2& directAt(const Key1& key1) const { return directMap_.at(key1); }
  const Key1& reverseAt(const Key2& key2) const { return reverseMap_.at(key2); }

  const std::unordered_map<Key1, Key2>& getDirectMap() const { return directMap_; }
  const std::unordered_map<Key2, Key1>& getReverseMap() const { return reverseMap_; }

private:
  template <typename DMap, typename RMap>
  void eraseKeyG(DMap& dmap, RMap& rmap, const typename DMap::key_type& key) {
    dmap.erase(key);
    bool found = false;
    for(const auto& rPair : rmap) {
      if(rPair.second == key) {
        rmap.erase(rPair.first);
        found = true;
        break;
      }
    }
    DAWN_ASSERT(found);
  }
};
} // namespace dawn
#endif
