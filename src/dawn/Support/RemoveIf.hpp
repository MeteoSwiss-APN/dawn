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

#ifndef DAWN_REMOVEIF_H
#define DAWN_REMOVEIF_H
#include <list>

namespace dawn {

// remove_if (std c++-14)
template <class ForwardIt, class UnaryPredicate>
ForwardIt RemoveIf(ForwardIt first, ForwardIt last, UnaryPredicate p) {
  first = std::find_if(first, last, p);
  if(first != last)
    for(ForwardIt i = first; ++i != last;)
      if(!p(*i)) {
        // std::move below, like in C++-14 impl is segfault for unknown reasons
        *first++ = *i; // std::move(*i);
      }
  return first;
}

// remove_if (std unordered_map implementation since c++-14 remove_if is not valid on an associative
// map [due to const key])
// @return true if element is found and removed
template <class K, class V, class UnaryPredicate>
bool RemoveIf(typename std::unordered_map<K, V>::iterator first,
              typename std::unordered_map<K, V>::iterator last, std::unordered_map<K, V>& cont,
              UnaryPredicate p) {
  bool r = false;

  while(first != last) {
    if(p(*first)) {
      first = cont.erase(first);
      r = true;
    } else {
      first++;
    }
  }
  return r;
}

} // namespace dawn

#endif
