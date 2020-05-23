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
#include <list>

namespace dawn {

//// remove_if (std c++-14)

// remove_if : remove elements from the container and modifies the size of the container
// notice semantic and implementation is different that C++14
// it supports associative containers
// @return true if element is found and removed
template <class Container, class UnaryPredicate>
bool RemoveIf(Container& cont, UnaryPredicate p) {
  bool r = false;

  auto first = cont.begin();
  auto last = cont.end();
  while(first != last) {
    if(p(*first)) {
      first = cont.erase(first);
      last = cont.end();
      r = true;
    } else {
      first++;
    }
  }
  return r;
}

} // namespace dawn



