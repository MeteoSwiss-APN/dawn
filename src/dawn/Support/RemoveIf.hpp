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

// TODO unittest
template <class T, class UnaryPredicate>
void RemoveIf(typename std::list<T>::iterator first, typename std::list<T>::iterator last,
              std::list<T>& cont, UnaryPredicate p) {
  while(first != last) {
    if(p(*first))
      first = cont.erase(first);
    else
      first++;
  }
}

// TODO make this generic
template <class T, class UnaryPredicate>
void RemoveIf(typename std::vector<T>::iterator first, typename std::vector<T>::iterator last,
              std::vector<T>& cont, UnaryPredicate p) {
  while(first != last) {
    if(p(*first))
      first = cont.erase(first);
    else
      first++;
  }
}

} // namespace dawn

#endif
