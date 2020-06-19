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

#include "dawn/Support/ArrayRef.h"
#include <algorithm>
#include <memory>

namespace dawn {

/// @brief Determine the edit distance between two sequences.
///
/// @param fromArray          The first sequence to compare.
/// @param toArray            The second sequence to compare.
/// @param allowReplacements  Whether to allow element replacements (change one element into
///                           another) as a single operation, rather than as two operations
///                           (an insertion and a removal).
/// @param maxEditDistance    If non-zero, the maximum edit distance that this routine is allowed to
///                           compute. If the edit distance will exceed that maximum, returns @c
///                           MaxEditDistance+1.
///
/// @returns the minimum number of element insertions, removals, or (if @p AllowReplacements is @c
/// true) replacements needed to transform one of the given sequences into the other. If zero,
/// the sequences are identical.
///
/// @see http://en.wikipedia.org/wiki/Levenshtein_distance
/// @ingroup support
///
/// @{
template <typename T>
unsigned computeEditDistance(ArrayRef<T> fromArray, ArrayRef<T> toArray,
                             bool allowReplacements = true, unsigned maxEditDistance = 0) {
  // The algorithm implemented below is the "classic" dynamic-programming algorithm for computing
  // the Levenshtein distance, which is described here:
  //
  //   http://en.wikipedia.org/wiki/Levenshtein_distance
  //
  // Although the algorithm is typically described using an m x n  array, only one row plus one
  // element are used at a time, so this implementation just keeps one vector for the row. To update
  // one entry, only the entries to the left, top, and top-left are needed. The left entry is in
  // Row[x-1], the top entry is what's in Row[x] from the last iteration, and the top-left entry is
  // stored in Previous.
  typename ArrayRef<T>::size_type m = fromArray.size();
  typename ArrayRef<T>::size_type n = toArray.size();

  if(n == 0)
    return m;

  const unsigned SmallBufferSize = 64;
  unsigned SmallBuffer[SmallBufferSize];
  std::unique_ptr<unsigned[]> Allocated;
  unsigned* Row = SmallBuffer;
  if(n + 1 > SmallBufferSize) {
    Row = new unsigned[n + 1];
    Allocated.reset(Row);
  }

  for(unsigned i = 1; i <= n; ++i)
    Row[i] = i;

  for(typename ArrayRef<T>::size_type y = 1; y <= m; ++y) {
    Row[0] = y;
    unsigned BestThisRow = Row[0];

    unsigned Previous = y - 1;
    for(typename ArrayRef<T>::size_type x = 1; x <= n; ++x) {
      int OldRow = Row[x];
      if(allowReplacements) {
        Row[x] = std::min(Previous + (fromArray[y - 1] == toArray[x - 1] ? 0u : 1u),
                          std::min(Row[x - 1], Row[x]) + 1);
      } else {
        if(fromArray[y - 1] == toArray[x - 1])
          Row[x] = Previous;
        else
          Row[x] = std::min(Row[x - 1], Row[x]) + 1;
      }
      Previous = OldRow;
      BestThisRow = std::min(BestThisRow, Row[x]);
    }

    if(maxEditDistance && BestThisRow > maxEditDistance)
      return maxEditDistance + 1;
  }

  unsigned Result = Row[n];
  return Result;
}

inline unsigned computeEditDistance(const std::string& fromStr, const std::string& toStr,
                                    bool allowReplacements = true, unsigned maxEditDistance = 0) {
  return computeEditDistance(makeArrayRef(fromStr), makeArrayRef(toStr), allowReplacements,
                             maxEditDistance);
}

/// @}

} // namespace dawn
