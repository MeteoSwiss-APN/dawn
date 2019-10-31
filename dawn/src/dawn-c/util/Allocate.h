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

#ifndef DAWN_C_UTIL_ALLOCATE_H
#define DAWN_C_UTIL_ALLOCATE_H

#include "dawn-c/ErrorHandling.h"
#include <cstdlib>
#include <cstring>

namespace dawn {

namespace util {

/// @brief Allocate memory for type `T` via malloc
/// @ingroup dawn_c_util
template <class T>
T* allocate() noexcept {
  T* data = (T*)std::malloc(sizeof(T));
  if(!data)
    dawnFatalError("out of memory");
  std::memset(data, 0, sizeof(T)); // NOLINT(clang-analyzer-core.NonNullParamChecker)
  return data;
}

/// @brief Allocate memory for array of size `n` of type `T` via malloc
/// @ingroup dawn_c_util
template <class T>
T* allocate(std::size_t n) noexcept {
  T* data = (T*)std::malloc(n * sizeof(T));
  if(!data)
    dawnFatalError("out of memory");
  std::memset(data, 0, sizeof(T) * n); // NOLINT(clang-analyzer-core.NonNullParamChecker)
  return data;
}

/// @brief Copy string into `char*` buffer
/// @ingroup dawn_c_util
template <class StringType>
inline char* allocateAndCopyString(StringType&& str) {
  char* buffer = NULL;
  buffer = (char*)std::malloc((str.size() + 1) * sizeof(char));
  if(!buffer)
    dawnFatalError("out of memory");
  // NOLINTNEXTLINE(clang-analyzer-core.NonNullParamChecker)
  std::memcpy(buffer, str.c_str(), str.size() + 1);
  return buffer;
}

} // namespace util

} // namespace dawn

#endif
