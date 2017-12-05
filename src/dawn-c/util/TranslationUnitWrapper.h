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

#ifndef DAWN_C_UTIL_TRANSLATIONUNITWRAPPER_H
#define DAWN_C_UTIL_TRANSLATIONUNITWRAPPER_H

#include "dawn-c/ErrorHandling.h"
#include "dawn-c/TranslationUnit.h"
#include "dawn/CodeGen/TranslationUnit.h"

namespace dawn {

namespace util {

/// @brief Convert `dawnTranslationUnit_t` to `TranslationUnit`
/// @ingroup dawn_c_util
/// @{
inline const TranslationUnit* toConstTranslationUnit(const dawnTranslationUnit_t* TU) {
  if(!TU->Impl)
    dawnFatalError("uninitialized TranslationUnit");
  return reinterpret_cast<const TranslationUnit*>(TU->Impl);
}

inline TranslationUnit* toTranslationUnit(dawnTranslationUnit_t* TU) {
  if(!TU->Impl)
    dawnFatalError("uninitialized TranslationUnit");
  return reinterpret_cast<TranslationUnit*>(TU->Impl);
}
/// @}

} // namespace util

} // namespace dawn

#endif
