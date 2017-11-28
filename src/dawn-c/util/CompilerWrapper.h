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

#ifndef DAWN_C_UTIL_COMPILERWRAPPER_H
#define DAWN_C_UTIL_COMPILERWRAPPER_H

#include "dawn-c/Compiler.h"
#include "dawn-c/ErrorHandling.h"
#include "dawn/Compiler/DawnCompiler.h"

namespace dawn {

namespace util {

/// @brief Convert `dawnCompiler_t` to `DawnCompiler`
/// @ingroup dawn_c_util
/// @{
inline const DawnCompiler* toConstCompiler(const dawnCompiler_t* compiler) {
  if(!compiler->Impl)
    dawnFatalError("uninitialized Compiler");
  return reinterpret_cast<const DawnCompiler*>(compiler->Impl);
}

inline DawnCompiler* toCompiler(dawnCompiler_t* compiler) {
  if(!compiler->Impl)
    dawnFatalError("uninitialized Compiler");
  return reinterpret_cast<DawnCompiler*>(compiler->Impl);
}
/// @}

} // namespace util

} // namespace dawn

#endif
