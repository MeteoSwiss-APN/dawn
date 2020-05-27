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

#include "dawn/Support/Compiler.h"

namespace dawn {

/// @ingroup support
/// @{

/// @fn dawn_unreachable_internal
/// @brief This function calls abort() and prints the optional message to stderr
///
/// Use the llvm_unreachable macro (that adds location info), instead of
/// calling this function directly.
DAWN_ATTRIBUTE_NORETURN void
dawn_unreachable_internal(const char* msg = nullptr, const char* file = nullptr, unsigned line = 0);

/// @macro dawn_unreachable
/// @brief Marks that the current location is not supposed to be reachable
///
/// In !NDEBUG builds, prints the message and location info to stderr. In NDEBUG builds, becomes an
/// optimizer hint that the current location is not supposed to be reachable. On compilers that
/// don't support such hints, prints a reduced message instead.
#ifndef NDEBUG
#define dawn_unreachable(msg) dawn::dawn_unreachable_internal(msg, __FILE__, __LINE__)
#elif defined(DAWN_BUILTIN_UNREACHABLE)
#define dawn_unreachable(msg) DAWN_BUILTIN_UNREACHABLE
#else
#define dawn_unreachable(msg) dawn::dawn_unreachable_internal()
#endif

/// @}

} // namespace dawn
