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

extern void assertionFailed(char const* expr, char const* function, char const* file, long line);
extern void assertionFailedMsg(char const* expr, char const* msg, char const* function,
                               char const* file, long line);

} // namespace dawn

#if defined(DAWN_DISABLE_ASSERTS) || defined(NDEBUG)
#define DAWN_USING_ASSERTS 0

#define DAWN_ASSERT(expr) ((void)0)
#define DAWN_ASSERT_MSG(expr, msg) ((void)0)

#else
#define DAWN_USING_ASSERTS 1

/// @macro DAWN_ASSERT
/// @brief Assert macro
/// @ingroup support
#define DAWN_ASSERT(expr)                                                                          \
  (DAWN_BUILTIN_LIKELY(!!(expr))                                                                   \
       ? ((void)0)                                                                                 \
       : dawn::assertionFailed(#expr, DAWN_CURRENT_FUNCTION, __FILE__, __LINE__))

/// @macro DAWN_ASSERT_MSG
/// @brief Assert macro with additional message
/// @ingroup support
#define DAWN_ASSERT_MSG(expr, msg)                                                                 \
  (DAWN_BUILTIN_LIKELY(!!(expr))                                                                   \
       ? ((void)0)                                                                                 \
       : dawn::assertionFailedMsg(#expr, msg, DAWN_CURRENT_FUNCTION, __FILE__, __LINE__))

#endif
