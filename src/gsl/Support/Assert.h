//===--------------------------------------------------------------------------------*- C++ -*-===//
//                                 ____ ____  _
//                                / ___/ ___|| |
//                               | |  _\___ \| |
//                               | |_| |___) | |___
//                                \____|____/|_____| - Generic Stencil Language
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#ifndef GSL_SUPPORT_ASSERT_H
#define GSL_SUPPORT_ASSERT_H

#include "gsl/Support/Compiler.h"

namespace gsl {

extern void assertionFailed(char const* expr, char const* function, char const* file, long line);
extern void assertionFailedMsg(char const* expr, char const* msg, char const* function,
                               char const* file, long line);

} // namespace gsl

#if defined(GSL_DISABLE_ASSERTS) || defined(NDEBUG)

#define GSL_ASSERT(expr) ((void)0)
#define GSL_ASSERT_MSG(expr, msg) ((void)0)

#else

/// @macro GSL_ASSERT
/// @brief Assert macro
/// @ingroup support
#define GSL_ASSERT(expr)                                                                           \
  (GSL_BUILTIN_LIKELY(!!(expr)) ? ((void)0) : gsl::assertionFailed(#expr, GSL_CURRENT_FUNCTION,    \
                                                                   __FILE__, __LINE__))

/// @macro GSL_ASSERT_MSG
/// @brief Assert macro with additional message
/// @ingroup support
#define GSL_ASSERT_MSG(expr, msg)                                                                  \
  (GSL_BUILTIN_LIKELY(!!(expr))                                                                    \
       ? ((void)0)                                                                                 \
       : gsl::assertionFailedMsg(#expr, msg, GSL_CURRENT_FUNCTION, __FILE__, __LINE__))

#endif

#endif
