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

#ifndef GSL_SUPPORT_UNREACHABLE_H
#define GSL_SUPPORT_UNREACHABLE_H

#include "gsl/Support/Compiler.h"

namespace gsl {

/// @ingroup support
/// @{

/// @fn gsl_unreachable_internal
/// @brief This function calls abort() and prints the optional message to stderr
///
/// Use the llvm_unreachable macro (that adds location info), instead of
/// calling this function directly.
GSL_ATTRIBUTE_NORETURN void gsl_unreachable_internal(const char* msg = nullptr,
                                                     const char* file = nullptr, unsigned line = 0);

/// @macro gsl_unreachable
/// @brief Marks that the current location is not supposed to be reachable
///
/// In !NDEBUG builds, prints the message and location info to stderr. In NDEBUG builds, becomes an
/// optimizer hint that the current location is not supposed to be reachable. On compilers that
/// don't support such hints, prints a reduced message instead.
#ifndef NDEBUG
#define gsl_unreachable(msg) gsl::gsl_unreachable_internal(msg, __FILE__, __LINE__)
#elif defined(GSL_BUILTIN_UNREACHABLE)
#define gsl_unreachable(msg) GSL_BUILTIN_UNREACHABLE
#else
#define gsl_unreachable(msg) gsl::gsl_unreachable_internal()
#endif

/// @}

} // namespace gsl

#endif
