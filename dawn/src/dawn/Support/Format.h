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
//
// This file includes the headers of the tinyformat library.
// See: https://github.com/c42f/tinyformat
//
//===------------------------------------------------------------------------------------------===//

#ifndef DAWN_SUPPORT_FORMAT_H
#define DAWN_SUPPORT_FORMAT_H

#include "dawn/Support/Assert.h"

#define TINYFORMAT_ERROR(reason) DAWN_ASSERT_MSG(0, reason)
#define TINYFORMAT_USE_VARIADIC_TEMPLATES
#include <tinyformat.h>

namespace dawn {

/// @fn format
/// @brief Format list of arguments according to the given format string and return the result as a
/// string
///
/// Signature:
/// @code
///   template<typename... Args>
///   std::string format(const char* fmt, const Args&... args);
/// @endcode
///
/// @see https://github.com/c42f/tinyformat
/// @ingroup support
using tfm::format;

} // namespace dawn

#endif
