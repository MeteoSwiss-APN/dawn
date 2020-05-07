//===--------------------------------------------------------------------------------*- C++ -*-===//
//                         _       _
//                        | |     | |
//                    __ _| |_ ___| | __ _ _ __   __ _
//                   / _` | __/ __| |/ _` | '_ \ / _` |
//                  | (_| | || (__| | (_| | | | | (_| |
//                   \__, |\__\___|_|\__,_|_| |_|\__, | - GridTools Clang DSL
//                    __/ |                       __/ |
//                   |___/                       |___/
//
//
//  This file is distributed under the MIT License (MIT).
//  See LICENSE.txt for details.
//
//===------------------------------------------------------------------------------------------===//

#include "dawn/Support/Logger.h"

#ifndef GTCLANG_SUPPORT_LOGGER_H
#define GTCLANG_SUPPORT_LOGGER_H

namespace gtclang {

/// @brief Make a Logger::Formatter for GTClang
dawn::Logger::Formatter makeGTClangFormatter(const std::string& prefix);

} // namespace gtclang

#endif
