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

#ifndef GTCLANG_FRONTEND_INCLUDEPROCESSOR_H
#define GTCLANG_FRONTEND_INCLUDEPROCESSOR_H

#include "clang/Frontend/FrontendActions.h"
#include "llvm/ADT/StringRef.h"
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace gtclang {

/// @brief Ensures that input file has necessary gtclang includes needed for stencil DSL
/// @ingroup frontend
class GTClangIncludeChecker {
public:
  static std::string UpdateHeader(const std::string& sourceFile);
};

} // namespace gtclang

#endif
