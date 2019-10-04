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

#ifndef GTCLANG_FRONTEND_CLANGFORMAT_H
#define GTCLANG_FRONTEND_CLANGFORMAT_H

#include <string>

namespace gtclang {

class GTClangContext;

/// @brief Adaption of the clang-format tool which automatically formats (fragments of) C++ code
/// @ingroup frontend
class ClangFormat {
  GTClangContext* context_;

public:
  ClangFormat(GTClangContext* context);

  /// @brief Run clang-format on the provided code-snippet
  std::string format(const std::string& code);
};

} // namespace gtclang

#endif
