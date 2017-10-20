#ifndef GTCLANG_SUPPORT_PARSINGHEADER_H
#define GTCLANG_SUPPORT_PARSINGHEADER_H

#include <string>

namespace gtclang {
struct HeaderWriter {
  HeaderWriter() = delete;
  static std::string longheader() {
    return R"(
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
           )";
  }
  static std::string includes() {
    return R"(
           #include "gridtools/clang_dsl.hpp"

           using namespace gridtools::clang;
            )";
  }
};

} // namespace gtclang

#endif
