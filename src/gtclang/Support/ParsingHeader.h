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

#ifndef GTCLANG_SUPPORT_PARSINGHEADER_H
#define GTCLANG_SUPPORT_PARSINGHEADER_H

#include <string>

namespace gtclang {
struct HeaderWriter {
  HeaderWriter() = delete;
  static std::string longheader() {
    return "//"
           "===--------------------------------------------------------------------------------*- "
           "C++ -*-===//\n"
           "//                         _       _\n"
           "//                        | |     | |\n"
           "//                    __ _| |_ ___| | __ _ _ __  \n"
           "//                   / _` | __/ __| |/ _` | '_ \\ / _` |\n"
           "//                  | (_| | || (__| | (_| | | | | (_| |\n"
           "//                   \\__, |\\__\\___|_|\\__,_|_| |_|\\__, | - GridTools Clang DSL\n"
           "//                    __/ |                       __/ |\n"
           "//                   |___/                       |___/\n"
           "//\n"
           "//\n"
           "//  This file is distributed under the MIT License (MIT).\n"
           "//  See LICENSE.txt for details.\n"
           "//\n"
           "//"
           "===------------------------------------------------------------------------------------"
           "------===//\"\n\n";
  }
  static std::string includes() {
    return "#include \"gridtools/clang_dsl.hpp\"\n"
           "\n"
           "using namespace gridtools::clang;\n";
  }
};

} // namespace gtclang

#endif
