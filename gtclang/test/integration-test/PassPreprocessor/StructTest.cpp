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

// RUN: %gtclang% %file% -fno-codegen -freport-pass-preprocessor

#include "gtclang_dsl_defs/gtclang_dsl.hpp"

using namespace gtclang::dsl;

stencil_function TestFun { // EXPECTED: %line%: struct TestFun : public gtclang::dsl::stencil_function { using gtclang::dsl::stencil_function::stencil_function;

  storage field_a0, field_a1;

  void Do() {
    field_a1 = field_a0;
  }
};

stencil Test1 { // EXPECTED: %line%: struct Test1 : public gtclang::dsl::stencil { using gtclang::dsl::stencil::stencil;

  storage field_a0, field_a1;

  void Do() {
    for(auto k : {k_end, k_start})
      field_a1 = field_a0;
  }
};

int main() {}
