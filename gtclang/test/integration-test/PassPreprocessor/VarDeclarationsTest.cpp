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

stencil_function bar {
  storage a, b;
  void Do() {
    var d = 10;
    a = b + d;
  }
};

stencil Test01 {
  storage foo;
  var a;

  void Do() {
    vertical_region(k_start, k_end) {
      var b = foo; //  EXPECTED: %line+5%: b = foo;
      var c = 10;  //  EXPECTED: %line+5%: c = 10;
    }
    vertical_region(k_start, k_end) {
      a = foo;
      var e = foo;
      var d = 10;
      foo = e + d;
    }
    vertical_region(k_start, k_end) {
      var b = foo;
      b = a;
      foo = b[i - 1] + b[i + 1];
    }
    vertical_region(k_start, k_end) {
      var temp = foo; //  EXPECTED: %line+5%: temp = foo;
      bar(foo, a);
    }
  }
};
