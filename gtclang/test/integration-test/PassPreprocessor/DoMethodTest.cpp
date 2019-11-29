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

stencil Test01 {
  storage foo;

  Do { // EXPECTED: %line%: void Do \(\) \{
    for(auto k : {k_end, k_start})
      foo = foo;
  }
};

stencil Test02 {
  storage foo;

  Do() { // EXPECTED: %line%: void Do\(\) \{
    for(auto k : {k_end, k_start})
      foo = foo;
  }
};

stencil_function TestFun01 {
  storage foo;

  Do { // EXPECTED: %line%: void Do \(\) \{
    foo = foo;
  }
};

stencil_function TestFun02 {
  storage foo;

  Do { // EXPECTED: %line%: double Do \(\) \{
    return foo;
  }
};

stencil_function TestFun03 {
  storage foo;

  Do() { // EXPECTED: %line%: void Do\(\) \{
    foo = foo;
  }
};

stencil_function TestFun04 {
  storage foo;

  Do() { // EXPECTED: %line%: double Do\(\) \{
    return foo;
  }
};

stencil_function TestFun05 {
  storage foo;

  void Do(k_from = k_start, k_to = k_end) { // EXPECTED: %line%: void Do\(interval0 k_from = k_start, interval0 k_to = k_end\) \{
    foo = foo;
  }
};

stencil_function TestFun06 {
  storage foo;

  Do(k_from = k_start, k_to = k_end) { // EXPECTED: %line%: void Do\(interval0 k_from = k_start, interval0 k_to = k_end\) \{
    foo = foo;
  }
};

stencil_function TestFun07 {
  storage foo;

  Do(k_from = k_start, k_to = k_end) { // EXPECTED: %line%: double Do\(interval0 k_from = k_start, interval0 k_to = k_end\) \{
    return foo;
  }
};

stencil_function TestFun08 {
  storage foo;

  Do() {} // EXPECTED: %line%: void Do\(\) {}
};

stencil_function TestFun09 {
  storage foo;

  Do {} // EXPECTED: %line%: void Do \(\) {}
};

int main() {}
