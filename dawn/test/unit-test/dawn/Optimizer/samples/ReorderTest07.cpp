#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
  storage field_a0, field_a1, field_a2, field_a3, field_a4, field_a5, field_a6, field_a7;

  Do {
    vertical_region(k_start, k_end) {
      field_a1 = field_a0(i + 1);
      field_a2 = field_a1(i + 1);
      field_a3 = field_a2(i + 1);
      field_a4 = field_a3(i + 1);
      field_a5 = field_a4(i + 1);
      field_a6 = field_a5(i + 1);
      field_a7 = field_a6(i + 1);
    }
  }
};