#include "gtclang_dsl_defs/gtclang_dsl.hpp"
using namespace gtclang::dsl;

stencil Test {
  storage field_a0, field_a1, field_a2;
  storage field_b0, field_b1, field_b2;

  Do {
    vertical_region(k_start, k_start) { field_a1 = field_a0(k + 1); }

    vertical_region(k_start + 2, k_end - 1) { field_a2 = field_a1(k + 1); }
  }
};